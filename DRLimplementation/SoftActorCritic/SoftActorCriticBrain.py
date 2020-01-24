# coding=utf-8

from typing import Tuple

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras

from blocAndTools.buildingbloc import (
    ExperimentSpec, GymPlayground, build_MLP_computation_graph, build_KERAS_MLP_computation_graph,
    learning_rate_scheduler,
    buil_MLP_with_initilizer,
    )
from blocAndTools.rl_vocabulary import rl_name
from blocAndTools.tensorflowbloc import (
    update_nn_weights, get_current_scope_variables,
    get_variable_explicitely_by_graph_key_from,
    )

tf_cv1 = tf.compat.v1  # shortcut
vocab = rl_name()

POLICY_LOG_STD_CAP_MAX = 2
POLICY_LOG_STD_CAP_MIN = -20
NUM_STABILITY_CORRECTION = 1e-6
USE_KERAS_LAYER = False  # expose controle fro unti-test purpose

# (NICE TO HAVE) todo:investigate?? --> kernel initialization effect on agent performance:
POLICY_NN_KERNEL_INIT = None
# POLICY_NN_KERNEL_INIT = tf_cv1.initializers.he_normal()

"""

   .|'''.|            .'|.   .       |               .                   '   ..|'''.|          ||    .    ||
   ||..  '    ...   .||.   .||.     |||      ....  .||.    ...   ... ..    .|'     '  ... ..  ...  .||.  ...    ....
    ''|||.  .|  '|.  ||     ||     |  ||   .|   ''  ||   .|  '|.  ||' ''   ||          ||' ''  ||   ||    ||  .|   ''
  .     '|| ||   ||  ||     ||    .''''|.  ||       ||   ||   ||  ||       '|.      .  ||      ||   ||    ||  ||
  |'....|'   '|..|' .||.    '|.' .|.  .||.  '|...'  '|.'  '|..|' .||.       ''|....'  .||.    .||.  '|.' .||.  '|...'

                                        '||                       ||
                                         || ...  ... ..   ....   ...  .. ...
                                         ||'  ||  ||' '' '' .||   ||   ||  ||
                                         ||    |  ||     .|' ||   ||   ||  ||
                                         '|...'  .||.    '|..'|' .||. .||. ||.


                                                                                                         +--- kban style
"""


def apply_action_bound(policy_pi: tf.Tensor, policy_pi_log_likelihood: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    """
    Apply a invertible squashing function (tanh) to bound the actions sampled from the policy distribution in a
    finite interval (See the Soft Actor-Critic paper, appendice C)

        Sum_i log(1-tanh^2(z_i))

        with z_i = squashed_policy_pi

    In the limit as z_i --> inf, the 1-tanh^2(z_i) goes to 0
    ==>
    this can cause numerical instability that we mitigate by adding a small constant NUM_STABILITY_CORRECTION

    :param policy_pi:
    :param policy_pi_log_likelihood:
    :return: squashed_policy_pi, squashed_policy_pi_log_likelihood
    """
    with tf_cv1.variable_scope(vocab.squashing_fct):
        # (nice to have) todo:implement --> a numericaly stable version : see p8 HW5c Sergey Levine DRL course
        squashed_policy_pi = tf_cv1.tanh(policy_pi)

        # \\\ My bloc \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        num_corr = tf_cv1.reduce_sum(tf_cv1.log(1 - squashed_policy_pi ** 2 + NUM_STABILITY_CORRECTION),
                                     axis=1)
        squashed_policy_pi_log_likelihood = policy_pi_log_likelihood - num_corr
        return squashed_policy_pi, squashed_policy_pi_log_likelihood
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ My bloc \\\(end)\\\


# (Priority) todo:assessment --> compare with mine: remove when done
def apply_action_bound_SpinningUp(policy_pi: tf.Tensor, policy_pi_log_likelihood: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    """
    Apply a invertible squashing function (tanh) to bound the actions sampled from the policy distribution in a
    finite interval (See the Soft Actor-Critic paper, appendice C)

        Sum_i log(1-tanh^2(z_i))

        with z_i = squashed_policy_pi

    In the limit as z_i --> inf, the 1-tanh^2(z_i) goes to 0
    ==>
    this can cause numerical instability that we mitigate by adding a small constant NUM_STABILITY_CORRECTION

    :param policy_pi:
    :param policy_pi_log_likelihood:
    :return: squashed_policy_pi, squashed_policy_pi_log_likelihood
    """
    with tf_cv1.variable_scope(vocab.squashing_fct):
        # (nice to have) todo:implement --> a numericaly stable version : see p8 HW5c Sergey Levine DRL course
        squashed_policy_pi = tf_cv1.tanh(policy_pi)
        
        # /// Original bloc ////////////////////////////////////////////////////////////////////////////////////////////
        # (Priority) todo:investigate?? --> Source from SpinningUp: Squashing function
        # pi = tf.tanh(pi)
        
        def clip_but_pass_gradient(x, l=-1., u=1.):
            clip_up = tf.cast(x > u, tf.float32)
            clip_low = tf.cast(x < l, tf.float32)
            return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)
        
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        policy_pi_log_likelihood -= tf.reduce_sum(tf.log(clip_but_pass_gradient(
            1 - squashed_policy_pi ** 2, l=0, u=1) + 1e-6), axis=1)

        return squashed_policy_pi, policy_pi_log_likelihood
        # //////////////////////////////////////////////////////////////////////////////////// Original bloc ///(end)///


def build_gaussian_policy_graph(obs_t_ph: tf.Tensor, exp_spec: ExperimentSpec,
                                playground: GymPlayground) -> Tuple[tf_cv1.Tensor, ...]:
    """
    The ACTOR graph(aka the policy network)
    A gaussian policy with state-dependent learnable mean and variance

        1. Actor network phi
            input: the observations collected
            output:
                - policy_mu layer --> the logits of each action in the action space as the mean of the Multivariate
                Normal distribution
                - policy_log_std layer --> the log standard deviation for the Multivariate Normal distribution

        2. Policy
            input: the actor network policy_mu and policy_log_std layers
            output: the sampled actions in the action space + corresponding log likelihood

    :return: policy_pi, policy_pi_log_likelihood, policy_mu
    """
    
    # ::Discrete case
    if isinstance(playground.env.action_space, gym.spaces.Discrete):
        raise ValueError("Discrete environment are not compatible with this Soft Actor-Critic implementation")

    # ::Continuous case
    elif isinstance(playground.env.action_space, gym.spaces.Box):
        """ ---- Assess the input shape compatibility ---- """
        are_compatible = obs_t_ph.shape.as_list()[-1] == playground.OBSERVATION_SPACE.shape[0]
        assert are_compatible, ("the obs_t_ph is incompatible with environment, "
                                "{} != {}").format(obs_t_ph.shape.as_list()[-1],
                                                   playground.OBSERVATION_SPACE.shape[0])

        """ ---- Build parameter PHI as a multilayer perceptron ---- """
        if USE_KERAS_LAYER:
            print(':: Use Keras style Dense layer')
            phi_mlp = build_KERAS_MLP_computation_graph(obs_t_ph, playground.ACTION_CHOICES,
                                                        exp_spec['phi_nn_h_layer_topo'],
                                                        hidden_layers_activation=exp_spec[
                                                            'phi_hidden_layers_activation'],
                                                        output_layers_activation=exp_spec[
                                                            'phi_hidden_layers_activation'],
                                                        name=vocab.phi)

            policy_mu = keras.layers.Dense(playground.ACTION_CHOICES,
                                           activation=exp_spec['phi_output_layers_activation'],  # tf_cv1.tanh,
                                           name=vocab.policy_mu)(phi_mlp)

            policy_log_std = keras.layers.Dense(playground.ACTION_CHOICES,
                                                activation=tf_cv1.tanh,
                                                name=vocab.policy_log_std)(phi_mlp)

        else:
            print(':: Use legacy tensorFlow Dense layer')
            phi_mlp = buil_MLP_with_initilizer(obs_t_ph, playground.ACTION_CHOICES,
                                               # phi_mlp = build_MLP_computation_graph(obs_t_ph,
                                               # playground.ACTION_CHOICES,
                                               exp_spec['phi_nn_h_layer_topo'],
                                               hidden_layers_activation=exp_spec[
                                                   'phi_hidden_layers_activation'],
                                               output_layers_activation=exp_spec[
                                                   'phi_hidden_layers_activation'],
                                               kernel_init=POLICY_NN_KERNEL_INIT,
                                               name=vocab.phi)
    
            policy_mu = tf_cv1.layers.dense(phi_mlp,
                                            playground.ACTION_CHOICES,
                                            activation=exp_spec['phi_output_layers_activation'],  # tf_cv1.tanh,
                                            # (NICE TO HAVE) todo:validate --> kernel_initializer specific to gaussian:
                                            kernel_initializer=POLICY_NN_KERNEL_INIT,
                                            name=vocab.policy_mu)
    
            policy_log_std = tf_cv1.layers.dense(phi_mlp,
                                                 playground.ACTION_CHOICES,
                                                 activation=tf_cv1.tanh,
                                                 # (NICE TO HAVE) todo:validate --> kernel_initializer specific to
                                                 #  gaussian:
                                                 kernel_initializer=POLICY_NN_KERNEL_INIT,
                                                 name=vocab.policy_log_std)

        # /// My bloc //////////////////////////////////////////////////////////////////////////////////////////////////
        # # Note: clip log standard deviation as in the sac_original_paper/sac/distributions/normal.py
        policy_log_std = tf_cv1.clip_by_value(policy_log_std, POLICY_LOG_STD_CAP_MIN, POLICY_LOG_STD_CAP_MAX)

        # ... pi distribution investigation ............................................................................
        # (NICE TO HAVE) todo:assessment --> check if changes in implementation detail make a difference:
        """ ---- Build the policy for continuous space ---- """
        policy_distribution = tfp.distributions.MultivariateNormalDiag(loc=policy_mu,
                                                                       scale_diag=tf_cv1.exp(policy_log_std),
                                                                       allow_nan_stats=False)
        # .................................................................... pi distribution investigation ...(end)...
        policy_pi = policy_distribution.sample(name=vocab.policy_pi)
        policy_pi_log_likelihood = policy_distribution.log_prob(policy_pi,
                                                                name=vocab.policy_pi_log_likelihood)
        # ////////////////////////////////////////////////////////////////////////////////////////// My bloc ///(end)///

    # ::Other gym environment
    else:
        print("\n>>> The agent implementation does not support that environment space "
              "{} yet.\n\n".format(playground.env.action_space))
        raise NotImplementedError

    return policy_pi, policy_pi_log_likelihood, tf_cv1.tanh(policy_mu)


# (Priority) todo:assessment --> compare with mine: remove when done
def build_gaussian_policy_graph_SpinningUp(obs_t_ph: tf.Tensor, exp_spec: ExperimentSpec,
                                           playground: GymPlayground) -> Tuple[tf_cv1.Tensor, ...]:
    """
    The ACTOR graph(aka the policy network)
    A gaussian policy with state-dependent learnable mean and variance

        1. Actor network phi
            input: the observations collected
            output:
                - policy_mu layer --> the logits of each action in the action space as the mean of the Multivariate
                Normal distribution
                - policy_log_std layer --> the log standard deviation for the Multivariate Normal distribution

        2. Policy
            input: the actor network policy_mu and policy_log_std layers
            output: the sampled actions in the action space + corresponding log likelihood

    :return: policy_pi, policy_pi_log_likelihood, policy_mu
    """
    
    # ::Discrete case
    if isinstance(playground.env.action_space, gym.spaces.Discrete):
        raise ValueError("Discrete environment are not compatible with this Soft Actor-Critic implementation")
    
    # ::Continuous case
    elif isinstance(playground.env.action_space, gym.spaces.Box):
        """ ---- Assess the input shape compatibility ---- """
        are_compatible = obs_t_ph.shape.as_list()[-1] == playground.OBSERVATION_SPACE.shape[0]
        assert are_compatible, ("the obs_t_ph is incompatible with environment, "
                                "{} != {}").format(obs_t_ph.shape.as_list()[-1],
                                                   playground.OBSERVATION_SPACE.shape[0])
        
        """ ---- Build parameter PHI as a multilayer perceptron ---- """
        if USE_KERAS_LAYER:
            print(':: Use Keras style Dense layer')
            phi_mlp = build_KERAS_MLP_computation_graph(obs_t_ph, playground.ACTION_CHOICES,
                                                        exp_spec['phi_nn_h_layer_topo'],
                                                        hidden_layers_activation=exp_spec[
                                                            'phi_hidden_layers_activation'],
                                                        output_layers_activation=exp_spec[
                                                            'phi_hidden_layers_activation'],
                                                        name=vocab.phi)
            
            policy_mu = keras.layers.Dense(playground.ACTION_CHOICES,
                                           activation=exp_spec['phi_output_layers_activation'],  # tf_cv1.tanh,
                                           name=vocab.policy_mu)(phi_mlp)
            
            policy_log_std = keras.layers.Dense(playground.ACTION_CHOICES,
                                                activation=tf_cv1.tanh,
                                                name=vocab.policy_log_std)(phi_mlp)
        
        else:
            print(':: Use legacy tensorFlow Dense layer')
            phi_mlp = buil_MLP_with_initilizer(obs_t_ph, playground.ACTION_CHOICES,
                                               # phi_mlp = build_MLP_computation_graph(obs_t_ph,
                                               # playground.ACTION_CHOICES,
                                               exp_spec['phi_nn_h_layer_topo'],
                                               hidden_layers_activation=exp_spec[
                                                   'phi_hidden_layers_activation'],
                                               output_layers_activation=exp_spec[
                                                   'phi_hidden_layers_activation'],
                                               kernel_init=POLICY_NN_KERNEL_INIT,
                                               name=vocab.phi)
    
            policy_mu = tf_cv1.layers.dense(phi_mlp,
                                            playground.ACTION_CHOICES,
                                            activation=exp_spec['phi_output_layers_activation'],  # tf_cv1.tanh,
                                            # (NICE TO HAVE) todo:validate --> kernel_initializer specific to gaussian:
                                            kernel_initializer=POLICY_NN_KERNEL_INIT,
                                            name=vocab.policy_mu)
    
            policy_log_std = tf_cv1.layers.dense(phi_mlp,
                                                 playground.ACTION_CHOICES,
                                                 activation=tf_cv1.tanh,
                                                 # (NICE TO HAVE) todo:validate --> kernel_initializer specific to
                                                 #  gaussian:
                                                 kernel_initializer=POLICY_NN_KERNEL_INIT,
                                                 name=vocab.policy_log_std)
        
        # \\\ Original bloc \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # (Priority) todo:assessment --> SpinningUp SAC implementation of the gaussian policy:
        def gaussian_likelihood(x, mu, log_std):
            pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + NUM_STABILITY_CORRECTION)) ** 2 + 2 * log_std + np.log(
                2 * np.pi))
            return tf.reduce_sum(pre_sum, axis=1)
        
        policy_log_std = POLICY_LOG_STD_CAP_MIN + 0.5 * (POLICY_LOG_STD_CAP_MAX - POLICY_LOG_STD_CAP_MIN) * (
                policy_log_std + 1)
        
        std = tf.exp(policy_log_std)
        policy_pi = policy_mu + tf.random_normal(tf.shape(policy_mu)) * std
        policy_pi_log_likelihood = gaussian_likelihood(policy_pi, policy_mu, policy_log_std)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ Original bloc \\\(end)\\\
    
    # ::Other gym environment
    else:
        print("\n>>> The agent implementation does not support that environment space "
              "{} yet.\n\n".format(playground.env.action_space))
        raise NotImplementedError
    
    return policy_pi, policy_pi_log_likelihood, tf_cv1.tanh(policy_mu)


def build_critic_graph_v_psi(obs_t_ph: tf.Tensor, obs_t_prime_ph: tf.Tensor, exp_spec: ExperimentSpec) -> Tuple[
    tf.Tensor, ...]:
    """
    Critic network psi
            input: the observations 's_t' and 's_{t+1}'
            output: the logits of V and frozen V

    :return: v_psi, frozen_v_psi
    """

    if USE_KERAS_LAYER:
        mlp = build_KERAS_MLP_computation_graph
    else:
        mlp = buil_MLP_with_initilizer
    
        # with tf_cv1.variable_scope(vocab.critic_network):
        """ ---- Build parameter '_psi' as a multilayer perceptron ---- """
    v_psi = mlp(obs_t_ph, 1, exp_spec['psi_nn_h_layer_topo'],
                hidden_layers_activation=exp_spec['psi_hidden_layers_activation'],
                output_layers_activation=exp_spec['psi_output_layers_activation'],
                kernel_init=POLICY_NN_KERNEL_INIT,
                name=vocab.V_psi)

    """ ---- Build frozen parameter '_psi' as a multilayer perceptron ---- """
    frozen_v_psi = mlp(obs_t_prime_ph, 1, exp_spec['psi_nn_h_layer_topo'],
                       hidden_layers_activation=exp_spec['psi_hidden_layers_activation'],
                       output_layers_activation=exp_spec['psi_output_layers_activation'],
                       kernel_init=POLICY_NN_KERNEL_INIT,
                       name=vocab.frozen_V_psi)

    v_psi = tf_cv1.squeeze(v_psi, axis=1)
    frozen_v_psi = tf_cv1.squeeze(frozen_v_psi, axis=1)
    return v_psi, frozen_v_psi


def build_critic_graph_q_theta(obs_t_ph: tf.Tensor, act_t_ph: tf.Tensor, policy_py: tf.Tensor,
                               exp_spec: ExperimentSpec, name: str) -> Tuple[
    tf.Tensor, ...]:
    """
    Critic network theta 1 & 2
            input: the observations collected 's_t' & the executed action 'a_t' at timestep t
            output: the logits of Q_1 according to sampled action and to the gaussian 'policy_py'

    :return: Q_action, Q_policy
    """
    if USE_KERAS_LAYER:
        mlp = build_KERAS_MLP_computation_graph
    else:
        mlp = buil_MLP_with_initilizer
    
    with tf_cv1.variable_scope(name):
        inputs_obs_act = tf_cv1.concat([obs_t_ph, act_t_ph], axis=-1)
        Q_action = mlp(inputs_obs_act, 1, exp_spec.theta_nn_h_layer_topo,
                       hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                       output_layers_activation=exp_spec.theta_output_layers_activation,
                       kernel_init=POLICY_NN_KERNEL_INIT,
                       name="mlp")
        
        Q_action = tf_cv1.squeeze(Q_action, axis=1)
    
    with tf_cv1.variable_scope(name, reuse=True):
        inputs_obs_pi = tf_cv1.concat([obs_t_ph, policy_py], axis=-1)
        Q_policy = mlp(inputs_obs_pi, 1, exp_spec.theta_nn_h_layer_topo,
                       hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                       output_layers_activation=exp_spec.theta_output_layers_activation,
                       kernel_init=POLICY_NN_KERNEL_INIT,
                       name="mlp")
        
        Q_policy = tf_cv1.squeeze(Q_policy, axis=1)
    
    return Q_action, Q_policy


def critic_v_psi_train(V_psi: tf.Tensor, Q_pi_1: tf.Tensor, Q_pi_2: tf.Tensor,
                       policy_pi_log_likelihood: tf.Tensor, exp_spec: ExperimentSpec, critic_lr_schedule,
                       critic_global_grad_step) -> Tuple[tf.Tensor, tf.Operation]:
    """
    Critic v_psi loss
                input:
                output: the Mean Squared Error (MSE)

    :param V_psi:
    :param Q_pi_1: Q_theta_1 according to the reparametrized policy
    :param Q_pi_2: Q_theta_2 according to the reparametrized policy
    :param policy_pi_log_likelihood:
    :param exp_spec:
    :param critic_lr_schedule:
    :param critic_global_grad_step:
    :return: v_loss, v_psi_optimizer
    """
    alpha = exp_spec['alpha']

    """ ---- Build the Mean Square Error loss function ---- """
    with tf_cv1.variable_scope(vocab.V_psi_loss):
        min_q_theta = tf_cv1.minimum(Q_pi_1, Q_pi_2)

        v_psi_target = tf_cv1.stop_gradient(min_q_theta - alpha * policy_pi_log_likelihood)

        v_loss = 0.5 * tf.reduce_mean((v_psi_target - V_psi) ** 2)

    """ ---- Fetch all tensor from V_psi and frozen_V_psi for update ---- """
    # (nice to have) todo:investigate?? --> find a other way to pass the network weight between the V and frozen_V:
    var_list = get_variable_explicitely_by_graph_key_from(vocab.critic_network + '/' + vocab.V_psi)
    assert len(var_list) is not 0

    """ ---- Critic optimizer ---- """
    v_psi_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=critic_lr_schedule
                                                 ).minimize(loss=v_loss,
                                                            var_list=var_list,
                                                            global_step=critic_global_grad_step)

    return v_loss, v_psi_optimizer


def critic_q_theta_train(frozen_v_psi: tf.Tensor, q_theta_1: tf.Tensor, q_theta_2: tf.Tensor, rew_ph: tf.Tensor,
                         trj_done_ph: tf.Tensor, exp_spec: ExperimentSpec,
                         critic_lr_schedule, critic_global_grad_step
                         ) -> Tuple[tf_cv1.Tensor, tf_cv1.Tensor, tf_cv1.Operation, tf_cv1.Operation]:
    """
    Critic q_theta {1,2} temporal difference loss
                input:
                output: the Mean Squared Error (MSE)

    :param critic_lr_schedule:
    :param critic_global:
    :return: q_theta_1_loss, q_theta_2_loss, q_theta_1_optimizer, q_theta_2_optimizer
    """
    
    q_target = tf_cv1.stop_gradient(
        exp_spec['reward_scaling'] * rew_ph + exp_spec.discout_factor * (1 - trj_done_ph) * frozen_v_psi)

    """ ---- Build the Mean Square Error loss function ---- """
    # with tf_cv1.variable_scope(vocab.critic_loss):
    with tf_cv1.variable_scope(vocab.Q_theta_1_loss):
        q_theta_1_loss = 0.5 * tf.reduce_mean((q_target - q_theta_1) ** 2)

    with tf_cv1.variable_scope(vocab.Q_theta_2_loss):
        q_theta_2_loss = 0.5 * tf.reduce_mean((q_target - q_theta_2) ** 2)

    """ ---- Critic optimizer & learning rate scheduler ---- """
    var_list_1 = get_variable_explicitely_by_graph_key_from(vocab.critic_network + '/' + vocab.Q_theta_1)
    var_list_2 = get_variable_explicitely_by_graph_key_from(vocab.critic_network + '/' + vocab.Q_theta_2)

    assert len(var_list_1) is not 0
    assert len(var_list_2) is not 0

    # note: global_step=critic_global_grad_step is already control by 'critic_v_psi_train' AdamOptimizer
    q_theta_1_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=critic_lr_schedule
                                                     ).minimize(loss=q_theta_1_loss,
                                                                var_list=var_list_1,
                                                                # global_step=critic_global_grad_step
                                                                )

    q_theta_2_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=critic_lr_schedule
                                                     ).minimize(loss=q_theta_2_loss,
                                                                var_list=var_list_2,
                                                                # global_step=critic_global_grad_step
                                                                )

    return q_theta_1_loss, q_theta_2_loss, q_theta_1_optimizer, q_theta_2_optimizer


def actor_train(policy_pi_log_likelihood: tf.Tensor, Q_pi_1: tf.Tensor, Q_pi_2: tf.Tensor,
                exp_spec: ExperimentSpec) -> Tuple[tf.Tensor, tf.Operation]:
    """
    Actor loss
        input:
        output:

    note: Hyperparameter alpha (aka Temperature, Entropy regularization coefficient )
      |    Control the trade-off between exploration-exploitation
      |    We recover the standard maximum expected return objective (the Q-fct) as alpha --> 0

    :return: actor_kl_loss, actor_policy_optimizer_op
    """

    alpha = exp_spec['alpha']

    """ ---- Build the Kullback-Leibler divergence loss function ---- """
    # ... Investigate ..................................................................................................
    # (nice to have) todo:investigate?? --> check wether to use min_Q1_Q2 or Q_1.
    #                                       SAC paper talk about using min_Q1_Q2
    #                                       but no implementation use it (SpinningUp, SAC original impl, ... ):

    # min_q_theta = tf_cv1.minimum(Q_pi_1, Q_pi_2)
    # actor_kl_loss = tf_cv1.reduce_mean(alpha * policy_pi_log_likelihood - min_q_theta,
    #                                    name=vocab.actor_kl_loss)

    actor_kl_loss = tf_cv1.reduce_mean(alpha * policy_pi_log_likelihood - Q_pi_1,
                                       name=vocab.actor_kl_loss)
    # .......................................................................................... Investigate ...(end)...

    """ ---- Actor optimizer & learning rate scheduler ---- """
    actor_lr_schedule, actor_global_grad_step = learning_rate_scheduler(
        max_gradient_step_expected=exp_spec['max_gradient_step_expected'],
        learning_rate=exp_spec.learning_rate,
        lr_decay_rate=exp_spec['actor_lr_decay_rate'],
        name_sufix='actor')

    var_list = get_variable_explicitely_by_graph_key_from(vocab.actor_network)
    assert len(var_list) is not 0
    actor_policy_optimizer_op = tf_cv1.train.AdamOptimizer(learning_rate=actor_lr_schedule
                                                           ).minimize(loss=actor_kl_loss,
                                                                      var_list=var_list,
                                                                      global_step=actor_global_grad_step,
                                                                      # name=vocab.policy_optimizer
                                                                      )
    return actor_kl_loss, actor_policy_optimizer_op


def update_frozen_v_psi_op(tau: float) -> tf.Operation:
    """
    Utility function: fetch all V_psi & frozen_V_psi graph key and update frozen_V_psi network
    :param tau: target_smoothing_coefficient
    :return: the update op
    """
    with tf_cv1.variable_scope(vocab.frozen_V_psi_update_ops):
        frozen_v_psi_update_ops = _update_frozen_v_psi_op(tau)
    return frozen_v_psi_update_ops


def init_frozen_v_psi() -> tf.Operation:
    """
    Pass a exact copy of the weight of V_psi to frozen_V_psi
    :return: the cloning op
    """
    with tf_cv1.variable_scope(vocab.frozen_V_psi + '_init_ops'):
        init_frozen_V_psi_op = update_frozen_v_psi_op(1.0)
    return init_frozen_V_psi_op


def _update_frozen_v_psi_op(tau):
    """Construct the target update op
    :param tau: target_smoothing_coefficient
    :return: the update op
    """
    v_psi_graph_key = get_variable_explicitely_by_graph_key_from(vocab.critic_network + '/' + vocab.V_psi)
    frozen_v_psi_graph_key = get_variable_explicitely_by_graph_key_from(vocab.critic_network + '/' + vocab.frozen_V_psi)
    assert len(v_psi_graph_key) is not 0
    assert len(frozen_v_psi_graph_key) is not 0
    frozen_v_psi_update_ops = update_nn_weights(v_psi_graph_key, frozen_v_psi_graph_key, tau)
    return frozen_v_psi_update_ops


def critic_learning_rate_scheduler(exp_spec: ExperimentSpec):
    return learning_rate_scheduler(
        max_gradient_step_expected=exp_spec['max_gradient_step_expected'] * exp_spec.max_epoch,
        learning_rate=exp_spec['critic_learning_rate'],
        lr_decay_rate=exp_spec['critic_lr_decay_rate'],
        name_sufix='critic')
