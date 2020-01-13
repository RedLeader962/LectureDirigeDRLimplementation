# coding=utf-8

from typing import Tuple

import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras

from blocAndTools.buildingbloc import (
    ExperimentSpec, GymPlayground, build_MLP_computation_graph, build_KERAS_MLP_computation_graph,
    learning_rate_scheduler,
    )
from blocAndTools.rl_vocabulary import rl_name
from blocAndTools.tensorflowbloc import update_nn_weights, get_variables_graph_key

tf_cv1 = tf.compat.v1  # shortcut
vocab = rl_name()

POLICY_LOG_STD_CAP_MAX = 2
POLICY_LOG_STD_CAP_MIN = -20
NUM_STABILITY_CORRECTION = 1e-6
USE_KERAS_LAYER = True

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
    with tf_cv1.variable_scope(vocab.squashing_fct, reuse=True):
        # (nice to have) todo:implement --> a numericaly stable version : see p8 HW5c Sergey Levine DRL course
        squashed_policy_pi = tf_cv1.tanh(policy_pi)
        num_corr = tf_cv1.reduce_sum(tf_cv1.log(1 - squashed_policy_pi ** 2 + NUM_STABILITY_CORRECTION),
                                     axis=1)
        squashed_policy_pi_log_likelihood = policy_pi_log_likelihood - num_corr
        return squashed_policy_pi, squashed_policy_pi_log_likelihood


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
    # with tf_cv1.variable_scope(vocab.actor_network):

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
                                           activation=tf_cv1.tanh,
                                           name=vocab.policy_mu)(phi_mlp)
        
            policy_log_std = keras.layers.Dense(playground.ACTION_CHOICES,
                                                activation=tf_cv1.tanh,
                                                name=vocab.policy_log_std)(phi_mlp)
    
        else:
            print(':: Use legacy tensorFlow Dense layer')
            phi_mlp = build_MLP_computation_graph(obs_t_ph, playground.ACTION_CHOICES,
                                                  exp_spec['phi_nn_h_layer_topo'],
                                                  hidden_layers_activation=exp_spec[
                                                      'phi_hidden_layers_activation'],
                                                  output_layers_activation=exp_spec[
                                                      'phi_hidden_layers_activation'],
                                                  name=vocab.phi)
        
            policy_mu = tf_cv1.layers.dense(phi_mlp,
                                            playground.ACTION_CHOICES,
                                            activation=tf_cv1.tanh,
                                            name=vocab.policy_mu)
        
            policy_log_std = tf_cv1.layers.dense(phi_mlp,
                                                 playground.ACTION_CHOICES,
                                                 activation=tf_cv1.tanh,
                                                 name=vocab.policy_log_std)
    
        # Note: clip log standard deviation as in the sac_original_paper/sac/distributions/normal.py
        policy_log_std = tf_cv1.clip_by_value(policy_log_std, POLICY_LOG_STD_CAP_MIN, POLICY_LOG_STD_CAP_MAX)
    
        """ ---- Build the policy for continuous space ---- """
        policy_distribution = tfp.distributions.MultivariateNormalDiag(loc=policy_mu,
                                                                       scale_diag=tf_cv1.exp(policy_log_std),
                                                                       allow_nan_stats=False)
        policy_pi = policy_distribution.sample(name=vocab.policy_pi)
        policy_pi_log_likelihood = policy_distribution.log_prob(policy_pi,
                                                                name=vocab.policy_pi_log_likelihood)

    # ::Other gym environment
    else:
        print("\n>>> The agent implementation does not support that environment space "
              "{} yet.\n\n".format(playground.env.action_space))
        raise NotImplementedError
    
    return policy_pi, policy_pi_log_likelihood, policy_mu


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
        mlp = build_MLP_computation_graph
    
        # with tf_cv1.variable_scope(vocab.critic_network):
        """ ---- Build parameter '_psi' as a multilayer perceptron ---- """
    v_psi = mlp(obs_t_ph, 1, exp_spec['psi_nn_h_layer_topo'],
                hidden_layers_activation=exp_spec['psi_hidden_layers_activation'],
                output_layers_activation=exp_spec['psi_output_layers_activation'],
                name=vocab.V_psi)

    """ ---- Build frozen parameter '_psi' as a multilayer perceptron ---- """
    frozen_v_psi = mlp(obs_t_prime_ph, 1, exp_spec['psi_nn_h_layer_topo'],
                       hidden_layers_activation=exp_spec['psi_hidden_layers_activation'],
                       output_layers_activation=exp_spec['psi_output_layers_activation'],
                       name=vocab.frozen_V_psi)
    return v_psi, frozen_v_psi


def init_frozen_v_psi() -> tf.Operation:
    """
    Pass a exact copy of the weight of V_psi to frozen_V_psi
    :return: the cloning op
    """
    init_frozen_V_psi_op = _update_frozen_v_psi(1.0)
    return init_frozen_V_psi_op


def build_critic_graph_q_theta(obs_t_ph: tf.Tensor, act_t_ph: tf.Tensor, exp_spec: ExperimentSpec) -> Tuple[
    tf.Tensor, ...]:
    """
    Critic network theta 1 & 2
            input: the observations collected 's_t' & the executed action 'a_t' at timestep t
            output: the logits of Q_1 and Q_2

    :return: q_theta_1, q_theta_2
    """
    if USE_KERAS_LAYER:
        mlp = build_KERAS_MLP_computation_graph
    else:
        mlp = build_MLP_computation_graph

    # with tf_cv1.variable_scope(vocab.critic_network):
    """ ---- Concat graph input: observation & action ---- """
    inputs = tf_cv1.concat([obs_t_ph, act_t_ph], axis=-1)

    """ ---- Build parameter '_theta_1' as a multilayer perceptron ---- """
    q_theta_1 = mlp(inputs, 1, exp_spec.theta_nn_h_layer_topo,
                    hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                    output_layers_activation=exp_spec.theta_output_layers_activation,
                    name=vocab.Q_theta_1)

    """ ---- Build parameter '_theta_2' as a multilayer perceptron ---- """
    q_theta_2 = mlp(inputs, 1, exp_spec.theta_nn_h_layer_topo,
                    hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                    output_layers_activation=exp_spec.theta_output_layers_activation,
                    name=vocab.Q_theta_2)

    return q_theta_1, q_theta_2


def actor_train(policy_pi_log_likelihood: tf.Tensor, q_theta_1: tf.Tensor, q_theta_2: tf.Tensor,
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

    # with tf_cv1.variable_scope(vocab.policy_training):

    # min_q_theta = tf_cv1.minimum(q_theta_1, q_theta_2)
    #
    # """ ---- Build the Kullback-Leibler divergence loss function ---- """
    # actor_kl_loss = tf_cv1.reduce_mean(alpha * policy_pi_log_likelihood - min_q_theta,
    #                                    name=vocab.actor_kl_loss)
    #

    """ ---- Build the Kullback-Leibler divergence loss function ---- """
    actor_kl_loss = tf_cv1.reduce_mean(alpha * policy_pi_log_likelihood - q_theta_1,
                                       name=vocab.actor_kl_loss)

    """ ---- Actor optimizer & learning rate scheduler ---- """
    actor_lr_schedule, actor_global_grad_step = learning_rate_scheduler(
        max_gradient_step_expected=exp_spec['max_gradient_step_expected'],
        learning_rate=exp_spec.learning_rate,
        lr_decay_rate=exp_spec['actor_lr_decay_rate'],
        name_sufix='actor')

    actor_policy_optimizer_op = tf_cv1.train.AdamOptimizer(learning_rate=actor_lr_schedule
                                                           ).minimize(loss=actor_kl_loss,
                                                                      global_step=actor_global_grad_step,
                                                                      name=vocab.policy_optimizer)
    return actor_kl_loss, actor_policy_optimizer_op


def critic_v_psi_train(v_psi: tf.Tensor, q_theta_1: tf.Tensor, q_theta_2: tf.Tensor,
                       policy_pi_log_likelihood: tf.Tensor, exp_spec: ExperimentSpec, critic_lr_schedule,
                       critic_global_grad_step) -> Tuple[tf.Tensor, tf.Operation, tf.Operation]:
    """
    Critic v_psi loss
                input:
                output: the Mean Squared Error (MSE)

    :return: v_loss, v_psi_optimizer, frozen_v_psi_update_ops
    """
    alpha = exp_spec['alpha']
    tau = exp_spec['target_smoothing_coefficient']
    
    # with tf_cv1.variable_scope(vocab.critic_training):
    # with tf_cv1.variable_scope(vocab.critic_loss):
    
    """ ---- Build the Mean Square Error loss function ---- """
    with tf_cv1.variable_scope(vocab.V_psi_loss, reuse=True):
        min_q_theta = tf_cv1.minimum(q_theta_1, q_theta_2)
    
        v_psi_target = tf_cv1.stop_gradient(min_q_theta - alpha * policy_pi_log_likelihood)
    
        v_loss = 0.5 * tf.reduce_mean((v_psi - v_psi_target) ** 2)
    
        """ ---- Critic optimizer & learning rate scheduler ---- """
        # critic_lr_schedule, critic_global_grad_step = critic_learning_rate_scheduler(exp_spec)

    v_psi_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=critic_lr_schedule
                                                 ).minimize(v_loss,
                                                            global_step=critic_global_grad_step)

    # (nice to have) todo:investigate?? --> find a other way to pass the network weight between the V and frozen_V:

    """ ---- Fetch all tensor from V_psi and frozen_V_psi for update ---- """
    frozen_v_psi_update_ops = _update_frozen_v_psi(tau)

    return v_loss, v_psi_optimizer, frozen_v_psi_update_ops


def _update_frozen_v_psi(tau: float):
    """
    Utility function: fetch all V_psi & frozen_V_psi graph key and update frozen_V_psi network
    :param tau: target_smoothing_coefficient
    :return: the update op
    """
    with tf_cv1.variable_scope(vocab.frozen_V_psi_update_ops, reuse=True):
        v_psi_graph_key = get_variables_graph_key(vocab.V_psi)
        frozen_v_psi_graph_key = get_variables_graph_key(vocab.frozen_V_psi)
        frozen_v_psi_update_ops = update_nn_weights(v_psi_graph_key, frozen_v_psi_graph_key, tau)
    return frozen_v_psi_update_ops


def critic_q_theta_train(frozen_v_psi: tf.Tensor, q_theta_1: tf.Tensor, q_theta_2: tf.Tensor, rew_ph: tf.Tensor,
                         trj_done_ph: tf.Tensor, exp_spec: ExperimentSpec,
                         critic_lr_schedule, critic_global_grad_step) -> Tuple[tf_cv1.Tensor, tf_cv1.Tensor,
                                                                               tf_cv1.Operation, tf_cv1.Operation]:
    """
    Critic q_theta {1,2} temporal difference loss
                input:
                output: the Mean Squared Error (MSE)

    :param critic_lr_schedule:
    :param critic_global:
    :return: q_theta_1_loss, q_theta_2_loss, q_theta_1_optimizer, q_theta_2_optimizer
    """
    # with tf_cv1.variable_scope(vocab.critic_training):
    
    # ... investigate
    # ...................................................................................................
    # (Priority) todo:investigate?? --> does the squeeze introduce a error?:
    # q_target = tf_cv1.stop_gradient(
    #     exp_spec['reward_scaling'] * rew_ph + (1 - trj_done_ph) * exp_spec.discout_factor * tf_cv1.squeeze(
    #         frozen_v_psi))

    q_target = tf_cv1.stop_gradient(
        exp_spec['reward_scaling'] * rew_ph + (1 - trj_done_ph) * exp_spec.discout_factor * frozen_v_psi)
    # ........................................................................................... investigate ...(
    # end)...

    """ ---- Build the Mean Square Error loss function ---- """
    # with tf_cv1.variable_scope(vocab.critic_loss):
    with tf_cv1.variable_scope(vocab.Q_theta_1_loss, reuse=True):
        q_theta_1_loss = 0.5 * tf.reduce_mean((q_target - q_theta_1) ** 2)

    with tf_cv1.variable_scope(vocab.Q_theta_2_loss, reuse=True):
        q_theta_2_loss = 0.5 * tf.reduce_mean((q_target - q_theta_2) ** 2)

    """ ---- Critic optimizer & learning rate scheduler ---- """
    q_theta_1_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=critic_lr_schedule
                                                     ).minimize(q_theta_1_loss,
                                                                global_step=critic_global_grad_step)

    q_theta_2_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=critic_lr_schedule
                                                     ).minimize(q_theta_2_loss,
                                                                global_step=critic_global_grad_step)
    
    return q_theta_1_loss, q_theta_2_loss, q_theta_1_optimizer, q_theta_2_optimizer


def critic_learning_rate_scheduler(exp_spec: ExperimentSpec):
    return learning_rate_scheduler(
        max_gradient_step_expected=exp_spec['max_gradient_step_expected'] * exp_spec.max_epoch,
        learning_rate=exp_spec['critic_learning_rate'],
        lr_decay_rate=exp_spec['critic_lr_decay_rate'],
        name_sufix='critic')
