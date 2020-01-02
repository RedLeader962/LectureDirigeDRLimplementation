# coding=utf-8
import gym
import tensorflow as tf
import tensorflow_probability as tfp

from blocAndTools.buildingbloc import (ExperimentSpec, GymPlayground, build_MLP_computation_graph,
                                       policy_theta_discrete_space, discrete_pseudo_loss, policy_optimizer,
                                       learning_rate_scheduler)
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1  # shortcut
vocab = rl_name()

POLICY_LOG_STD_CAP_MAX = 2
POLICY_LOG_STD_CAP_MIN = -20
NUM_STABILITY_CORRECTION = 1e-6


def build_gaussian_policy_graph(observation_placeholder: tf.Tensor, experiment_spec: ExperimentSpec,
                                playground: GymPlayground) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    The ACTOR graph(aka the policy network)
    A gaussian policy with state-dependent learnable mean and variance

        1. Actor network phi
            input: the observations collected
            output:
                - policy_mu layer --> the logits of each action in the action space as the mean of the Multivariate Normal distribution
                - policy_log_std layer --> the log standard deviation for the Multivariate Normal distribution

        2. Policy
            input: the actor network policy_mu and policy_log_std layers
            output: the sampled actions in the action space + corresponding log likelihood

    :return: sampled_action, sampled_action_logLikelihood, policy_mu
    """
    with tf.name_scope(vocab.actor_network) as scope:

        # ::Discrete case
        if isinstance(playground.env.action_space, gym.spaces.Discrete):
            raise ValueError("Discrete environment are not compatible with this Soft Actor-Critic implementation")

        # ::Continuous case
        elif isinstance(playground.env.action_space, gym.spaces.Box):
            """ ---- Assess the input shape compatibility ---- """
            are_compatible = observation_placeholder.shape.as_list()[-1] == playground.OBSERVATION_SPACE.shape[0]
            assert are_compatible, ("the observation_placeholder is incompatible with environment, "
                                    "{} != {}").format(observation_placeholder.shape.as_list()[-1],
                                                       playground.OBSERVATION_SPACE.shape[0])

            """ ---- Build parameter PHI as a multilayer perceptron ---- """
            phi_mlp = build_MLP_computation_graph(observation_placeholder, playground.ACTION_CHOICES,
                                                    experiment_spec.phi_nn_h_layer_topo,
                                                    hidden_layers_activation=experiment_spec.phi_hidden_layers_activation,
                                                    output_layers_activation=experiment_spec.phi_output_layers_activation,
                                                    name=vocab.policy_network_phi)

            policy_mu = tf_cv1.layers.dense(phi_mlp,
                                            playground.ACTION_CHOICES,
                                            activation=experiment_spec.phi_output_layers_activation,
                                            name=vocab.policy_network_phi + '/' + vocab.policy_mu)

            policy_log_std = tf_cv1.layers.dense(phi_mlp,
                                                 playground.ACTION_CHOICES,
                                                 activation=tf_cv1.tanh,
                                                 name=vocab.policy_network_phi + '/' + vocab.policy_log_std)

            # Note: clip log standard deviation as in the sac_original_paper/sac/distributions/normal.py
            policy_log_std = tf_cv1.clip_by_value(policy_log_std, POLICY_LOG_STD_CAP_MIN, POLICY_LOG_STD_CAP_MAX)

            """ ---- Build the policy for continuous space ---- """
            policy_distribution = tfp.distributions.MultivariateNormalDiag(loc=policy_mu, scale_diag=tf_cv1.exp(policy_log_std))
            sampled_action = policy_distribution.sample(name=vocab.sampled_action)
            sampled_action_logLikelihood = policy_distribution.log_prob(sampled_action, name=vocab.sampled_action_logLikelihood)

        # ::Other gym environment
        else:
            print("\n>>> The agent implementation does not support that environment space "
                  "{} yet.\n\n".format(playground.env.action_space))
            raise NotImplementedError

    return sampled_action, sampled_action_logLikelihood, policy_mu


def apply_action_bound(sampled_action: tf_cv1.Tensor, sampled_action_logLikelihood: tf_cv1.Tensor) -> (tf_cv1.Tensor, tf_cv1.Tensor):
    """
    Apply a invertible squashing function (tanh) to bound the actions sampled from the policy distribution in a finite interval
    See the Soft Actor-Critic paper: appendice C

        Sum_i log(1-tanh^2(z_i))

        with z_i = squashed_sampled_action

    In the limit as z_i --> inf, the 1-tanh^2(z_i) goes to 0
    ==>
    this can cause numerical instability that we mitigate by adding a small constant NUM_STABILITY_CORRECTION

    :param sampled_action:
    :param sampled_action_logLikelihood:
    :return: sampled_action, sampled_action_logLikelihood
    """
    # (nice to have) todo:implement --> a numericaly stable version : see p8 HW5c Sergey Levine DRL course
    squashed_sampled_action = tf_cv1.tanh(sampled_action)
    corr = tf_cv1.reduce_sum(tf_cv1.log(1-tf_cv1.tanh(squashed_sampled_action)**2 + NUM_STABILITY_CORRECTION), axis=1)
    squashed_sampled_action_logLikelihood = sampled_action_logLikelihood - corr
    return squashed_sampled_action, squashed_sampled_action_logLikelihood


def build_critic_graph_V_psi(obs_t_ph: tf.Tensor, exp_spec: ExperimentSpec) -> tf.Tensor:
    """
    Critic network psi
            input: the observations 's_t'
            output: the logits of V

    :return: critic_V_psi
    """

    with tf.name_scope(vocab.critic_network) as scope:
        """ ---- Build parameter '_psi' as a multilayer perceptron ---- """
        critic_V_psi = build_MLP_computation_graph(obs_t_ph, 1, exp_spec.theta_nn_h_layer_topo,
                                                   hidden_layers_activation=exp_spec.psi_hidden_layers_activation,
                                                   output_layers_activation=exp_spec.psi_output_layers_activation,
                                                   name=vocab.critic_network_V_psi)
    return critic_V_psi


def build_critic_graph_Q_theta(obs_t_ph: tf.Tensor, act_t_ph: tf_cv1.Tensor, exp_spec: ExperimentSpec) -> (tf.Tensor, tf.Tensor):
    """
    Critic network theta 1 & 2
            input: the observations collected 's_t' & the executed action 'a_t' at timestep t
            output: the logits of Q_1 and Q_2

    :return: critic_Q_theta_1, critic_Q_theta_2
    """

    with tf.name_scope(vocab.critic_network) as scope:
        inputs = tf_cv1.concat([obs_t_ph, act_t_ph], axis=-1)

        """ ---- Build parameter '_theta_1' as a multilayer perceptron ---- """
        critic_Q_theta_1 = build_MLP_computation_graph(inputs, 1, exp_spec.theta_nn_h_layer_topo,
                                                       hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                                                       output_layers_activation=exp_spec.theta_output_layers_activation,
                                                       name=vocab.critic_network_Q_theta + '_1')

        """ ---- Build parameter '_theta_2' as a multilayer perceptron ---- """
        critic_Q_theta_2 = build_MLP_computation_graph(inputs, 1, exp_spec.theta_nn_h_layer_topo,
                                                       hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                                                       output_layers_activation=exp_spec.theta_output_layers_activation,
                                                       name=vocab.critic_network_Q_theta + '_2')

    return critic_Q_theta_1, critic_Q_theta_2


def actor_train(sampled_action_logLikelihood: tf_cv1.Tensor, critic_graph_Q_theta: tf_cv1.Tensor,
                experiment_spec: ExperimentSpec, playground: GymPlayground) -> (tf.Tensor, tf.Operation):
    """
    Actor loss
        input:
        output:

    note: ExperimentSpec.temperature (hparam alpha)
      |    Control the trade-off between exploration-exploitation
      |    We recover a standard maximum expected return objectiv as alpha --> 0

    :return: actor_KL_loss, actor_policy_optimizer_op
    """

    alpha = ExperimentSpec.temperature

    with tf_cv1.name_scope(vocab.policy_training):
        """ ---- Build the pseudo loss function ---- """
        actor_KL_loss = tf_cv1.reduce_mean(alpha * sampled_action_logLikelihood - critic_graph_Q_theta)

        """ ---- Actor optimizer & learning rate scheduler ---- """
        actor_lr_schedule, actor_global_grad_step = learning_rate_scheduler(
            max_gradient_step_expected=experiment_spec.max_epoch,
            learning_rate=experiment_spec.learning_rate,
            lr_decay_rate=experiment_spec['actor_lr_decay_rate'],
            name_sufix='actor')

        # actor_policy_optimizer_op = policy_optimizer(actor_KL_loss,
        #                                              learning_rate=actor_lr_schedule,
        #                                              global_gradient_step=actor_global_grad_step)

        # actor_policy_optimizer_op = policy_optimizer(actor_KL_loss, experiment_spec.learning_rate)

        actor_policy_optimizer_op = tf_cv1.train.AdamOptimizer(learning_rate=actor_lr_schedule
                                                               ).minimize(loss=actor_KL_loss,
                                                                          global_step=actor_global_grad_step,
                                                                          name=vocab.policy_optimizer)
    return actor_KL_loss, actor_policy_optimizer_op


def critic_train(advantage, experiment_spec: ExperimentSpec) -> (tf.Tensor, tf.Operation):
    """
    Critic loss
                input: the target y (either Monte Carlo target or Bootstraped estimate target)
                output: the Mean Squared Error (MSE)

    :return: critic_loss, critic_optimizer
    """
    with tf_cv1.name_scope(vocab.critic_training):
        pass  # (Priority) todo:implement --> task:

        """ ---- Build the Mean Square Error loss function ---- """
    #     with tf.name_scope(vocab.critic_loss):
    #         critic_loss = tf.reduce_mean(advantage ** 2)
    #
    #     """ ---- Critic optimizer & learning rate scheduler ---- """
    #     critic_lr_schedule, critic_global_grad_step = learning_rate_scheduler(
    #         max_gradient_step_expected=experiment_spec['critique_loop_len'] * experiment_spec.max_epoch,
    #         learning_rate=experiment_spec['critic_learning_rate'],
    #         lr_decay_rate=experiment_spec['critic_lr_decay_rate'],
    #         name_sufix='critic')
    #
    #     critic_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=critic_lr_schedule
    #                                                   ).minimize(critic_loss,
    #                                                              global_step=critic_global_grad_step,
    #                                                              name=vocab.critic_optimizer)
    #     # critic_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=experiment_spec['critic_learning_rate']).minimize(critic_loss, name=vocab.critic_optimizer)
    # return critic_loss, critic_optimizer
