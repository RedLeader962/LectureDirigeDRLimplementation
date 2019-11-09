# coding=utf-8
import gym
import tensorflow as tf

from blocAndTools.buildingbloc import (ExperimentSpec, GymPlayground, build_MLP_computation_graph,
                                       policy_theta_discrete_space, discrete_pseudo_loss, learning_rate_scheduler,
                                       policy_optimizer, )
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1  # shortcut
vocab = rl_name()


def build_actor_critic_shared_graph(obs_ph: tf.Tensor, exp_spec: ExperimentSpec,
                                    playground: GymPlayground) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
    """
    The ACTOR-CRITIC shared network variant architecture

        1. Actor network theta
            input: the observations collected
            output: the logits of each action in the action space

        2. Policy
            input: the actor network
            output: a selected action & the probabilities of each action in the action space

        3. Critic network phi
            input: the observations collected
            output: the logits of each action in the action space

    :return: sampled_action, log_pi_all, theta_shared_MLP, critic
    """
    """ ---- Assess the input shape compatibility ---- """
    are_compatible = obs_ph.shape.as_list()[-1] == playground.OBSERVATION_SPACE.shape[0]
    assert are_compatible, ("the observation_placeholder is incompatible with environment, "
                            "{} != {}").format(obs_ph.shape.as_list()[-1],
                                               playground.OBSERVATION_SPACE.shape[0])

    # ::Discrete case
    if isinstance(playground.env.action_space, gym.spaces.Discrete):

        """ ---- Build parameter THETA as a multilayer perceptron ---- """
        theta_shared_MLP = build_MLP_computation_graph(obs_ph, playground.ACTION_CHOICES,
                                                       exp_spec.theta_nn_h_layer_topo,
                                                       hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                                                       output_layers_activation=exp_spec.theta_output_layers_activation,
                                                       reuse=None,          # <-- (!)
                                                       name=vocab.shared_network)
        """ ---- Build the policy for discrete space ---- """
        sampled_action, log_pi_all = policy_theta_discrete_space(theta_shared_MLP, playground)

    # ::Continuous case
    elif isinstance(playground.env.action_space, gym.spaces.Box):
        raise NotImplementedError  # (Ice-Boxed) todo:implement -->  for policy for continuous space:

    # ::Other gym environment
    else:
        print("\n>>> The agent implementation does not support that environment space "
              "{} yet.\n\n".format(playground.env.action_space))
        raise NotImplementedError

    """ ---- Build the Critic ---- """
    phi_shared_MLP = build_MLP_computation_graph(obs_ph, playground.ACTION_CHOICES,
                                                 exp_spec.theta_nn_h_layer_topo,
                                                 hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                                                 output_layers_activation=exp_spec.theta_output_layers_activation,
                                                 reuse=True,  # <-- (!)
                                                 name=vocab.shared_network)

    critic = build_MLP_computation_graph(phi_shared_MLP, 1, (),
                                         hidden_layers_activation=exp_spec.theta_hidden_layers_activation,
                                         output_layers_activation=exp_spec.theta_output_layers_activation,
                                         name=vocab.V_estimate)

    return sampled_action, log_pi_all, theta_shared_MLP, critic

def actor_shared_train(action_placeholder: tf_cv1.Tensor, log_pi, advantage: tf_cv1.Tensor,
                       experiment_spec: ExperimentSpec,
                       playground: GymPlayground) -> (tf.Tensor, tf.Operation):
    """
    Actor loss
        input: the probabilities of each action in the action space, the collected actions, the computed advantages
        output: Grad_theta log pi_theta * A^pi

    :return: actor_loss, actor_policy_optimizer_op
    """
    with tf_cv1.name_scope(vocab.policy_training):
        """ ---- Build the pseudo loss function ---- """
        actor_loss = discrete_pseudo_loss(log_pi, action_placeholder, advantage,
                                          playground, name=vocab.actor_loss)

        """ ---- Actor optimizer & learning rate scheduler ---- """
        # (Ice-Boxed) todo:implement --> finish lr sheduler for online shared algo:
        # (Ice-Boxed) todo:implement --> add 'global_timestep_max' to hparam:
        # actor_lr_schedule, actor_global_grad_step = learning_rate_scheduler(
        #     max_gradient_step_expected=experiment_spec['global_timestep_max'] / experiment_spec['batch_size_in_ts'],
        #     learning_rate=experiment_spec.learning_rate,
        #     lr_decay_rate=experiment_spec['actor_lr_decay_rate'],
        #     name_sufix='actor')
        #
        # actor_policy_optimizer_op = tf_cv1.train.AdamOptimizer(learning_rate=actor_lr_schedule
        #                                                        ).minimize(loss=actor_loss,
        #                                                                   global_step=actor_global_grad_step,
        #                                                                   name=vocab.policy_optimizer)

        actor_policy_optimizer_op = policy_optimizer(actor_loss, experiment_spec.learning_rate)
    return actor_loss, actor_policy_optimizer_op


def critic_shared_train(advantage, experiment_spec: ExperimentSpec) -> (tf.Tensor, tf.Operation):
    """
    Critic loss
                input: the target y (either Monte Carlo target or Bootstraped estimate target)
                output: the Mean Squared Error (MSE)

    :return: critic_loss, critic_optimizer
    """
    with tf_cv1.name_scope(vocab.critic_training):
        """ ---- Build the Mean Square Error loss function ---- """
        with tf.name_scope(vocab.critic_loss):
            critic_loss = tf.reduce_mean(advantage ** 2)

        """ ---- Critic optimizer & learning rate scheduler ---- """

        # (Ice-Boxed) todo:implement --> finish lr sheduler for online shared algo:
        # (Ice-Boxed) todo:implement --> add 'global_timestep_max' to hparam:
        # critic_lr_schedule, critic_global_grad_step = learning_rate_scheduler(
        #     max_gradient_step_expected=experiment_spec['critique_loop_len'] * experiment_spec.max_epoch,
        #     learning_rate=experiment_spec['critic_learning_rate'],
        #     lr_decay_rate=experiment_spec['critic_lr_decay_rate'],
        #     name_sufix='critic')
        #
        # critic_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=critic_lr_schedule
        #                                               ).minimize(critic_loss,
        #                                                          global_step=critic_global_grad_step,
        #                                                          name=vocab.critic_optimizer)

        critic_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=experiment_spec['critic_learning_rate']).minimize(critic_loss, name=vocab.critic_optimizer)
    return critic_loss, critic_optimizer

