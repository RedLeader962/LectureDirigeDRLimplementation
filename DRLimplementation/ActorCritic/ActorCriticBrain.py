# coding=utf-8
import gym
import tensorflow as tf

from blocAndTools.buildingbloc import (ExperimentSpec, GymPlayground, build_MLP_computation_graph,
                                       policy_theta_discrete_space, discrete_pseudo_loss, policy_optimizer)
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1  # shortcut
vocab = rl_name()


def build_actor_policy_graph(observation_placeholder: tf.Tensor, action_placeholder: tf.Tensor,
                             advantage_placeholder: tf.Tensor, experiment_spec: ExperimentSpec,
                             playground: GymPlayground) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Operation):
    """
    The ACTOR graph(aka the policy network)

        1. Actor network theta
            input: the observations collected
            output: the logits of each action in the action space

        2. Policy
            input: the actor network
            output: a selected action & the probabilities of each action in the action space

        3. Actor loss
            input: the probabilities of each action in the action space, the collected actions, the computed advantages
        output: Grad_theta log pi_theta * A^pi

    :return: sampled_action, theta_mlp, actor_loss, actor_policy_optimizer_op
    """
    with tf.name_scope(vocab.actor_network) as scope:

        """ ---- Build parameter THETA as a multilayer perceptron ---- """
        theta_mlp = build_MLP_computation_graph(observation_placeholder, playground.ACTION_CHOICES,
                                                experiment_spec.theta_nn_h_layer_topo,
                                                hidden_layers_activation=experiment_spec.theta_hidden_layers_activation,
                                                output_layers_activation=experiment_spec.theta_output_layers_activation,
                                                name=vocab.theta_NeuralNet)

        # ::Discrete case
        if isinstance(playground.env.action_space, gym.spaces.Discrete):

            """ ---- Assess the input shape compatibility ---- """
            are_compatible = observation_placeholder.shape.as_list()[-1] == playground.OBSERVATION_SPACE.shape[0]
            assert are_compatible, ("the observation_placeholder is incompatible with environment, "
                                    "{} != {}").format(observation_placeholder.shape.as_list()[-1],
                                                       playground.OBSERVATION_SPACE.shape[0])

            """ ---- Build the policy for discrete space ---- """
            sampled_action, log_p_all = policy_theta_discrete_space(theta_mlp, playground)

            """ ---- Build the pseudo loss function ---- """
            actor_loss = discrete_pseudo_loss(log_p_all, action_placeholder, advantage_placeholder,
                                              playground, name=vocab.actor_loss)

        # ::Continuous case
        elif isinstance(playground.env.action_space, gym.spaces.Box):
            raise NotImplementedError  # (Ice-Boxed) todo:implement -->  for policy for continuous space:

        # ::Other gym environment
        else:
            print("\n>>> The agent implementation does not support that environment space "
                  "{} yet.\n\n".format(playground.env.action_space))
            raise NotImplementedError

        """ ---- Actor optimizer ---- """
        actor_policy_optimizer_op = policy_optimizer(actor_loss, experiment_spec.learning_rate)

        """ ---- Actor summary ops ---- """
        # tf.summary.scalar('Sampled_action', sampled_action)                                     # \\\\\\    hlThis    \\\\\\
        tf.summary.scalar('Actor_loss', actor_loss)                                             # \\\\\\    hlThis    \\\\\\

    return sampled_action, theta_mlp, actor_loss, actor_policy_optimizer_op


def build_critic_graph(observation_placeholder: tf.Tensor, target_placeholder: tf.Tensor,
                       experiment_spec: ExperimentSpec) -> (tf.Tensor, tf.Tensor, tf.Operation):
    """
    The CRITIC graph

        1. Critic network phi
            input: the observations collected
            output: the logits of each action in the action space

        2. Critic loss
            input: the target y (either Monte Carlo target or Bootstraped estimate target)
            output: the Mean Squared Error (MSE)

    :return: critic, critic_loss, critic_optimizer

    """

    """ Tensor first dimension shape compatibility assessment """
    assert observation_placeholder.shape.as_list()[0] == target_placeholder.shape.as_list()[0], \
        "target_ph shape {} first dimension is NOT compatible with Obs_ph shape first dimension {} ".format(
            target_placeholder.shape.as_list()[0], observation_placeholder.shape.as_list()[0])

    with tf.name_scope(vocab.critic_network) as scope:
        """ ---- Build parameter PHI as a multilayer perceptron ---- """
        critic = build_MLP_computation_graph(observation_placeholder, 1,
                                             experiment_spec.theta_nn_h_layer_topo,
                                             hidden_layers_activation=experiment_spec.theta_hidden_layers_activation,
                                             output_layers_activation=experiment_spec.theta_output_layers_activation,
                                             name=vocab.phi_NeuralNet)

        """ ---- Build the Mean Square Error loss function ---- """
        with tf.name_scope(vocab.critic_loss):
            critic_loss = tf.reduce_mean((target_placeholder - critic) ** 2)

        """ ---- Critic optimizer ---- """
        critic_optimizer = tf_cv1.train.AdamOptimizer(learning_rate=experiment_spec['critic_learning_rate']
                                                      ).minimize(critic_loss, name=vocab.critic_optimizer)

        """ ---- Actor summary ops ---- """
        tf.summary.scalar('Critic_loss', critic_loss)                                         # \\\\\\    hlThis    \\\\\\

    return critic, critic_loss, critic_optimizer
