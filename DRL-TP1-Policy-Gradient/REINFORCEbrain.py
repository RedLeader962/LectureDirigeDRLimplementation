# coding=utf-8
import gym
import tensorflow as tf

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground, vocab, build_MLP_computation_graph, \
    policy_theta_discrete_space, discrete_pseudo_loss


def REINFORCE_policy(observation_placeholder: tf.Tensor, action_placeholder: tf.Tensor, Q_values_placeholder: tf.Tensor,
                     experiment_spec: ExperimentSpec, playground: GymPlayground) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    The learning agent: REINFORCE (aka: Vanila policy gradient)
    Based on the paper by Williams, R. J.
         Simple statistical gradient-following algorithms for connectionist reinforcement learning. (1992)

    :type observation_placeholder: tf.Tensor
    :type action_placeholder: tf.Tensor
    :type Q_values_placeholder: tf.Tensor
    :type playground: GymPlayground
    :type experiment_spec: ExperimentSpec
    :return: (sampled_action, theta_mlp, pseudo_loss)
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
    """
    with tf.name_scope(vocab.REINFORCE) as scope:

        """ ---- Build parameter theta as a multilayer perceptron ---- """
        theta_mlp = build_MLP_computation_graph(observation_placeholder, playground,
                                                experiment_spec.nn_h_layer_topo,
                                                hidden_layers_activation=experiment_spec.hidden_layers_activation,
                                                output_layers_activation=experiment_spec.output_layers_activation,
                                                name_scope=vocab.theta_NeuralNet)

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
            pseudo_loss = discrete_pseudo_loss(log_p_all, action_placeholder, Q_values_placeholder, playground)

        # ::Continuous case
        elif isinstance(playground.env.action_space, gym.spaces.Box):
            raise NotImplementedError   # (Ice-Boxed) todo:implement -->  for policy for continuous space:

        # ::Other gym environment
        else:
            print("\n>>> The agent implementation does not support environment space "
                  "{} yet.\n\n".format(playground.env.action_space))
            raise NotImplementedError

    return sampled_action, theta_mlp, pseudo_loss