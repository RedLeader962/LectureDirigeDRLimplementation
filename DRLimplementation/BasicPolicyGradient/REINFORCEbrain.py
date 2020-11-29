# coding=utf-8
import gym
import tensorflow as tf

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground, build_MLP_computation_graph, \
    policy_theta_discrete_space, discrete_pseudo_loss

from blocAndTools.rl_vocabulary import rl_name
vocab = rl_name()


def REINFORCE_policy(observation_placeholder: tf.Tensor, action_placeholder: tf.Tensor, Q_values_placeholder: tf.Tensor,
                     experiment_spec: ExperimentSpec, playground: GymPlayground) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    The learning agent: REINFORCE (aka: Basic Policy Gradient)
    Based on the paper by Williams, R. J.
         Simple statistical gradient-following algorithms for connectionist reinforcement learning. (1992)

    Policy gradient is a on-policy method which seek to directly optimize the policy π_θ by using sampled trajectories τ
    as weight. Those weight will then be used to indicate how good the policy performed.
    Based on that knowledge, the algorithm update the parameter θ of his policy to make action leading to similar good
    trajectories more likely and similar bad trajectories less likely.
    In the case of Deep Reinforcement Learning, the policy parameter θ is a neural net.

    Input layer: state
    Output layer: action, stopGradient(Qvalues)

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
            pseudo_loss = discrete_pseudo_loss(log_p_all, action_placeholder, Q_values_placeholder, playground,
                                               vocab.pseudo_loss)

        # ::Continuous case
        elif isinstance(playground.env.action_space, gym.spaces.Box):
            raise NotImplementedError   # (Ice-Boxed) todo:implement -->  for policy for continuous space:

        # ::Other gym environment
        else:
            print("\n>>> The agent implementation does not support that environment space "
                  "{} yet.\n\n".format(playground.env.action_space))
            raise NotImplementedError

    return sampled_action, theta_mlp, pseudo_loss
