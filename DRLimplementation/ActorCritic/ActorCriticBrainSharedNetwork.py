# coding=utf-8
import gym
import tensorflow as tf

from blocAndTools.buildingbloc import (ExperimentSpec, GymPlayground, build_MLP_computation_graph,
                                       policy_theta_discrete_space, discrete_pseudo_loss, policy_optimizer)
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1  # shortcut
vocab = rl_name()


def build_actor_critic_shared_graph(observation_placeholder: tf.Tensor, experiment_spec: ExperimentSpec,
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

    :return: sampled_action, log_pi_all, shared_network, critic
    """
    """ ---- Assess the input shape compatibility ---- """
    are_compatible = observation_placeholder.shape.as_list()[-1] == playground.OBSERVATION_SPACE.shape[0]
    assert are_compatible, ("the observation_placeholder is incompatible with environment, "
                            "{} != {}").format(observation_placeholder.shape.as_list()[-1],
                                               playground.OBSERVATION_SPACE.shape[0])

    """ ---- Build parameter THETA as a multilayer perceptron ---- """
    shared_network = build_MLP_computation_graph(observation_placeholder, playground.ACTION_CHOICES,
                                                 experiment_spec.theta_nn_h_layer_topo,
                                                 hidden_layers_activation=experiment_spec.theta_hidden_layers_activation,
                                                 output_layers_activation=experiment_spec.theta_output_layers_activation,
                                                 name=vocab.shared_network)

    # ::Discrete case
    if isinstance(playground.env.action_space, gym.spaces.Discrete):

        """ ---- Build the policy for discrete space ---- """
        sampled_action, log_pi_all = policy_theta_discrete_space(shared_network, playground)

    # ::Continuous case
    elif isinstance(playground.env.action_space, gym.spaces.Box):
        raise NotImplementedError  # (Ice-Boxed) todo:implement -->  for policy for continuous space:

    # ::Other gym environment
    else:
        print("\n>>> The agent implementation does not support that environment space "
              "{} yet.\n\n".format(playground.env.action_space))
        raise NotImplementedError

    """ ---- Build the Critic ---- """
    critic = build_MLP_computation_graph(shared_network, 1,
                                         (),
                                         hidden_layers_activation=experiment_spec.theta_hidden_layers_activation,
                                         output_layers_activation=experiment_spec.theta_output_layers_activation,
                                         name=vocab.V_estimate)

    return sampled_action, log_pi_all, shared_network, critic

