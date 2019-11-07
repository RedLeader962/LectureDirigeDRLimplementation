# coding=utf-8
import gym
import tensorflow as tf

from blocAndTools.buildingbloc import (ExperimentSpec, GymPlayground, build_MLP_computation_graph,
                                       policy_theta_discrete_space, )
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

