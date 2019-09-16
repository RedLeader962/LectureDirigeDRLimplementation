#!/usr/bin/env python
import gym
import pretty_printing
import numpy as np

import tensorflow as tf
from tensorflow import keras
tf_cv1 = tf.compat.v1   # shortcut

from vocabulary import rl_name as vocab


class Playground(object):
    def __init__(self,
                 environment_name='LunarLanderContinuous-v2',
                 trajectory_batch_size=10,
                 max_trajectory_lenght=400,
                 max_timestep=2000,
                 random_seed=42):
        """
        Setup the learning playground for the agent:
            the environment in witch he will play and for how long

        Defaut environment: LunarLanderContinuous-v2
                env: <TimeLimit<LunarLanderContinuous<LunarLanderContinuous-v2>>>

                Metadata: {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

                REWARD range: (-inf, inf)

                ACTION SPACE:
                    Type: Box(2,)
                        Higher bound: [1. 1.]
                        Lower bound: [-1. -1.]

                Action is two floats [main engine, left-right engines].
                    Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power.
                                (!) Engine can't work with less than 50% power.
                    Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

                OBSERVATION SPACE:
                    Type: Box(8,)
                        Higher bound: [inf inf inf inf inf inf inf inf]
                        Lower bound: [-inf -inf -inf -inf -inf -inf -inf -inf]

        :param environment_name: a gym environment
        :type environment_name: str
        :param trajectory_batch_size:
        :type trajectory_batch_size: int
        :param max_trajectory_lenght:
        :type max_trajectory_lenght: int
        :param max_timestep:
        :type max_timestep: int
        :param random_seed:
        :type random_seed: int
        """

        self.ENVIRONMENT_NAME = environment_name
        self.TRAJECTORY_BATCH_SIZE = trajectory_batch_size
        self.TJ_BS = trajectory_batch_size                  # shortcut
        self.MAX_TRAJECTORY_LEN = max_trajectory_lenght
        self.max_TJ_len = max_trajectory_lenght             # shortcut
        self.MAX_TIMESTEP = max_timestep
        self.max_TS = max_timestep                          # shortcut
        self.RANDOM_SEED = random_seed
        self.seed = random_seed                             # shortcut

        self.env = gym.make(self.ENVIRONMENT_NAME)

        self.ACTION_SPACE_DIMENSION = len(self.env.action_space.high)
        self.OBSERVATION_SPACE_DIMENSION = len(self.env.observation_space.high)

        if self.ENVIRONMENT_NAME is 'LunarLanderContinuous-v2':
            action_space_doc = "Action is two floats [main engine, left-right engines].\n" \
                           "\tMain engine: -1..0 off, 0..+1 throttle from 50% to 100% power.\n" \
                           "\t\t\t\t(!) Engine can't work with less than 50% power.\n" \
                           "\tLeft-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off"
            info_str = pretty_printing.environnement_doc_str(self.env, action_space_doc)
        else:
            info_str = pretty_printing.environnement_doc_str(self.env)

        print(info_str)


def build_MLP_computation_graph(input_placeholder: tf.Tensor, output_placeholder: tf.Tensor, hidden_layer_topology: list,
                                hidden_layers_activation: tf.Tensor = tf.tanh,
                                output_layers_activation: tf.Tensor = tf.sigmoid) -> tf.Tensor:
    """
    Builder function for Low Level TensorFlow API.
    Return a Multi Layer Perceptron with topology:

        input_placeholder | *hidden_layer_topology | output_placeholder

    :param input_placeholder:
    :type input_placeholder:
    :param output_placeholder:
    :type output_placeholder:
    :param hidden_layer_topology:
    :type hidden_layer_topology:
    :param hidden_layers_activation:
    :type hidden_layers_activation:
    :param output_layers_activation:
    :type output_layers_activation:
    :return:
    :rtype:
    """

    with tf.name_scope(vocab.Multi_Layer_Perceptron) as scope:
        # Create input layer
        ops = tf_cv1.layers.Dense(hidden_layer_topology[0], input_shape=input_placeholder.shape,
                                  activation=hidden_layers_activation, name=vocab.input_layer)

        parent_layer = ops(input_placeholder)

        # create & connect all hidden layer
        for id in range(len(hidden_layer_topology)):
            h_layer = tf_cv1.layers.Dense(hidden_layer_topology[id], activation=hidden_layers_activation, name='{}{}'.format(vocab.hidden_, id + 1))
            parent_layer = h_layer(parent_layer)
            print(parent_layer) # todo-->remove

        # create & connect the ouput layer
        output_layer = tf_cv1.layers.Dense(output_placeholder.shape[-1], activation=output_layers_activation, name=vocab.output_layer)

        return output_layer(parent_layer)


def neural_network_policy_theta(observation, action, environment, hidden_layer_shape=(16, 16)):
    """"""
    raise NotImplementedError


def continuous_space_placeholder(space, name=None):
    return tf_cv1.placeholder(dtype=tf.float32, shape=(None, *space), name=name)


if __name__ == '__main__':
    """
    Start TensorBoard in terminal:
        tensorboard --logdir=DRL-TP1-Policy-Gradient/graph/
    
    In browser, go to:
        http://0.0.0.0:6006/ 
    """
    env = Playground().env

    """Hyperparameter"""
    batch_size = 10
    hidden_layer_topology = (4, 2, 2, 4)
    observation_space_size = env.observation_space.shape
    action_space = env.action_space.shape

    # fake input data
    input_data = np.ones((batch_size, *observation_space_size))
    # input_data = [[1, 1, 1], [1, 1, 1]]

    """Configure handle for feeding value to the computation graph
        Continuous space    -->     dtype=tf.float32
        Discreet scpace     -->     dtype=tf.int32
    """
    input_placeholder = continuous_space_placeholder(observation_space_size, name=vocab.input_placeholder)
    out_placeholder = continuous_space_placeholder(action_space, name=vocab.output_placeholder)

    """Build a Multi Layer Perceptron (MLP) as the policy parameter theta using a computation graph"""
    theta = build_MLP_computation_graph(input_placeholder, out_placeholder, hidden_layer_topology)

    writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    with tf_cv1.Session() as sess:
        # initialize random variable in the computation graph
        sess.run(tf_cv1.global_variables_initializer())

        # execute mlp computation graph with input data
        a = sess.run(theta, feed_dict={input_placeholder: input_data})

        print(a)

    writer.close()

