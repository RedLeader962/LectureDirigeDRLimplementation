#!/usr/bin/env python

import gym
import pretty_printing
import numpy as np

import tensorflow_weak_warning_supressor as no_cpu_compile_warn
no_cpu_compile_warn.execute()

import tensorflow as tf
from tensorflow import keras
tf_cv1 = tf.compat.v1   # shortcut

from vocabulary import rl_name as vocab


class ExperimentSpec(object):
    def __init__(self, trajectory_batch_size=10, max_trajectory_lenght=400,
                 max_timestep=2000, random_seed=42, neural_net_hidden_layer_topology: list =[32, 32]):
        """
        Gather the specification for a experiement

        :param trajectory_batch_size:
        :type trajectory_batch_size:
        :param max_trajectory_lenght:
        :type max_trajectory_lenght:
        :param max_timestep:
        :type max_timestep:
        :param random_seed:
        :type random_seed:
        :param neural_net_hidden_layer_topology:
        :type neural_net_hidden_layer_topology:
        """

        self.TRAJECTORY_BATCH_SIZE = trajectory_batch_size
        self.tj_bs = trajectory_batch_size  # shortcut
        self.MAX_TRAJECTORY_LEN = max_trajectory_lenght
        self.max_TJ_len = max_trajectory_lenght  # shortcut
        self.MAX_TIMESTEP = max_timestep
        self.max_TS = max_timestep  # shortcut
        self.RANDOM_SEED = random_seed
        self.seed = random_seed  # shortcut
        self.NEURAL_NET_HIDDEN_LAYER_TOPOLOGY = neural_net_hidden_layer_topology
        self.nn_h_layer_topo = neural_net_hidden_layer_topology  # shortcut

        assert isinstance(neural_net_hidden_layer_topology, list)


class GymPlayground(object):
    def __init__(self, environment_name='LunarLanderContinuous-v2'):
        """
        Setup the learning playground for the agent (the environment in witch he will play) and gather relevant spec

        Note: 'LunarLanderContinuous-v2' (DEFAUT environment)
                env: <TimeLimit<LunarLanderContinuous<LunarLanderContinuous-v2>>>
                Metadata:
                    {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

                OBSERVATION SPACE:
                    Type: Box(8,)
                        Higher bound: [inf inf inf inf inf inf inf inf]
                        Lower bound: [-inf -inf -inf -inf -inf -inf -inf -inf]

                ACTION SPACE:
                    Type: Box(2,)
                        Higher bound: [1. 1.]
                        Lower bound: [-1. -1.]

                Action is two floats [main engine, left-right engines].
                    Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power.
                                (!) Engine can't work with less than 50% power.
                    Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

                REWARD range: (-inf, inf)

        Note: 'LunarLander-v2' (Discrete version):
                env: <TimeLimit<LunarLander<LunarLander-v2>>>
                Metadata:
                    {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

            OBSERVATION SPACE:
                Type: Box(8,)
                    Higher bound: [inf inf inf inf inf inf inf inf]
                    Lower bound: [-inf -inf -inf -inf -inf -inf -inf -inf]

            ACTION SPACE:
                Type: Discrete(4)
                    Higher bound: 1
                    Lower bound: 0

            REWARD range: (-inf, inf)

        :param environment_name: a gym environment
        :type environment_name: str
        """

        self.ENVIRONMENT_NAME = environment_name

        try:
            self.env = gym.make(self.ENVIRONMENT_NAME)
        except gym.error.Error as e:
            raise gym.error.Error("GymPlayground did not find the specified Gym environment.") from e


        if isinstance(self.env.action_space, gym.spaces.Box):
            print("\n\n>>> Action space is Contiuous")
            self.ACTION_SPACE_SHAPE = self.env.action_space.shape
        else:
            print("\n\n>>> Action space is Discrete")
            self.ACTION_SPACE_SHAPE = self.env.action_space.n

        self.OBSERVATION_SPACE_SHAPE = self.env.observation_space.shape

        if self.ENVIRONMENT_NAME == 'LunarLanderContinuous-v2':
            action_space_doc = "\tAction is two floats [main engine, left-right engines].\n" \
                           "\tMain engine: -1..0 off, 0..+1 throttle from 50% to 100% power.\n" \
                           "\t\t\t\t(!) Engine can't work with less than 50% power.\n" \
                           "\tLeft-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off\n\n"
            info_str = pretty_printing.environnement_doc_str(self.env, action_space_doc=action_space_doc)
        else:
            info_str = pretty_printing.environnement_doc_str(self.env)

        print(info_str)


def build_MLP_computation_graph(input_placeholder: tf.Tensor, output_placeholder: tf.Tensor,
                                hidden_layer_topology: list,
                                hidden_layers_activation: tf.Tensor = tf.tanh,
                                output_layers_activation: tf.Tensor = tf.sigmoid) -> tf.Tensor:
    """
    Builder function for Low Level TensorFlow API.
    Return a Multi Layer Perceptron computatin graph with topology:

        input_placeholder | *hidden_layer_topology | output_placeholder

    It's last layer is called the 'logits' (aka: the raw output of the MLP)

    In the context of deep learning, 'logits' is the equivalent of 'raw output' of our prediction.
    It will later be transform into probabilies using the 'softmax function'

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
    :return: a well construct computation graph
    :rtype: tf.Tensor
    """

    with tf.name_scope(vocab.Multi_Layer_Perceptron) as scope:
        # Create input layer
        ops = keras.layers.Dense(hidden_layer_topology[0], input_shape=input_placeholder.shape,
                                  activation=hidden_layers_activation, name=vocab.input_layer)

        parent_layer = ops(input_placeholder)

        # create & connect all hidden layer
        for id in range(len(hidden_layer_topology)):
            h_layer = keras.layers.Dense(hidden_layer_topology[id], activation=hidden_layers_activation, name='{}{}'.format(vocab.hidden_, id + 1))
            parent_layer = h_layer(parent_layer)
            print(parent_layer) # todo-->remove

        # create & connect the ouput layer: the logits
        logits = keras.layers.Dense(output_placeholder.shape[-1], activation=output_layers_activation, name=vocab.logits)

        return logits(parent_layer)


def continuous_space_placeholder(space, name=None):
    return tf_cv1.placeholder(dtype=tf.float32, shape=(None, *space), name=name)

def discrete_space_placeholder(space: tuple, name=None):
    if isinstance(space, tuple):
        space = space[0]
    return tf_cv1.placeholder(dtype=tf.int32, shape=(None, space), name=name)

def gym_playground_to_tensorflow_graph_adapter(playground: GymPlayground) -> (tf_cv1.placeholder, tf_cv1.placeholder):
    """
    Configure handle for feeding value to the computation graph
            Continuous space    -->     dtype=tf.float32
            Discrete scpace     -->     dtype=tf.int32
    :param playground:
    :type playground: GymPlayground
    :return: input_placeholder, output_placeholder
    :rtype: (tf.placeholder, tf.placeholder)
    """
    assert isinstance(playground, GymPlayground), "\n\n>>> gym_playground_to_tensorflow_graph_adapter() expected a builded GymPlayground.\n\n"

    if isinstance(playground.env.observation_space, gym.spaces.Box):
        """observation space is continuous"""
        input_placeholder = continuous_space_placeholder(playground.OBSERVATION_SPACE_SHAPE, name=vocab.input_placeholder)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """observation space is discrete"""
        input_placeholder = discrete_space_placeholder(playground.OBSERVATION_SPACE_SHAPE, name=vocab.input_placeholder)
    else:
        raise NotImplementedError

    if isinstance(playground.env.action_space, gym.spaces.Box):
        """action space is continuous"""
        output_placeholder = continuous_space_placeholder(playground.ACTION_SPACE_SHAPE, name=vocab.output_placeholder)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """action space is discrete"""
        output_placeholder = discrete_space_placeholder(playground.ACTION_SPACE_SHAPE, name=vocab.output_placeholder)
    else:
        raise NotImplementedError

    return input_placeholder, output_placeholder


def policy_theta_discrete_space(playground: GymPlayground, neural_net_hyperparam: dict):
    playground = GymPlayground()
    observation_placeholder, action_placeholder = gym_playground_to_tensorflow_graph_adapter(playground)
    build_MLP_computation_graph(observation_placeholder, action_placeholder)



if __name__ == '__main__':
    """
    Start TensorBoard in terminal:
        tensorboard --logdir=DRL-TP1-Policy-Gradient/graph/
    
    In browser, go to:
        http://0.0.0.0:6006/ 
    """
    exp_spec = ExperimentSpec()
    continuous_play = GymPlayground(environment_name='LunarLander-v2')


    # (!) fake input data
    input_data = np.ones((exp_spec.TRAJECTORY_BATCH_SIZE, *continuous_play.OBSERVATION_SPACE_SHAPE))
    # input_data = [[1, 1, 1], [1, 1, 1]]

    input_placeholder, output_placeholder = gym_playground_to_tensorflow_graph_adapter(continuous_play)

    # todo: we are here
    """Build a Multi Layer Perceptron (MLP) as the policy parameter theta using a computation graph"""
    theta = build_MLP_computation_graph(input_placeholder, output_placeholder, exp_spec.nn_h_layer_topo)

    writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    with tf_cv1.Session() as sess:
        # initialize random variable in the computation graph
        sess.run(tf_cv1.global_variables_initializer())

        # execute mlp computation graph with input data
        a = sess.run(theta, feed_dict={input_placeholder: input_data})

        print(a)

    writer.close()

