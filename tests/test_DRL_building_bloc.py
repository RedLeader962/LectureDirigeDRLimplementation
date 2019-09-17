# coding=utf-8
import sys

import gym
import pytest
import DRL_building_bloc as bloc
import numpy as np

import tensorflow as tf
from tensorflow import keras
tf_cv1 = tf.compat.v1   # shortcut


# ---- setup & teardown -------------------------------------------------------------------------------------------

@pytest.fixture
def tf_setup():
    in_p = tf_cv1.placeholder(tf.float32, shape=(None, 8))
    out_p = tf_cv1.placeholder(tf.float32, shape=(None, 2))
    nn_shape = (2, 2)

    return in_p, out_p, nn_shape

@pytest.fixture
def gym_continuous_setup():
    return bloc.BuildGymPlayground('LunarLanderContinuous-v2')

@pytest.fixture
def gym_discrete_setup():
    return bloc.BuildGymPlayground('LunarLander-v2')

# ---- playground ------------------------------------------------------------------------------------------------

def test_Playground_init_ENV_FAIL():
    with pytest.raises(Exception):
        bloc.BuildGymPlayground('UnExistingEnvironment!!!')

def test_Playground_init_ENV_TOPOLOGY_FAIL():
    with pytest.raises(Exception):
        bloc.BuildGymPlayground('LunarLanderContinuous-v2', neural_net_hidden_layer_topology=(1,))


def test_Playground_continuous():
    play = bloc.BuildGymPlayground('LunarLanderContinuous-v2')
    assert play.ACTION_SPACE_SHAPE == (2,)
    assert play.OBSERVATION_SPACE_SHAPE == (8,)

def test_Playground_discreet():
    play = bloc.BuildGymPlayground('LunarLander-v2')
    assert play.ACTION_SPACE_SHAPE == 4


# ---- gym_playground_to_tensorflow_graph_adapter --------------------------------------------------------------------

def test_gym_env_to_tf_graph_adapter_WRONG_IMPORT_TYPE():
    with pytest.raises(AssertionError):
        bloc.gym_playground_to_tensorflow_graph_adapter(gym)

def test_gym_env_to_tf_graph_adapter_DISCRETE_PASS(gym_discrete_setup):
    playground = gym_discrete_setup
    input_placeholder, output_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    assert input_placeholder.shape[-1] == playground.OBSERVATION_SPACE_SHAPE[0]
    print(output_placeholder.shape)
    assert output_placeholder.shape[-1] == playground.ACTION_SPACE_SHAPE

def test_gym_env_to_tf_graph_adapter_CONTINUOUS_PASS(gym_continuous_setup):
    playground = gym_continuous_setup
    input_placeholder, output_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    assert input_placeholder.shape[-1] == playground.OBSERVATION_SPACE_SHAPE[0]
    assert output_placeholder.shape[-1] == playground.ACTION_SPACE_SHAPE[0]


# --- build_MLP_computation_graph ------------------------------------------------------------------------------------

def test_build_MLP_computation_graph_io(tf_setup):
    _, out_p, nn_shape = tf_setup
    keras_input = keras.Input(shape=(12,))

    mlp_hidden_ops = bloc.build_MLP_computation_graph(keras_input, out_p, nn_shape)
    print("\n\n>>> {}\n\n".format(mlp_hidden_ops))
    model = keras.Model(inputs=keras_input, outputs=mlp_hidden_ops)
    # print(model.to_yaml())


def test_build_MLP_computation_graph_with_DISCRETE_adapter(gym_discrete_setup):
    input_placeholder, out_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(gym_discrete_setup)
    bloc.build_MLP_computation_graph(input_placeholder, out_placeholder, hidden_layer_topology=[2,2])

def test_build_MLP_computation_graph_with_CONTINUOUS_adapter(gym_continuous_setup):
    input_placeholder, out_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(gym_continuous_setup)
    bloc.build_MLP_computation_graph(input_placeholder, out_placeholder, hidden_layer_topology=[2,2])




def test_integration_Playground_to_adapter_to_build_graph():
    continuous_play = bloc.BuildGymPlayground(trajectory_batch_size=10,
                                         neural_net_hidden_layer_topology=[2, 2])

    # (!) fake input data
    input_data = np.ones((continuous_play.TRAJECTORY_BATCH_SIZE, *continuous_play.OBSERVATION_SPACE_SHAPE))

    input_placeholder, out_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(continuous_play)

    """Build a Multi Layer Perceptron (MLP) as the policy parameter theta using a computation graph"""
    theta = bloc.build_MLP_computation_graph(input_placeholder, out_placeholder, continuous_play.nn_h_layer_topo)

    # writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    with tf_cv1.Session() as sess:
        # initialize random variable in the computation graph
        sess.run(tf_cv1.global_variables_initializer())

        # execute mlp computation graph with input data
        a = sess.run(theta, feed_dict={input_placeholder: input_data})

        # print("\n\n>>>run theta:\n{}\n\n".format(a))
    # writer.close()


# ---- tensor experiment ------------------------------------------------------------------------------------------

def test_create_tensor():
    ops_a = tf.add(3, 5)
    # print(">>> ops_a: {}".format(ops_a))

    with tf_cv1.Session() as sess:
        # HINT: could be --> output_stream = sys.stderr
        print_ops = tf.print("ops_a:", ops_a, output_stream=sys.stdout)

        run_result = sess.run([ops_a, print_ops])

        print(">>> run result: {}\n\n".format(run_result))