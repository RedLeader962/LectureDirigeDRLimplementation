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

def test_Playground_continuous():
    play = bloc.BuildGymPlayground('LunarLanderContinuous-v2')
    assert play.ACTION_SPACE_SHAPE == (2,)
    assert play.OBSERVATION_SPACE_SHAPE == (8,)

def test_Playground_discreet():
    play = bloc.BuildGymPlayground('LunarLander-v2')
    assert play.ACTION_SPACE_SHAPE == 4


def test_build_MLP_computation_graph_io(tf_setup):
    _, out_p, nn_shape = tf_setup
    keras_input = keras.Input(shape=(12,))

    mlp_hidden_ops = bloc.build_MLP_computation_graph(keras_input, out_p, nn_shape)
    print("\n\n>>> {}\n\n".format(mlp_hidden_ops))
    model = keras.Model(inputs=keras_input, outputs=mlp_hidden_ops)
    # print(model.to_yaml())

# ---- playground_to_tensorflow_graph_adapter --------------------------------------------------------------------------------

def test_gym_env_to_tf_graph_adapter_WRONG_IMPORT_TYPE():
    with pytest.raises(AssertionError):
        bloc.playground_to_tensorflow_graph_adapter(gym)

def test_gym_env_to_tf_graph_adapter_CONTINUOUS_PASS(gym_continuous_setup):
    playground = gym_continuous_setup
    input_placeholder, output_placeholder = bloc.playground_to_tensorflow_graph_adapter(playground)
    assert input_placeholder.shape[-1] == playground.OBSERVATION_SPACE_SHAPE[0]
    assert output_placeholder.shape[-1] == playground.ACTION_SPACE_SHAPE[0]

def test_gym_env_to_tf_graph_adapter_DISCRETE_PASS(gym_discrete_setup):
    playground = gym_discrete_setup
    input_placeholder, output_placeholder = bloc.playground_to_tensorflow_graph_adapter(playground)
    assert input_placeholder.shape[-1] == playground.OBSERVATION_SPACE_SHAPE[0]
    print(output_placeholder.shape)
    assert output_placeholder.shape[-1] == playground.ACTION_SPACE_SHAPE


# ---- tensor experiment ------------------------------------------------------------------------------------------

def test_create_tensor():
    ops_a = tf.add(3, 5)
    # print(">>> ops_a: {}".format(ops_a))

    with tf_cv1.Session() as sess:
        # HINT: could be --> output_stream = sys.stderr
        print_ops = tf.print("ops_a:", ops_a, output_stream=sys.stdout)

        run_result = sess.run([ops_a, print_ops])

        print(">>> run result: {}".format(run_result))