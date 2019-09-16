# coding=utf-8

import pytest
import DRL_building_bloc as bloc
import numpy as np

import tensorflow as tf
from tensorflow import keras
tf_cv1 = tf.compat.v1   # shortcut

@pytest.fixture
def tf_setup():
    in_p = tf_cv1.placeholder(tf.float32, shape=(None, 4))
    out_p = tf_cv1.placeholder(tf.float32, shape=(None, 2))
    nn_shape = (2, 2)

    return in_p, out_p, nn_shape

def test_build_MLP_computation_graph_io(tf_setup):
    in_p, out_p, nn_shape = tf_setup
    mlp_hidden_ops = bloc.build_MLP_computation_graph(in_p, out_p, nn_shape)

    print("\n\n>>> {}\n\n".format(mlp_hidden_ops))

    # keras_input = keras.Input(shape=(12,))
    # model = keras.Model(
    #     inputs=keras_input,
    #     outputs=mlp_hidden_ops
    # )




# def test_create_tensor():
#     ops_a = tf.add(3, 5)
#     print(ops_a)
#
#     with tf_cv1.Session() as sess:
#         print(sess.run(ops_a))