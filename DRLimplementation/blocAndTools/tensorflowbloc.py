# coding=utf-8
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from blocAndTools.buildingbloc import tf_cv1
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1  # shortcut
vocab = rl_name()


def update_nn_weights(graph_key_from: List[str], graph_key_to: List[str],
                      target_smoothing_coefficient: float) -> tf.Operation:
    """ Fetch all tensor in list graph_key_from and update tensor weight of those in graph_key_to
    Pre condition: Botch tensor graph key list must match
    """
    with tf_cv1.variable_scope('update_nn_weights_op', reuse=tf_cv1.AUTO_REUSE):
        op = tf.group(
            [tf_cv1.assign(
                updated_tensor,
                target_smoothing_coefficient * source_tensor + (1 - target_smoothing_coefficient) * updated_tensor)
                for source_tensor, updated_tensor in zip(graph_key_from, graph_key_to)]
            )
    return op


def get_variables_graph_key(name: str) -> List[str]:
    """
    Fetch the list of all parameter graph key under a specific variable name
    
        >>> the_V = build_MLP_computation_graph(obs_t_ph, 1, (4, 4), name='V_psi')
        >>> the_frozen_V = build_MLP_computation_graph(obs_t_prime_ph, 1, (4, 4), name='frozen_V_psi')
        >>> v_psy_key = get_variables_graph_key('V_psi')
        >>> print(v_psy_key)
        [<tf.Variable 'V_psi/hidden_1/kernel:0' shape=(4, 4) dtype=float32_ref>,
         <tf.Variable 'V_psi/hidden_1/bias:0' shape=(4,) dtype=float32_ref>,
         <tf.Variable 'V_psi/hidden_2/kernel:0' shape=(4, 4) dtype=float32_ref>,
         <tf.Variable 'V_psi/hidden_2/bias:0' shape=(4,) dtype=float32_ref>,
         <tf.Variable 'V_psi/logits/kernel:0' shape=(4, 1) dtype=float32_ref>,
         <tf.Variable 'V_psi/logits/bias:0' shape=(1,) dtype=float32_ref>]
    :param name: variable name
    :return: a list of all variable parameter graph key
    """
    scope_name = tf_cv1.get_variable_scope().name
    if len(scope_name) > 0:
        scope_name += '/' + name + '/'
    else:
        scope_name = name + '/'
    
    param_key = tf_cv1.get_collection(
        tf_cv1.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
    return param_key


def build_feed_dictionary(placeholders: list, arrays_of_values: list) -> dict:
    """
    Build a feed dictionary ready to use in a TensorFlow run session.

    It map TF placeholder to corresponding array of values so be advise, order is important.

    :param placeholders: a list of tensorflow placeholder
    :type placeholders: [tf.Tensor, ...]
    :param arrays_of_values: a list of array
    :type arrays_of_values: [array, ...]
    :return: a feed dictionary
    :rtype: dict
    """
    assert isinstance(placeholders, list), "Wrong input type, placeholders must be a list of tensorflow placeholder"
    assert isinstance(arrays_of_values, list), "Wrong input type, arrays_of_values must be a list of array"
    assert len(placeholders) == len(arrays_of_values), "placeholders and arrays_of_values must be of the same lenght"
    for placeholder in placeholders:
        assert isinstance(placeholder, tf.Tensor), ("Wrong input type, placeholders must "
                                                    "be a list of tensorflow placeholder")
    
    feed_dict = dict()
    for placeholder, array in zip(placeholders, arrays_of_values):
        feed_dict[placeholder] = array
    
    return feed_dict


def to_scalar(action_array: np.ndarray):
    return action_array.item()
