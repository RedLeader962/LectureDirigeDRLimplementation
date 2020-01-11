# coding=utf-8

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.python.util.deprecation as deprecation
from blocAndTools.buildingbloc import continuous_space_placeholder, build_MLP_computation_graph
from blocAndTools.tensorflowbloc import get_variables_graph_key
from blocAndTools import rl_name

tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()


class TfSessionInConsoleManager(object):
    """
    Make a session that WILL NOT be the default session
    Why:    1. There's no way to close the session automaticaly in PyCharm console
            2. To prevent conflict beetween unit-test run and console execution
    """
    
    def __init__(self):
        self._sessConsole = tf_cv1.Session()
    
    @property
    def session(self):
        return self._sessConsole
    
    def reset_console_session(self) -> None:
        self.__del__()
        self._sessConsole = tf_cv1.Session()
        return None
    
    def __del__(self):
        self._sessConsole.close()

# var1 = tf.get_variable('my_var1', (2,2))
# var2 = tf.get_variable('my_var2', (2,2))

# obs_t_ph = continuous_space_placeholder(env.observation_space)
# obs_t_prime_ph = continuous_space_placeholder(env.observation_space)
#
# V_psi = build_MLP_computation_graph(obs_t_ph, 1, (4, 4), name='V_spi')
# frozen_V_psi = build_MLP_computation_graph(obs_t_prime_ph, 1, (4, 4), name='frozen_V_psi')
