# coding=utf-8
import pytest
import tensorflow as tf

tf_cv1 = tf.compat.v1   # shortcut

from BasicPolicyGradient import REINFORCEbrain
from blocAndTools import buildingbloc as bloc


@pytest.fixture
def gym_and_tf_continuous_setup():
    """
    :return: (obs_p, act_p, exp_spec, playground)
    :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    playground = bloc.GymPlayground('LunarLanderContinuous-v2')
    obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
        playground, action_shape_constraint=(1,))
    yield obs_p, act_p, exp_spec, playground
    tf_cv1.reset_default_graph()

@pytest.fixture
def gym_and_tf_discrete_setup():
    """
    :return: (obs_p, act_p, exp_spec, playground)
    :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    playground = bloc.GymPlayground('LunarLander-v2')
    obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
        playground, action_shape_constraint=(1,))
    yield obs_p, act_p, exp_spec, playground
    tf_cv1.reset_default_graph()


# --- REINFORCED_agent -------------------------------------------------------------------------------------------
def test_REINFORCE_agent_DISCRETE_PASS(gym_and_tf_discrete_setup):

    obs_p, act_p, exp_spec, playground = gym_and_tf_discrete_setup
    q_values_p = tf_cv1.placeholder(tf.float32, shape=(None,), name='q_values_placeholder')

    reinforce_policy = REINFORCEbrain.REINFORCE_policy(obs_p, act_p, q_values_p, exp_spec, playground)
    sampled_action, theta_mlp, pseudo_loss = reinforce_policy



