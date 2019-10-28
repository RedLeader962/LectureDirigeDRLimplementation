# coding=utf-8
import pytest
import tensorflow as tf

from ActorCritic import ActorCriticBrainSplitNetwork
from blocAndTools import buildingbloc as bloc

tf_cv1 = tf.compat.v1   # shortcut

@pytest.fixture
def gym_and_tf_discrete_setup():
    """
    :return: (obs_p, act_p, exp_spec, playground)
    :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    exp_spec.set_experiment_spec({'critic_learning_rate': 1e-3})
    playground = bloc.GymPlayground('LunarLander-v2')
    obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                action_shape_constraint=(1,))
    yield obs_p, act_p, Q_values_ph, exp_spec, playground
    tf_cv1.reset_default_graph()

@pytest.fixture
def gym_and_tf_continuous_setup():
    """
    :return: (obs_p, act_p, exp_spec, playground)
    :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    exp_spec.set_experiment_spec({'critic_learning_rate': 1e-3})

    playground = bloc.GymPlayground('LunarLanderContinuous-v2')
    obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                action_shape_constraint=(1,))
    yield obs_p, act_p, exp_spec, playground
    tf_cv1.reset_default_graph()

# --- ActorCritic_agent -------------------------------------------------------------------------------------------
def test_ActorCritic_agent_ACTOR_DISCRETE_PASS(gym_and_tf_discrete_setup):

    obs_p, act_p, _, exp_spec, playground = gym_and_tf_discrete_setup
    A_ph = tf_cv1.placeholder(tf.float32, shape=(None,), name='advantage_placeholder')

    ActorCriticBrainSplitNetwork.build_actor_policy_graph(obs_p, exp_spec, playground)


def test_ActorCritic_agent_CRITIC_DISCRETE_PASS(gym_and_tf_discrete_setup):

    obs_p, _, Q_values_ph, exp_spec, playground = gym_and_tf_discrete_setup
    target_ph = tf_cv1.placeholder(tf.float32, shape=(None,), name='target_placeholder')

    ActorCriticBrainSplitNetwork.build_critic_graph(obs_p, exp_spec)




