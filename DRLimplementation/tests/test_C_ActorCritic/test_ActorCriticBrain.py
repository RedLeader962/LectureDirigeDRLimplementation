# coding=utf-8
import pytest
import tensorflow as tf

tf_cv1 = tf.compat.v1   # shortcut

from ActorCritic import ActorCriticBrain
from blocAndTools import buildingbloc as bloc


@pytest.fixture(scope="function")
def gym_and_tf_continuous_setup():
    """
    :return: (obs_p, act_p, exp_spec, playground)
    :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, neural_net_hidden_layer_topology=(2, 2))
    playground = bloc.GymPlayground('LunarLanderContinuous-v2')
    obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground, (1,))
    yield obs_p, act_p, exp_spec, playground
    tf_cv1.reset_default_graph()

@pytest.fixture(scope="function")
def gym_and_tf_discrete_setup():
    """
    :return: (obs_p, act_p, exp_spec, playground)
    :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, neural_net_hidden_layer_topology=(2, 2))
    playground = bloc.GymPlayground('LunarLander-v2')
    obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground, (1,))
    yield obs_p, act_p, exp_spec, playground
    tf_cv1.reset_default_graph()


# --- ActorCritic_agent -------------------------------------------------------------------------------------------
def test_ActorCritic_agent_ACTOR_DISCRETE_PASS(gym_and_tf_discrete_setup):

    obs_p, act_p, exp_spec, playground = gym_and_tf_discrete_setup
    A_ph = tf_cv1.placeholder(tf.float32, shape=(None,), name='advantage_placeholder')

    actor_policy = ActorCriticBrain.actor_policy(obs_p, act_p, A_ph, exp_spec, playground)
    sampled_action, theta_mlp, actor_loss = actor_policy


def test_ActorCritic_agent_CRITIC_DISCRETE_PASS(gym_and_tf_discrete_setup):

    obs_p, act_p, exp_spec, playground = gym_and_tf_discrete_setup
    A_ph = tf_cv1.placeholder(tf.float32, shape=(None,), name='advantage_placeholder')

    critic_policy = ActorCriticBrain.critic(obs_p, act_p, A_ph, exp_spec, playground)
    sampled_action, theta_mlp, critic_loss = critic_policy



