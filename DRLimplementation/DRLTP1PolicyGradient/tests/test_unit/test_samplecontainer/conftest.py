# coding=utf-8

import pytest

from DRLimplementation.blocAndTools import TrajectoryCollector, UniformBatchCollector
from DRLimplementation.blocAndTools import ExperimentSpec, GymPlayground

# Do not rename that file (!) "conftest.py" is required for Pytest test discovery.

@pytest.fixture(scope="function")
def gym_discrete_setup():
    exp_spec = ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, neural_net_hidden_layer_topology=(2, 2))
    playground = GymPlayground('LunarLander-v2')

    trajectory_collector = TrajectoryCollector(exp_spec, playground)
    uni_batch_collector = UniformBatchCollector(capacity=exp_spec.batch_size_in_ts)

    env = playground.env
    initial_observation = env.reset()
    yield exp_spec, playground, trajectory_collector, uni_batch_collector, env, initial_observation

def take_one_random_step(env):
    action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
    observation, reward, done, _ = env.step(action)
    events = (observation, action, reward)
    return events, done




