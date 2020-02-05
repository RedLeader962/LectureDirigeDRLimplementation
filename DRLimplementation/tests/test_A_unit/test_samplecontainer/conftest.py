# coding=utf-8
from typing import Union, Any, Tuple
import pytest

import numpy as np
from gym.wrappers import TimeLimit

from blocAndTools.container.samplecontainer import TrajectoryCollector, UniformBatchCollector
from blocAndTools.container.samplecontainer_batch_advantage import (
    TrajectoryCollectorBatchAdvantage,
    UniformBatchCollectorBatchAdvantage,
    )
from blocAndTools.container.trajectories_pool import PoolManager, TimestepSample, SampleBatch, TrajectoriesPool
from blocAndTools.container.FAST_trajectories_pool import (
    Fast_PoolManager, Fast_SampleBatch, Fast_TimestepSample,
    Fast_TrajectoriesPool,
    )
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground


# <--(!) "conftest.py" is required for Pytest test discovery. DO NOT rename that file as it will break the functionality


@pytest.fixture(scope="function")
def gym_discrete_setup():
    exp_spec = ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    playground = GymPlayground('LunarLander-v2')
    
    trajectory_collector = TrajectoryCollector(exp_spec, playground)
    uni_batch_collector = UniformBatchCollector(capacity=exp_spec.batch_size_in_ts)
    
    env = playground.env
    initial_observation = env.reset()
    yield exp_spec, playground, trajectory_collector, uni_batch_collector, env, initial_observation


@pytest.fixture(scope="function")
def gym_discrete_batch_advantage_setup():
    exp_spec = ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    playground = GymPlayground('LunarLander-v2')
    
    trajectory_batch_adv_collector = TrajectoryCollectorBatchAdvantage(exp_spec, playground)
    uni_batch_batch_adv_collector = UniformBatchCollectorBatchAdvantage(capacity=exp_spec.batch_size_in_ts)
    
    env = playground.env
    initial_observation = env.reset()
    yield exp_spec, playground, trajectory_batch_adv_collector, uni_batch_batch_adv_collector, env, initial_observation


def take_one_random_step(env):
    action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
    observation, reward, done, _ = env.step(action)
    events = (observation, action, reward)
    return events, done


@pytest.fixture(scope="function")
def gym_continuous_pool_setup() -> (ExperimentSpec, GymPlayground, Union[PoolManager, Fast_PoolManager],
                                    Union[TimestepSample, Fast_TimestepSample],
                                    Union[TimestepSample, Fast_TimestepSample],
                                    Union[SampleBatch, Fast_SampleBatch],
                                    Union[TrajectoriesPool, Fast_TrajectoriesPool],
                                    Union[TimeLimit, Any], np.ndarray):
    """
    :return: (exp_spec, playground, poolmanager,
                timestepsampleOne, timestepsampleTwo,
                samplebatch, trajectoriespool,
                env, initial_observation)
    """
    exp_spec = ExperimentSpec(batch_size_in_ts=20, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2),
                              environment_name='LunarLanderContinuous-v2')
    exp_spec.set_experiment_spec({'pool_capacity': 100})
    playground = GymPlayground(exp_spec.prefered_environment)
    
    poolmanager = PoolManager(exp_spec, playground)
    timestepsampleOne = TimestepSample(container_id=1, playground=playground)
    timestepsampleTwo = TimestepSample(container_id=2, playground=playground)
    samplebatch = SampleBatch(batch_size=exp_spec.batch_size_in_ts, playground=playground)
    trajectoriespool = TrajectoriesPool(capacity=exp_spec['pool_capacity'], batch_size=exp_spec.batch_size_in_ts,
                                        playground=playground)
    
    env = playground.env
    initial_observation = env.reset()
    
    yield (exp_spec, playground, poolmanager,
           timestepsampleOne,
           timestepsampleTwo,
           samplebatch,
           trajectoriespool,
           env, initial_observation)
