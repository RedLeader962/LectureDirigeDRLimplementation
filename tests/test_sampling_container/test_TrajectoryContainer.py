# coding=utf-8
from typing import Union, Any
from gym.wrappers import TimeLimit

from sample_container import TrajectoryCollector, UniformeBatchContainer
from DRL_building_bloc import ExperimentSpec, GymPlayground
from tests.test_sampling_container.conftest import take_one_random_step

import numpy as np


def test_TrajectoryContainer_INTERNAL(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformeBatchContainer
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for trj in range(10):

        events, done = take_one_random_step(env)

        trajectory_collector.collect(*events)
        trajectory_collector.trajectory_ended()

        trj_container = trajectory_collector.pop_trajectory_and_reset()
        assert trj_container.trajectory_id == trj
        assert len(trj_container) == 1


def test_TrajectoryContainer_ONE_STEP_POP_DUMMY(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformeBatchContainer
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    events, done = take_one_random_step(env)

    obs = np.ones((1, 4))
    act = 1
    rew = 1
    dummy_events = (obs, act, rew)

    trajectory_collector.collect(*dummy_events)
    trajectory_collector.trajectory_ended()

    trj_container = trajectory_collector.pop_trajectory_and_reset()
    assert trj_container.trajectory_id == 1
    assert len(trj_container) == 1