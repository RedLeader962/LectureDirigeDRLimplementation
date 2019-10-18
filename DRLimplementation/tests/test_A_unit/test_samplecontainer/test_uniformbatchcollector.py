# coding=utf-8
from typing import Union, Any

import pytest
from gym.wrappers import TimeLimit

from blocAndTools.container.samplecontainer import TrajectoryCollector, UniformBatchCollector
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from .conftest import take_one_random_step

import numpy as np


def test_UniformBatchCollector_INIT(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    state = uni_batch_collector.internal_state()
    assert len(state.trajectories_list) == 0
    assert state.timestep_count == 0
    assert state.trajectory_count == 0
    assert state.remaining_space == uni_batch_collector.CAPACITY


def test_UniformBatchCollector_ONE_STEP(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    events, done = take_one_random_step(env)
    trajectory_collector.collect(*events)
    trajectory_collector.trajectory_ended()
    aTrajectory = trajectory_collector.pop_trajectory_and_reset()

    uni_batch_collector.collect(aTrajectory)

    state = uni_batch_collector.internal_state()
    assert len(state.trajectories_list) == 1
    assert state.timestep_count == 1
    assert state.trajectory_count == 1
    assert state.remaining_space == uni_batch_collector.CAPACITY - 1
    assert uni_batch_collector.is_not_full()


def test_UniformBatchCollector_20_STEP_OVER_2_TRJ(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for trj in range(2):
        for step in range(10):
            events, done = take_one_random_step(env)
            trajectory_collector.collect(*events)

        trajectory_collector.trajectory_ended()
        aTrajectory = trajectory_collector.pop_trajectory_and_reset()

        uni_batch_collector.collect(aTrajectory)

    state = uni_batch_collector.internal_state()
    assert len(state.trajectories_list) == 2
    assert state.timestep_count == 20
    assert state.trajectory_count == 2
    assert state.remaining_space == uni_batch_collector.CAPACITY - 20

    assert uni_batch_collector.is_not_full()


def test_UniformBatchCollector_AT_BATCH_CAPACITY(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    uni_batch_collector = UniformBatchCollector(capacity=200)
    exp_spec, playground, trajectory_collector, _, env, _ = gym_discrete_setup

    for trj in range(2):
        for step in range(100):
            obs = np.ones((1, 4))
            act = 1
            rew = 1
            dummy_events = (obs, act, rew)

            trajectory_collector.collect(*dummy_events)

        trajectory_collector.trajectory_ended()
        aTrajectory = trajectory_collector.pop_trajectory_and_reset()

        uni_batch_collector.collect(aTrajectory)

    state = uni_batch_collector.internal_state()
    assert len(state.trajectories_list) == 2
    assert state.timestep_count == 200
    assert state.trajectory_count == 2
    assert state.remaining_space == uni_batch_collector.CAPACITY - 200

    assert not uni_batch_collector.is_not_full()

def test_UniformBatchCollector_AT_BATCH_CAPACITY_MINUS_ONE(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    uni_batch_collector = UniformBatchCollector(capacity=200)
    exp_spec, playground, trajectory_collector, _, env, _ = gym_discrete_setup

    for step in range(100):
        obs = np.ones((1, 4))
        act = 1
        rew = 1
        dummy_events = (obs, act, rew)
        trajectory_collector.collect(*dummy_events)

    trajectory_collector.trajectory_ended()
    aTrajectory_1 = trajectory_collector.pop_trajectory_and_reset()
    uni_batch_collector.collect(aTrajectory_1)

    for step in range(99):
        obs = np.ones((1, 4))
        act = 1
        rew = 1
        dummy_events = (obs, act, rew)
        trajectory_collector.collect(*dummy_events)

    trajectory_collector.trajectory_ended()
    aTrajectory_2 = trajectory_collector.pop_trajectory_and_reset()

    uni_batch_collector.collect(aTrajectory_2)

    state = uni_batch_collector.internal_state()
    assert len(state.trajectories_list) == 2
    assert state.timestep_count == 199
    assert state.trajectory_count == 2
    assert state.remaining_space == uni_batch_collector.CAPACITY - 199

    assert uni_batch_collector.is_not_full()


def test_UniformBatchCollector_OVER_CAPACITY(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    uni_batch_collector = UniformBatchCollector(capacity=200)
    exp_spec, playground, trajectory_collector, _, env, _ = gym_discrete_setup

    timestep_patern = [100, 100, 10]

    with pytest.raises(AssertionError):
        for trj in range(3):
            for step in range(timestep_patern[trj]):
                obs = np.ones((1, 4))
                act = 1
                rew = 1
                dummy_events = (obs, act, rew)

                trajectory_collector.collect(*dummy_events)

            trajectory_collector.trajectory_ended()
            aTrajectory = trajectory_collector.pop_trajectory_and_reset()

            uni_batch_collector.collect(aTrajectory)


def test_UniformBatchCollector_TWO_BATCHS(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    uni_batch_collector = UniformBatchCollector(capacity=200)
    exp_spec, playground, trajectory_collector, _, env, _ = gym_discrete_setup

    timestep_patern = [100, 100, 10]

    for trj in range(3):
        for step in range(timestep_patern[trj]):
            obs = np.ones((1, 4))
            act = 1
            rew = 1
            dummy_events = (obs, act, rew)

            trajectory_collector.collect(*dummy_events)

        trajectory_collector.trajectory_ended()
        aTrajectory = trajectory_collector.pop_trajectory_and_reset()

        if uni_batch_collector.is_not_full():
            uni_batch_collector.collect(aTrajectory)
        else:
            batch_container = uni_batch_collector.pop_batch_and_reset()
            uni_batch_collector.collect(aTrajectory)

    state = uni_batch_collector.internal_state()
    assert len(state.trajectories_list) == 1
    assert state.timestep_count == 10
    assert state.trajectory_count == 1
    assert state.remaining_space == uni_batch_collector.CAPACITY - 10

    assert uni_batch_collector.is_not_full()


def test_UniformBatchCollector_CUT_TRAJECTORY(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    uni_batch_collector = UniformBatchCollector(capacity=200)
    exp_spec, playground, trajectory_collector, _, env, _ = gym_discrete_setup

    timestep_patern = [100, 150]

    for trj in range(2):
        for step in range(timestep_patern[trj]):
            obs = np.ones((1, 4))
            act = 1
            rew = 1
            dummy_events = (obs, act, rew)

            trajectory_collector.collect(*dummy_events)

        trajectory_collector.trajectory_ended()
        aTrajectory = trajectory_collector.pop_trajectory_and_reset()

        uni_batch_collector.collect(aTrajectory)


    state = uni_batch_collector.internal_state()
    assert len(state.trajectories_list) == 2
    assert state.timestep_count == 200
    assert state.trajectory_count == 2
    assert state.remaining_space == uni_batch_collector.CAPACITY - 200

    assert not uni_batch_collector.is_not_full()




