# coding=utf-8
from typing import Union, Any
from gym.wrappers import TimeLimit

from sample_container import TrajectoryCollector, UniformBatchCollector
from DRL_building_bloc import ExperimentSpec, GymPlayground
from tests.test_sampling_container.conftest import take_one_random_step

import numpy as np


def test_TrajectoryContainer_INTERNAL(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for trj in range(10):

        events, done = take_one_random_step(env)

        trajectory_collector.collect(*events)
        trajectory_collector.trajectory_ended()

        trj_container = trajectory_collector.pop_trajectory_and_reset()
        assert trj_container.trajectory_id == trj + 1
        assert len(trj_container) == 1


def test_TrajectoryContainer_ONE_STEP_POP_DUMMY(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    obs = np.ones((1, 4))
    act = 1
    rew = 1
    dummy_events = (obs, act, rew)

    trajectory_collector.collect(*dummy_events)
    trajectory_collector.trajectory_ended()

    trj_container = trajectory_collector.pop_trajectory_and_reset()
    assert trj_container.trajectory_id == 1
    assert len(trj_container) == 1

    observations, actions, rewards, Q_values, trajectory_return, trajectory_lenght = trj_container.unpack()

    assert np.equal(obs, observations[0]).any()
    assert act == actions[0]
    assert rew == rewards[0]
    assert 1 == Q_values[0]
    assert 1 == trajectory_return


def test_TrajectoryContainer_10_STEP_DUMMY(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    test_trj_lenght = 10
    for step in range(test_trj_lenght):
        obs = np.ones((1, 4))
        act = 1
        rew = 1
        dummy_events = (obs, act, rew)

        trajectory_collector.collect(*dummy_events)

    trajectory_collector.trajectory_ended()

    trj_container = trajectory_collector.pop_trajectory_and_reset()
    assert trj_container.trajectory_id == 1
    assert len(trj_container) == 10

    observations, actions, rewards, Q_values, trajectory_return, trajectory_lenght = trj_container.unpack()

    for step in range(test_trj_lenght):
        assert np.equal(obs, observations[step]).any()
        assert act == actions[step]
        assert rew == rewards[step]

    assert 10 == len(Q_values)
    assert 10 == trajectory_return


def test_TrajectoryContainer_10_STEP_ON_SECOND_TRJ_DUMMY(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    test_trj_lenght = [5, 10]
    for trj in range(2):
        for step in range(test_trj_lenght[trj]):
            obs = np.ones((1, 4))
            act = 1
            rew = 1
            dummy_events = (obs, act, rew)

            trajectory_collector.collect(*dummy_events)

        trajectory_collector.trajectory_ended()

        trj_container = trajectory_collector.pop_trajectory_and_reset()

        assert trj_container.trajectory_id == trj + 1
        assert len(trj_container) == test_trj_lenght[trj]

    # Inspect container
    observations, actions, rewards, Q_values, trajectory_return, trajectory_lenght = trj_container.unpack()

    for step in range(test_trj_lenght[1]):
        assert np.equal(obs, observations[step]).any()
        assert act == actions[step]
        assert rew == rewards[step]

    assert 10 == len(Q_values)
    assert 10 == trajectory_return



def test_TrajectoryContainer_CUT(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    lenght_before_cut = 20
    for step in range(lenght_before_cut):
        obs = np.ones((1, 4))
        act = 1
        rew = 1
        dummy_events = (obs, act, rew)

        trajectory_collector.collect(*dummy_events)

    trajectory_collector.trajectory_ended()
    trj_container = trajectory_collector.pop_trajectory_and_reset()

    # Inspect container
    assert lenght_before_cut == len(trj_container)
    observations, actions, rewards, Q_values, trajectory_return, trajectory_lenght = trj_container.unpack()
    assert lenght_before_cut == len(observations)
    assert lenght_before_cut == len(actions)
    assert lenght_before_cut == len(rewards)
    assert lenght_before_cut == len(Q_values)

    lenght_after_cut = 9
    trj_container.cut(lenght_after_cut)

    # Inspect container
    assert lenght_after_cut == len(trj_container)
    observations, actions, rewards, Q_values, trajectory_return, trajectory_lenght = trj_container.unpack()
    assert lenght_after_cut == len(observations)
    assert lenght_after_cut == len(actions)
    assert lenght_after_cut == len(rewards)
    assert lenght_after_cut == len(Q_values)

