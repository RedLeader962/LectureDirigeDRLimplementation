from typing import Union, Any

from gym.wrappers import TimeLimit

from samplecontainer import TrajectoryCollector, UniformBatchCollector
from buildingbloc import ExperimentSpec, GymPlayground

import numpy as np


def test_UniformBatchContainer_INIT(gym_discrete_setup):
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



def test_UniformBatchContainer_AT_BATCH_CAPACITY(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    uni_batch_collector = UniformBatchCollector(capacity=200)
    exp_spec, playground, trajectory_collector, _, env, _ = gym_discrete_setup

    trj = 0
    while uni_batch_collector.is_not_full():
        trj += 1
        for step in range(100):
            obs = np.ones((1, 4))
            act = 1
            rew = 1
            dummy_events = (obs, act, rew)

            trajectory_collector.collect(*dummy_events)

        trajectory_collector.trajectory_ended()
        aTrajectory = trajectory_collector.pop_trajectory_and_reset()

        uni_batch_collector.collect(aTrajectory)

    aBatch_contaner = uni_batch_collector.pop_batch_and_reset()

    assert len(aBatch_contaner) == 200
    assert aBatch_contaner.trajectories_count() == trj


def test_UniformBatchContainer_METRIC(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    uni_batch_collector = UniformBatchCollector(capacity=200)
    exp_spec, playground, trajectory_collector, _, env, _ = gym_discrete_setup

    trj = 0
    while uni_batch_collector.is_not_full():
        trj += 1
        for step in range(100):
            obs = np.ones((1, 4))
            act = 1
            rew = 1
            dummy_events = (obs, act, rew)

            trajectory_collector.collect(*dummy_events)

        trajectory_collector.trajectory_ended()
        aTrajectory = trajectory_collector.pop_trajectory_and_reset()

        uni_batch_collector.collect(aTrajectory)

    aBatch_contaner = uni_batch_collector.pop_batch_and_reset()

    batch_average_trjs_return, batch_average_trjs_lenght = aBatch_contaner.compute_metric()
    assert batch_average_trjs_return == 100
    assert batch_average_trjs_lenght == 100