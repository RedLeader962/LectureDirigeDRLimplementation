# coding=utf-8
from typing import Union, Any
from gym.wrappers import TimeLimit

from blocAndTools.container.samplecontainer import TrajectoryCollector, UniformBatchCollector
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from .conftest import take_one_random_step


def test_TrajectoryCollector_INIT(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == 0
    assert state.trj_collected == 0
    assert state.q_values_computed_on_current_trj is False


def test_TrajectoryCollector_ONE_STEP(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    events, done = take_one_random_step(env)
    trajectory_collector.collect_OAR(*events)

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == 1
    assert state.trj_collected == 0
    assert state.q_values_computed_on_current_trj is False


def test_TrajectoryCollector_ONE_STEP_END(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    events, done = take_one_random_step(env)
    trajectory_collector.collect_OAR(*events)

    trajectory_collector.trajectory_ended()
    trajectory_collector.compute_Qvalues_as_rewardToGo()

    step_nb = 1
    trj_nb = 1
    Qvalues_computed = True

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == step_nb
    assert state.trj_collected == trj_nb
    assert state.q_values_computed_on_current_trj is Qvalues_computed

    assert len(trajectory_collector.observations) == step_nb
    assert len(trajectory_collector.actions) == step_nb
    assert len(trajectory_collector.rewards) == step_nb
    assert trajectory_collector.lenght == step_nb


def test_TrajectoryCollector_ONE_STEP_POP(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    events, done = take_one_random_step(env)
    trajectory_collector.collect_OAR(*events)
    trajectory_collector.trajectory_ended()
    trajectory_collector.compute_Qvalues_as_rewardToGo()
    Trj_container = trajectory_collector.pop_trajectory_and_reset()

    step_nb = 1
    trj_nb = 1
    Qvalues_computed = False

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == step_nb
    assert state.trj_collected == trj_nb
    assert state.q_values_computed_on_current_trj is Qvalues_computed


def test_TrajectoryCollector_ONE_STEP_RESET(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    events, done = take_one_random_step(env)
    trajectory_collector.collect_OAR(*events)
    trajectory_collector.trajectory_ended()
    trajectory_collector.compute_Qvalues_as_rewardToGo()
    Trj_container = trajectory_collector.pop_trajectory_and_reset()

    assert len(trajectory_collector.observations) == 0
    assert len(trajectory_collector.actions) == 0
    assert len(trajectory_collector.rewards) == 0
    assert trajectory_collector.lenght is None


def test_TrajectoryCollector_10_STEP(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for _ in range(10):
        events, done = take_one_random_step(env)
        trajectory_collector.collect_OAR(*events)

    step_nb = 10
    trj_nb = 0
    Qvalues_computed = False

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == step_nb
    assert state.trj_collected == trj_nb
    assert state.q_values_computed_on_current_trj is Qvalues_computed


def test_TrajectoryCollector_10_STEP_END(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for _ in range(10):
        events, done = take_one_random_step(env)
        trajectory_collector.collect_OAR(*events)

    trajectory_collector.trajectory_ended()
    trajectory_collector.compute_Qvalues_as_rewardToGo()

    step_nb = 10
    trj_nb = 1
    Qvalues_computed = True

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == step_nb
    assert state.trj_collected == trj_nb
    assert state.q_values_computed_on_current_trj is Qvalues_computed

    assert len(trajectory_collector.observations) == step_nb
    assert len(trajectory_collector.actions) == step_nb
    assert len(trajectory_collector.rewards) == step_nb
    assert trajectory_collector.lenght == step_nb


def test_TrajectoryCollector_10_STEP_POP(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for _ in range(10):
        events, done = take_one_random_step(env)
        trajectory_collector.collect_OAR(*events)

    trajectory_collector.trajectory_ended()
    trajectory_collector.compute_Qvalues_as_rewardToGo()
    trajectory_collector.pop_trajectory_and_reset()

    step_nb = 10
    trj_nb = 1
    Qvalues_computed = False

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == step_nb
    assert state.trj_collected == trj_nb
    assert state.q_values_computed_on_current_trj is Qvalues_computed


def test_TrajectoryCollector_10_STEP_RESET(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for _ in range(10):
        events, done = take_one_random_step(env)
        trajectory_collector.collect_OAR(*events)

    trajectory_collector.trajectory_ended()
    trajectory_collector.compute_Qvalues_as_rewardToGo()
    trajectory_collector.pop_trajectory_and_reset()

    assert len(trajectory_collector.observations) == 0
    assert len(trajectory_collector.actions) == 0
    assert len(trajectory_collector.rewards) == 0
    assert trajectory_collector.lenght is None


def test_TrajectoryCollector_20_STEP_2_TRJ(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for trj in range(2):
        for _ in range(10):
            events, done = take_one_random_step(env)
            trajectory_collector.collect_OAR(*events)
        trajectory_collector.trajectory_ended()
        trajectory_collector.compute_Qvalues_as_rewardToGo()
        trajectory_collector.pop_trajectory_and_reset()

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == 20
    assert state.trj_collected == 2
    assert state.q_values_computed_on_current_trj is False


def test_TrajectoryCollector_20_STEP_END(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for _ in range(10):
        events, done = take_one_random_step(env)
        trajectory_collector.collect_OAR(*events)

    trajectory_collector.trajectory_ended()
    trajectory_collector.compute_Qvalues_as_rewardToGo()
    trajectory_collector.pop_trajectory_and_reset()

    for _ in range(10):
        events, done = take_one_random_step(env)
        trajectory_collector.collect_OAR(*events)

    trajectory_collector.trajectory_ended()
    # trajectory_collector.compute_Qvalues_as_rewardToGo()

    global_step = 20
    step_in_buffer = 10
    trj_nb = 2
    Qvalues_computed = False

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == global_step
    assert state.trj_collected == trj_nb
    assert state.q_values_computed_on_current_trj is Qvalues_computed

    assert len(trajectory_collector.observations) == step_in_buffer
    assert len(trajectory_collector.actions) == step_in_buffer
    assert len(trajectory_collector.rewards) == step_in_buffer
    assert trajectory_collector.lenght == step_in_buffer


def test_TrajectoryCollector_20_STEP_POP(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for trj in range(2):
        for _ in range(10):
            events, done = take_one_random_step(env)
            trajectory_collector.collect_OAR(*events)

        trajectory_collector.trajectory_ended()
        trajectory_collector.compute_Qvalues_as_rewardToGo()
        trajectory_collector.pop_trajectory_and_reset()

    global_step_nb = 20
    trj_nb = 2
    Qvalues_computed = False

    state = trajectory_collector.internal_state()
    assert state.step_count_since_begining_of_training == global_step_nb
    assert state.trj_collected == trj_nb
    assert state.q_values_computed_on_current_trj is Qvalues_computed


def test_TrajectoryCollector_20_STEP_RESET(gym_discrete_setup):
    # region ::Type hint bloc ...
    exp_spec: ExperimentSpec
    playground: GymPlayground
    trajectory_collector: TrajectoryCollector
    uni_batch_collector: UniformBatchCollector
    env: Union[TimeLimit, Any]
    # endregion

    exp_spec, playground, trajectory_collector, uni_batch_collector, env, _ = gym_discrete_setup

    for trj in range(2):
        for _ in range(10):
            events, done = take_one_random_step(env)
            trajectory_collector.collect_OAR(*events)

        trajectory_collector.trajectory_ended()
        trajectory_collector.compute_Qvalues_as_rewardToGo()
        trajectory_collector.pop_trajectory_and_reset()

    assert len(trajectory_collector.observations) == 0
    assert len(trajectory_collector.actions) == 0
    assert len(trajectory_collector.rewards) == 0
    assert trajectory_collector.lenght is None