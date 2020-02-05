# coding=utf-8
import random
from copy import copy
from typing import Union, Any, Tuple

import pytest
from gym.wrappers import TimeLimit

from blocAndTools.container.FAST_trajectories_pool import (
    Fast_PoolManager, Fast_TimestepSample, Fast_SampleBatch,
    Fast_TrajectoriesPool,
    )
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground

import numpy as np

from .pool_testingUtility import step_foward_and_collect


# note: exp_spec key specific to SAC
#   |   'pool_capacity'


def test_Trajectories_pool_init(gym_continuous_pool_setup):
    (exp_spec, playground, poolmanager,
     timestepsampleOne,
     timestepsampleTwo,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    for idx in range(exp_spec['pool_capacity']):
        tss: Fast_TimestepSample = trajectoriespool._pool[idx]
        assert tss._container_id == idx + 1
    
    assert trajectoriespool.size == 0


def test_TrajectoriesPool_COLLECT_20(gym_continuous_pool_setup):
    (exp_spec, playground, poolmanager,
     timestepsampleOne,
     timestepsampleTwo,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    obs_t = initial_observation
    for idx in range(20):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, trajectoriespool)
        if idx < 19:
            obs_t = obs_t_prime
        else:
            # assess collected stuff authenticity
            tss20: Fast_TimestepSample = trajectoriespool._pool[19]
            assert 20 is tss20._container_id
            assert obs_t is tss20.obs_t
            assert act_t is tss20.act_t
            assert obs_t_prime is tss20.obs_t_prime
            assert rew_t is tss20.rew_t
            assert done_t is tss20.done_t
            obs_t = obs_t_prime
    
    for idx in range(20):
        tss: Fast_TimestepSample = trajectoriespool._pool[idx]
        assert tss._container_id == idx + 1
    
    assert trajectoriespool._idx == 20
    assert trajectoriespool.size == 20
    assert trajectoriespool._pool_full() is False


def test_TrajectoriesPool_COLLECT_TO_CAPACITY(gym_continuous_pool_setup):
    (exp_spec, playground, poolmanager,
     timestepsampleOne,
     timestepsampleTwo,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    obs_t = initial_observation
    capacity = exp_spec['pool_capacity']
    for idx in range(capacity):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, trajectoriespool)
        last_item_idx = capacity - 1
        if idx < last_item_idx:
            obs_t = obs_t_prime
        else:
            # assess collected stuff authenticity
            tssLast: Fast_TimestepSample = trajectoriespool._pool[last_item_idx]
            assert last_item_idx + 1 is tssLast._container_id
            assert obs_t is tssLast.obs_t
            assert act_t is tssLast.act_t
            assert obs_t_prime is tssLast.obs_t_prime
            assert rew_t is tssLast.rew_t
            assert done_t is tssLast.done_t
            obs_t = obs_t_prime
    
    for idx in range(capacity):
        tss: Fast_TimestepSample = trajectoriespool._pool[idx]
        assert tss._container_id == idx + 1
    
    assert trajectoriespool._idx == 0
    assert trajectoriespool.size == capacity
    assert trajectoriespool._pool_full() is True


def test_TrajectoriesPool_RENEW_SAMPLE(gym_continuous_pool_setup):
    (exp_spec, playground, poolmanager,
     timestepsampleOne,
     timestepsampleTwo,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    obs_t = initial_observation
    for idx in range(exp_spec['pool_capacity']):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, trajectoriespool)
        if idx != 19:
            obs_t = obs_t_prime
        elif idx == 19:
            # assess collected stuff authenticity
            tss20: Fast_TimestepSample = trajectoriespool._pool[19]
            assert 20 is tss20._container_id
            assert obs_t is tss20.obs_t
            assert act_t is tss20.act_t
            assert obs_t_prime is tss20.obs_t_prime
            assert rew_t is tss20.rew_t
            assert done_t is tss20.done_t
            obs_t = obs_t_prime
    
    assert trajectoriespool._idx == 0
    assert trajectoriespool.size == exp_spec['pool_capacity']
    
    # Do --> overfill capacity
    act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, trajectoriespool)
    
    assert trajectoriespool._idx == 1
    assert trajectoriespool.size == exp_spec['pool_capacity']
    
    assert trajectoriespool._pool
    
    for idx in range(exp_spec['pool_capacity']):
        tss: Fast_TimestepSample = trajectoriespool._pool[idx]
        assert tss._container_id == idx + 1


def test_PoolManager_COLLECT_20(gym_continuous_pool_setup):
    (exp_spec, playground, poolmanager,
     timestepsampleOne,
     timestepsampleTwo,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    obs_t = initial_observation
    for idx in range(20):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, poolmanager)
        if idx < 19:
            obs_t = obs_t_prime
        else:
            # assess collected stuff authenticity
            tss20: Fast_TimestepSample = poolmanager._trajectories_pool._pool[19]
            assert 20 is tss20._container_id
            assert obs_t is tss20.obs_t
            assert act_t is tss20.act_t
            assert obs_t_prime is tss20.obs_t_prime
            assert rew_t is tss20.rew_t
            assert done_t is tss20.done_t
            obs_t = obs_t_prime
    
    for idx in range(20):
        tss: Fast_TimestepSample = poolmanager._trajectories_pool._pool[idx]
        assert tss._container_id == idx + 1
    
    assert poolmanager.current_pool_size == 20
    assert poolmanager.timestep_collected_so_far() == 20


def test_PoolManager_COLLECT_TO_CAPACITY(gym_continuous_pool_setup):
    (exp_spec, playground, poolmanager,
     timestepsampleOne,
     timestepsampleTwo,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    obs_t = initial_observation
    capacity = exp_spec['pool_capacity']
    for idx in range(capacity):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, poolmanager)
        last_item_idx = capacity - 1
        if idx < last_item_idx:
            obs_t = obs_t_prime
        else:
            # assess collected stuff authenticity
            tssLast: Fast_TimestepSample = poolmanager._trajectories_pool._pool[last_item_idx]
            assert last_item_idx + 1 is tssLast._container_id
            assert obs_t is tssLast.obs_t
            assert act_t is tssLast.act_t
            assert obs_t_prime is tssLast.obs_t_prime
            assert rew_t is tssLast.rew_t
            assert done_t is tssLast.done_t
            obs_t = obs_t_prime
    
    for idx in range(capacity):
        tss: Fast_TimestepSample = poolmanager._trajectories_pool._pool[idx]
        assert tss._container_id == idx + 1
    
    assert poolmanager.current_pool_size == capacity
    assert poolmanager.timestep_collected_so_far() == capacity


def test_PoolManager_RENEW_SAMPLE(gym_continuous_pool_setup):
    (exp_spec, playground, poolmanager,
     timestepsampleOne,
     timestepsampleTwo,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    obs_1 = initial_observation
    act_1, obs_t_prim1, rew_1, done_1 = step_foward_and_collect(env, obs_1, poolmanager)
    
    obs_t = obs_1
    capacity = exp_spec['pool_capacity']
    for idx in range(1, capacity):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, poolmanager)
        obs_t = obs_t_prime
    
    obs_101 = obs_t
    assert poolmanager.current_pool_size == capacity
    
    # Do --> overfill capacity
    act_101, obs_t_prim101, rew_101, done_101 = step_foward_and_collect(env, obs_101, poolmanager)
    
    assert poolmanager.current_pool_size == capacity
    assert poolmanager.timestep_collected_so_far() == capacity + 1
    
    # assess collected stuff authenticity: timestep 101
    tss101: Fast_TimestepSample = poolmanager._trajectories_pool._pool[0]
    assert 1 is tss101._container_id
    assert obs_101 is tss101.obs_t
    assert act_101 is tss101.act_t
    assert obs_t_prim101 is tss101.obs_t_prime
    assert rew_101 is tss101.rew_t
    assert done_101 is tss101.done_t
    
    # assess collected stuff authenticity timestep 101 != timestep 1
    tss0: Fast_TimestepSample = poolmanager._trajectories_pool._pool[0]
    assert 1 is tss0._container_id
    assert obs_1 is not tss0.obs_t
    assert act_1 is not tss0.act_t
    assert obs_t_prim1 is not tss0.obs_t_prime
    assert rew_1 is not tss0.rew_t
    
    for idx in range(capacity):
        tss: Fast_TimestepSample = poolmanager._trajectories_pool._pool[idx]
        assert tss._container_id == idx + 1


def test_PoolManager_TRAJECTORY_ENDED(gym_continuous_pool_setup):
    (exp_spec, playground, poolmanager,
     timestepsampleOne,
     timestepsampleTwo,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    trj1 = 100
    obs_t = initial_observation
    for _ in range(trj1):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, poolmanager, dummy_rew=1.0)
        obs_t = obs_t_prime
    
    assert poolmanager.timestep_collected_so_far() == trj1
    assert poolmanager.trj_collected_so_far() == 0
    
    trajectory_return, trajectory_lenght = poolmanager.trajectory_ended()
    
    assert poolmanager.trj_collected_so_far() == 1
    assert trajectory_return == trj1
    assert trajectory_lenght == trj1
    
    trj2 = 20
    obs_t = env.reset()
    for _ in range(trj2):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, poolmanager, dummy_rew=1.0)
        obs_t = obs_t_prime
    
    assert poolmanager.timestep_collected_so_far() == trj1 + trj2
    assert poolmanager.trj_collected_so_far() == 1
    
    trajectory_return, trajectory_lenght = poolmanager.trajectory_ended()
    
    assert poolmanager.trj_collected_so_far() == 2
    assert poolmanager.timestep_collected_so_far() == trj1 + trj2
    assert trajectory_return == trj2
    assert trajectory_lenght == trj2


@pytest.mark.skip(reason="Assess later")  # (Priority) todo:fixme!! --> intanciate more timestepsample* sample:
def test_PoolManager_PRODUCE_MINIBATCH(gym_continuous_pool_setup):
    (_, _, _, _, _, _, _, env, initial_observation) = gym_continuous_pool_setup
    
    POOL_CAPACITY_4 = 4
    exp_spec = ExperimentSpec(batch_size_in_ts=2, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2),
                              environment_name='LunarLanderContinuous-v2')
    exp_spec.set_experiment_spec({'pool_capacity': POOL_CAPACITY_4})
    playground = GymPlayground(exp_spec.prefered_environment)
    poolmanager = Fast_PoolManager(exp_spec, playground)
    
    timestepsample1 = Fast_TimestepSample(container_id=1, playground=playground)
    timestepsample2 = Fast_TimestepSample(container_id=2, playground=playground)
    timestepsample3 = Fast_TimestepSample(container_id=3, playground=playground)
    timestepsample4 = Fast_TimestepSample(container_id=4, playground=playground)
    tss_collection = [timestepsample1, timestepsample2, timestepsample3, timestepsample4]
    
    obs_t = initial_observation
    tss: Fast_TimestepSample
    for tss in tss_collection:
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, poolmanager)
        tss.replace(obs_t=obs_t, act_t=act_t, obs_t_prime=obs_t_prime, rew_t=rew_t, done_t=done_t)
        obs_t = obs_t_prime
    
    assert poolmanager.timestep_collected_so_far() == POOL_CAPACITY_4
    
    minibatch_list1 = poolmanager._trajectories_pool.sample_from_pool_as_list()
    minibatch_list2 = poolmanager._trajectories_pool.sample_from_pool_as_list()
    
    print(minibatch_list1)
    print(minibatch_list2)
    
    assert minibatch_list1 != minibatch_list2
    
    popu = [idx for idx in range(10)]
    popu_sample1 = random.sample(popu, 3)
    popu_sample2 = random.sample(popu, 3)
    
    assert popu_sample1 != popu_sample2
    
    minibatch1: Fast_SampleBatch = poolmanager.sample_from_pool()
    minibatch2: Fast_SampleBatch = poolmanager.sample_from_pool()
    
    print(minibatch1)
    print(minibatch2)
    
    flag1 = [tss in minibatch1 for tss in tss_collection]
    flag2 = [tss in minibatch2 for tss in tss_collection]
    
    assert sum(flag1) == 2
    assert sum(flag2) == 2
    
    assert minibatch1 != minibatch2
    
    # (Priority) todo:unit-test --> continue implementation of test:
    
    # # assess collected stuff authenticity
    # tss20: Fast_TimestepSample = poolmanager._trajectories_pool._pool[19]
    # assert 20 is tss20._container_id
    # assert act_t is tss20.act_t
    # assert obs_t_prime is tss20.obs_t_prime
    # assert rew_t is tss20.rew_t
    # assert done_t is tss20.done_t
