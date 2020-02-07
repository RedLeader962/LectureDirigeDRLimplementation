# coding=utf-8
from typing import Union, Any
from copy import deepcopy

import pytest
from gym.wrappers import TimeLimit

import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import numpy as np

from blocAndTools.container.trajectories_pool import PoolManager, SampleBatch, TrajectoriesPool
from blocAndTools.container.FAST_trajectories_pool import Fast_PoolManager
from blocAndTools.container.FAST_trajectories_pool import Fast_SampleBatch as Fast_SampleBatch
from blocAndTools.container.FAST_trajectories_pool import Fast_TrajectoriesPool as Fast_TrajectoriesPool
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground

from blocAndTools.container.pool_utility.pool_testingUtility_MOCK_experience import (
    mock_simulate_a_SAC_trj_run,
    mock_full_pool_to_minimum,
    ExperienceMocker,
    )

from blocAndTools.container.pool_utility.pool_testingUtility_general import (
    instantiate_top_component,
    print_final_pool_sideEffect,
    )

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf_cv1 = tf.compat.v1  # shortcut


# note: exp_spec key specific to SAC
#   |   'pool_capacity'


@pytest.fixture(params=['ORIGINAL', 'FAST'], scope="function")
def AUTO_PoolManager_setup(request) -> (ExperimentSpec, GymPlayground, Union[PoolManager, Fast_PoolManager],
                                        Union[SampleBatch, Fast_SampleBatch],
                                        Union[TrajectoriesPool, Fast_TrajectoriesPool],
                                        Union[TimeLimit, Any], np.ndarray):
    """
    :return: (exp_spec, playground, poolmanager, samplebatch, trajectoriespool, env, initial_observation)
    """
    
    if request.param == 'ORIGINAL':
        exp_spec, playground = instantiate_top_component(manager=PoolManager)
        poolmanager = PoolManager(exp_spec, playground)
        samplebatch = SampleBatch(batch_size=exp_spec.batch_size_in_ts, playground=playground)
        trajectoriespool = TrajectoriesPool(capacity=exp_spec['pool_capacity'], batch_size=exp_spec.batch_size_in_ts,
                                            playground=playground)
    elif request.param == 'FAST':
        exp_spec, playground = instantiate_top_component(manager=Fast_PoolManager)
        poolmanager = Fast_PoolManager(exp_spec, playground)
        samplebatch = Fast_SampleBatch(batch_size=exp_spec.batch_size_in_ts, playground=playground)
        trajectoriespool = Fast_TrajectoriesPool(capacity=exp_spec['pool_capacity'],
                                                 batch_size=exp_spec.batch_size_in_ts,
                                                 playground=playground)
    else:
        raise ValueError("Wrong request type: ", str(request.param))

    env = playground.env
    experience_mocker = ExperienceMocker(env)
    
    yield (exp_spec, playground,
           poolmanager,
           samplebatch,
           trajectoriespool,
           env, experience_mocker)
    
    del poolmanager
    del samplebatch
    del trajectoriespool


# --- PoolManager benchmark test MOCK EXPERIENCE -----------------------------------------------------------------------

# @pytest.mark.skip(reason=">>> Dont benchmark")
def test_BENCH_PoolManager_COLLECT(benchmark, AUTO_PoolManager_setup):
    (exp_spec, playground,
     initinal_poolmanager,
     initinal_samplebatch,
     initinal_trajectoriespool,
     env, experience_mocker) = AUTO_PoolManager_setup
    
    def benchmark_full_pool_to_minimum():
        poolmanager = deepcopy(initinal_poolmanager)
        b_poolmanager = mock_full_pool_to_minimum(exp_spec=exp_spec, poolmanager=poolmanager,
                                                  experience_mocker=experience_mocker)
        return b_poolmanager
    
    b_poolmanager = benchmark(benchmark_full_pool_to_minimum)
    
    assert initinal_poolmanager.current_pool_size == 0
    
    print(b_poolmanager)
    print_final_pool_sideEffect(exp_spec, b_poolmanager)


# @pytest.mark.skip(reason=">>> Dont benchmark")
def test_BENCH_PoolManager_GRADIENT_STEP(benchmark, AUTO_PoolManager_setup):
    (exp_spec, playground,
     initinal_poolmanager,
     initinal_samplebatch,
     initinal_trajectoriespool,
     env, experience_mocker) = AUTO_PoolManager_setup
    
    initinal_poolmanager = mock_full_pool_to_minimum(exp_spec=exp_spec,
                                                     poolmanager=initinal_poolmanager,
                                                     experience_mocker=experience_mocker)
    
    def benchmark_simulate_a_SAC_trj_run():
        poolmanager = deepcopy(initinal_poolmanager)
        b_poolmanager = mock_simulate_a_SAC_trj_run(exp_spec=exp_spec, poolmanager=poolmanager,
                                                    experience_mocker=experience_mocker)
        return b_poolmanager
    
    b_poolmanager = benchmark(benchmark_simulate_a_SAC_trj_run)
    
    assert initinal_poolmanager.current_pool_size == exp_spec['min_pool_size']
    
    print(b_poolmanager)
    print_final_pool_sideEffect(exp_spec, b_poolmanager)
