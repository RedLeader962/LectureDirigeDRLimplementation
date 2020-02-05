# coding=utf-8
from typing import Union, Any, Type

import pytest
from gym.wrappers import TimeLimit

import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import numpy as np

from SoftActorCritic import SoftActorCriticAgent

from blocAndTools.container.trajectories_pool import PoolManager, TimestepSample, SampleBatch, TrajectoriesPool
from blocAndTools.container.FAST_trajectories_pool import Fast_PoolManager
from blocAndTools.container.FAST_trajectories_pool import Fast_SampleBatch as Fast_SampleBatch
from blocAndTools.container.FAST_trajectories_pool import Fast_TrajectoriesPool as Fast_TrajectoriesPool
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground

from .pool_testingUtility import step_foward_and_collect

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf_cv1 = tf.compat.v1  # shortcut

# note: exp_spec key specific to SAC
#   |   'pool_capacity'


sac_BENCHMARK_hparam = {
    'rerun_tag':                      'TrjPool-Benchmark',
    'paramameter_set_name':           'SAC',
    'algo_name':                      'Soft Actor Critic',
    'AgentType':                      SoftActorCriticAgent,
    'prefered_environment':           'LunarLanderContinuous-v2',
    'max_trj_steps':                  1000,
    
    'expected_reward_goal':           90,  # Note: trigger model save on reach
    'max_epoch':                      10,
    'timestep_per_epoch':             1000,
    
    'reward_scaling':                 5.0,
    
    'discout_factor':                 0.99,  # SAC paper: 0.99
    'learning_rate':                  0.003,  # SAC paper: 30e-4
    'critic_learning_rate':           0.003,  # SAC paper: 30e-4
    'max_gradient_step_expected':     600,
    'actor_lr_decay_rate':            0.01,  # Note: set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           0.01,  # Note: set to 1 to swith OFF scheduler
    
    'target_smoothing_coefficient':   0.005,  # SAC paper: EXPONENTIAL MOVING AVERAGE ~ 0.005, 1 <==> HARD TARGET update
    'target_update_interval':         1,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    'gradient_step_interval':         1,
    
    'alpha':                          1,  # HW5: we recover a standard max expected return objective as alpha --> 0
    
    'max_eval_trj':                   10,  #SpiningUp: 10
    
    'pool_capacity':                  5000,  # SAC paper: 1e6
    'min_pool_size':                  500,
    'batch_size_in_ts':               200,  # SAC paper:256, SpinningUp:100
    
    'theta_nn_h_layer_topo':          (2,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'theta_hidden_layers_activation': tf.nn.relu,
    'theta_output_layers_activation': None,
    'phi_nn_h_layer_topo':            (2,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'phi_hidden_layers_activation':   tf.nn.relu,
    'phi_output_layers_activation':   None,
    'psi_nn_h_layer_topo':            (2,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'psi_hidden_layers_activation':   tf.nn.relu,
    'psi_output_layers_activation':   None,
    
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'random_seed':                    0,  # Note: 0 --> turned OFF (default)
    'isTestRun':                      True,
    'show_plot':                      False,
    'note':                           'My note ...',
    }


#
# @pytest.fixture(scope="function")
# def gym_continuous_ORIGINAL_pool_setup() -> (ExperimentSpec, GymPlayground, PoolManager,
#                                              SampleBatch, TrajectoriesPool,
#                                              Union[TimeLimit, Any], np.ndarray):
#     """
#     :return: (exp_spec, playground, poolmanager, samplebatch, trajectoriespool, env, initial_observation)
#     """
#     exp_spec = ExperimentSpec()
#     exp_spec.set_experiment_spec(sac_BENCHMARK_hparam)
#     playground = GymPlayground(exp_spec.prefered_environment)
#
#     poolmanager = PoolManager(exp_spec, playground)
#     samplebatch = SampleBatch(batch_size=exp_spec.batch_size_in_ts, playground=playground)
#     trajectoriespool = TrajectoriesPool(capacity=exp_spec['pool_capacity'], batch_size=exp_spec.batch_size_in_ts,
#                                         playground=playground)
#
#     env = playground.env
#     initial_observation = env.reset()
#
#     yield (exp_spec, playground,
#            poolmanager,
#            samplebatch,
#            trajectoriespool,
#            env, initial_observation)
#
#
# @pytest.fixture(scope="function")
# def gym_continuous_FAST_pool_setup() -> (ExperimentSpec, GymPlayground, Fast_PoolManager,
#                                          Fast_SampleBatch, Fast_TrajectoriesPool,
#                                          Union[TimeLimit, Any], np.ndarray):
#     """
#     :return: (exp_spec, playground, fast_poolmanager, fast_samplebatch, fast_trajectoriespool, env,
#     initial_observation)
#     """
#     exp_spec = ExperimentSpec()
#     exp_spec.set_experiment_spec(sac_BENCHMARK_hparam)
#     playground = GymPlayground(exp_spec.prefered_environment)
#
#     fast_poolmanager = Fast_PoolManager(exp_spec, playground)
#     fast_samplebatch = Fast_SampleBatch(batch_size=exp_spec.batch_size_in_ts, playground=playground)
#     fast_trajectoriespool = Fast_TrajectoriesPool(capacity=exp_spec['pool_capacity'],
#                                                   batch_size=exp_spec.batch_size_in_ts,
#                                                   playground=playground)
#
#     env = playground.env
#     initial_observation = env.reset()
#
#     yield (exp_spec, playground,
#            fast_poolmanager,
#            fast_samplebatch,
#            fast_trajectoriespool,
#            env, initial_observation)

@pytest.fixture(params=['ORIGINAL', 'FAST'], scope="function")
def gym_continuous_pool_setup(request) -> (ExperimentSpec, GymPlayground, Union[PoolManager, Fast_PoolManager],
                                           Union[SampleBatch, Fast_SampleBatch],
                                           Union[TrajectoriesPool, Fast_TrajectoriesPool],
                                           Union[TimeLimit, Any], np.ndarray):
    """
    :return: (exp_spec, playground, poolmanager, samplebatch, trajectoriespool, env, initial_observation)
    """
    
    if request == 'ORIGINAL':
        exp_spec, playground = instantiate_top_component(manager=PoolManager)
        poolmanager = PoolManager(exp_spec, playground)
        samplebatch = SampleBatch(batch_size=exp_spec.batch_size_in_ts, playground=playground)
        trajectoriespool = TrajectoriesPool(capacity=exp_spec['pool_capacity'], batch_size=exp_spec.batch_size_in_ts,
                                            playground=playground)
    else:
        exp_spec, playground = instantiate_top_component(manager=Fast_PoolManager)
        poolmanager = Fast_PoolManager(exp_spec, playground)
        samplebatch = Fast_SampleBatch(batch_size=exp_spec.batch_size_in_ts, playground=playground)
        trajectoriespool = Fast_TrajectoriesPool(capacity=exp_spec['pool_capacity'],
                                                 batch_size=exp_spec.batch_size_in_ts,
                                                 playground=playground)
    
    env = playground.env
    initial_observation = env.reset()
    
    yield (exp_spec, playground,
           poolmanager,
           samplebatch,
           trajectoriespool,
           env, initial_observation)


def instantiate_top_component(manager: Type[Union[PoolManager, Fast_PoolManager]]) -> (ExperimentSpec, GymPlayground):
    exp_spec = ExperimentSpec()
    pool_fixture_hparam = dict(sac_BENCHMARK_hparam)
    pool_fixture_hparam.update({'comment': manager.__name__})
    exp_spec.set_experiment_spec(pool_fixture_hparam)
    playground = GymPlayground(exp_spec.prefered_environment)
    return exp_spec, playground


# --- PoolManager benchmark test ---------------------------------------------------------------------------------------

# @pytest.mark.parametrize(argnames="pooL_fixture",
#                          argvalues=[gym_continuous_ORIGINAL_pool_setup, gym_continuous_FAST_pool_setup])
def test_PoolManager_COLLECT(gym_continuous_pool_setup):
    (exp_spec, playground,
     poolmanager,
     samplebatch,
     trajectoriespool,
     env, initial_observation) = gym_continuous_pool_setup
    
    trj1 = 100
    obs_t = initial_observation
    for _ in range(trj1):
        act_t, obs_t_prime, rew_t, done_t = step_foward_and_collect(env, obs_t, poolmanager, dummy_rew=1.0)
        obs_t = obs_t_prime
    
    trajectory_return, trajectory_lenght = poolmanager.trajectory_ended()
