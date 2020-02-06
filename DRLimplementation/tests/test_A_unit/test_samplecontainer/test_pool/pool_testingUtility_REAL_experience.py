# coding=utf-8
from typing import Union, Any, Tuple
import copy

import numpy as np
from gym.wrappers import TimeLimit

from blocAndTools import PoolManager, Fast_PoolManager, ExperimentSpec
from blocAndTools.buildingbloc import GymPlayground
from blocAndTools.container.trajectories_pool import TrajectoriesPool


def real_step_foward_and_collect(env: Union[TimeLimit, Any], obs_t: np.ndarray,
                                 pool: Union[TrajectoriesPool, PoolManager, Fast_PoolManager], dummy_rew: float = None):
    """
    Testing utility function.
        Execute:
            1) select a action
            2) execute in the environment
            3) collect (obs_t, act_t, obs_t_prime, rew_t, done_t) in container
    :param env: gym environment
    :param obs_t: the current observation
    :param pool: the container to pass the collected stuff
    :param dummy_rew: (optional) overwrite collected rew_t with dummy_rew
    :return: (act_t, obs_t_prime, rew_t, done_t)
    """
    act_t = env.action_space.sample()
    obs_t_prime, rew_t, done_t, _ = env.step(act_t)
    
    if dummy_rew is not None:
        rew_t = dummy_rew
    
    pool.collect_OAnORD(obs_t,
                        act_t,
                        obs_t_prime,
                        rew_t,
                        done_t)
    return act_t, obs_t_prime, rew_t, done_t


def collect_many_step(for_nb_step: int, poolmanager: Union[PoolManager, Fast_PoolManager], obs_t: np.ndarray,
                      env: Union[TimeLimit, Any], dummy_rew: float = None, reset_at_end: bool = False
                      ) -> (Union[PoolManager, Fast_PoolManager], np.ndarray):
    for _ in range(for_nb_step):
        act_t, obs_t_prime, rew_t, done_t = real_step_foward_and_collect(env, obs_t, poolmanager, dummy_rew)
        obs_t = obs_t_prime
    
    if reset_at_end:
        poolmanager._reset()
    return poolmanager, obs_t


def collect_and_sample_for_many_step(for_nb_step: int, poolmanager: Union[PoolManager, Fast_PoolManager],
                                     obs_t: np.ndarray, env: Union[TimeLimit, Any], dummy_rew: float = None
                                     ) -> (Union[PoolManager, Fast_PoolManager], np.ndarray):
    for _ in range(for_nb_step):
        act_t, obs_t_prime, rew_t, done_t = real_step_foward_and_collect(env, obs_t, poolmanager, dummy_rew)
        obs_t = obs_t_prime
        poolmanager.sample_from_pool()
    return poolmanager, obs_t


def full_pool_to_minimum(exp_spec: ExperimentSpec, poolmanager: Union[PoolManager, Fast_PoolManager],
                         obs_t: np.ndarray, env: Union[TimeLimit, Any], dummy_rew: float = None
                         ) -> (Union[PoolManager, Fast_PoolManager], np.ndarray):
    """ Collect experience until pool is full enough to sample
    
    :param exp_spec: the experiment specification
    :param poolmanager: any poolmanager
    :param obs_t: the last observation
    :param env: the Gym environment
    :param dummy_rew: (optional) overwrite collected rew_t with dummy_rew
    :return: the pollmanager and the last observation
    """
    
    assert poolmanager.current_pool_size == 0, (">>> 'poolmanager.current_pool_size' SHOULD BE EMPTY "
                                                "but as {}".format(poolmanager.current_pool_size))
    
    trajectories = int(exp_spec['min_pool_size'] / exp_spec['max_trj_steps'])
    for _ in range(trajectories):
        poolmanager, obs_t = collect_many_step(exp_spec['max_trj_steps'], poolmanager, obs_t, env, dummy_rew,
                                               reset_at_end=False)
        poolmanager.trajectory_ended()
    
    assert poolmanager.current_pool_size == exp_spec['min_pool_size'], (
        ">>> 'poolmanager.current_pool_size' SHOULD BE AT MIN SIZE but as {}".format(poolmanager.current_pool_size))
    return poolmanager, obs_t


def simulate_a_SAC_trj_run(exp_spec: ExperimentSpec, poolmanager: Union[PoolManager, Fast_PoolManager],
                           obs_t: np.ndarray, env: Union[TimeLimit, Any], dummy_rew: float = None
                           ) -> (Union[PoolManager, Fast_PoolManager], np.ndarray):
    """ Simulate the way a standard Soft Actor-Critic algorithm uses the experience pool.
    
    Repeate until epoch end:
        1) Collect step
        2) sample experience
        3) take gradient step (not simulated)
    
    :param exp_spec: the experiment specification
    :param poolmanager: any poolmanager
    :param obs_t: the last observation
    :param env: the Gym environment
    :param dummy_rew: (optional) overwrite collected rew_t with dummy_rew
    :return: the pollmanager and the last observation
    """
    assert poolmanager.current_pool_size == exp_spec['min_pool_size'], (
        ">>> 'poolmanager.current_pool_size' SHOULD BE AT MIN SIZE but as {}".format(poolmanager.current_pool_size))
    
    trajectories = int(exp_spec['timestep_per_epoch'] / exp_spec['max_trj_steps'])
    for _ in range(trajectories):
        poolmanager, obs_t = collect_and_sample_for_many_step(exp_spec['max_trj_steps'], poolmanager, obs_t, env,
                                                              dummy_rew)
        poolmanager.trajectory_ended()
    
    assert (poolmanager.timestep_collected_so_far() - exp_spec['min_pool_size']) == exp_spec['timestep_per_epoch'], (
        ">>> 'poolmanager.timestep_collected_so_far' SHOULD BE GREATER than exp_spec['timestep_per_epoch'] "
        "but as {}".format(poolmanager.timestep_collected_so_far))
    return poolmanager, obs_t
