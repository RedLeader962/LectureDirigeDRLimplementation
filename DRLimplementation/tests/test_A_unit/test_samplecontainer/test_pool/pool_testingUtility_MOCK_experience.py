# coding=utf-8
from typing import Union, Any, Tuple
import copy

import numpy as np
from gym.wrappers import TimeLimit

from blocAndTools import PoolManager, Fast_PoolManager, ExperimentSpec
from blocAndTools.buildingbloc import GymPlayground
from blocAndTools.container.trajectories_pool import TrajectoriesPool


class ExperienceMocker(object):
    __slots__ = ['obs_t', 'act_t', 'obs_t', 'rew_t', 'done_t']
    
    def __init__(self, env: Union[TimeLimit, Any]) -> None:
        self.obs_t: np.ndarray = np.ones_like(env.reset())
        self.act_t: np.ndarray = np.ones_like(env.action_space.sample())
        self.obs_t_prime: np.ndarray = self.obs_t.copy()
        self.rew_t: float = 1.0
        self.done_t: bool = False
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        return self.obs_t.copy(), self.act_t.copy(), self.obs_t_prime.copy(), copy(self.rew_t), copy(self.done_t)


def mock_step_foward_and_collect(pool_manager: Union[TrajectoriesPool, PoolManager, Fast_PoolManager],
                                 experience_mocker: ExperienceMocker) -> None:
    """
    Testing utility function.
        Execute:
            1) select a action
            2) execute in the environment
            3) collect (obs_t, act_t, obs_t_prime, rew_t, done_t) in container
    :param pool_manager: the container to pass the collected stuff
    :param experience_mocker: a ExperienceMocker object
    :return: None
    """
    
    pool_manager.collect_OAnORD(experience_mocker.generate())
    
    return None


def mock_collect_many_step(for_nb_step: int, poolmanager: Union[PoolManager, Fast_PoolManager],
                           experience_mocker: ExperienceMocker,
                           reset_at_end: bool = False
                           ) -> Union[PoolManager, Fast_PoolManager]:
    for _ in range(for_nb_step):
        mock_step_foward_and_collect(poolmanager, experience_mocker)
    
    if reset_at_end:
        poolmanager._reset()
    return poolmanager


def mock_collect_and_sample_for_many_step(for_nb_step: int, poolmanager: Union[PoolManager, Fast_PoolManager],
                                          experience_mocker: ExperienceMocker,
                                          ) -> Union[PoolManager, Fast_PoolManager]:
    for _ in range(for_nb_step):
        mock_step_foward_and_collect(poolmanager, experience_mocker)
        poolmanager.sample_from_pool()
    return poolmanager


def mock_full_pool_to_minimum(exp_spec: ExperimentSpec, poolmanager: Union[PoolManager, Fast_PoolManager],
                              experience_mocker: ExperienceMocker,
                              ) -> Union[PoolManager, Fast_PoolManager]:
    """ Collect experience until pool is full enough to sample
    
    :param exp_spec: the experiment specification
    :param poolmanager: any poolmanager
    :param experience_mocker: a ExperienceMocker object
    :return: the pollmanager fulled at minimum with mocked experience
    """
    
    assert poolmanager.current_pool_size == 0, (">>> 'poolmanager.current_pool_size' SHOULD BE EMPTY "
                                                "but as {}".format(poolmanager.current_pool_size))
    
    trajectories = int(exp_spec['min_pool_size'] / exp_spec['max_trj_steps'])
    for _ in range(trajectories):
        poolmanager, obs_t = mock_collect_many_step(exp_spec['max_trj_steps'], poolmanager,
                                                    experience_mocker,
                                                    reset_at_end=False)
        poolmanager.trajectory_ended()
    
    assert poolmanager.current_pool_size == exp_spec['min_pool_size'], (
        ">>> 'poolmanager.current_pool_size' SHOULD BE AT MIN SIZE but as {}".format(poolmanager.current_pool_size))
    return poolmanager


def simulate_a_SAC_trj_run(exp_spec: ExperimentSpec, poolmanager: Union[PoolManager, Fast_PoolManager],
                           experience_mocker: ExperienceMocker,
                           ) -> Union[PoolManager, Fast_PoolManager]:
    """ Simulate the way a standard Soft Actor-Critic algorithm uses the experience pool.
    
    Repeate until epoch end:
        1) Collect step
        2) sample experience
        3) take gradient step (not simulated)
    
    :param exp_spec: the experiment specification
    :param poolmanager: any poolmanager
    :param experience_mocker: a ExperienceMocker object
    :return: the pollmanager fulled with the required amount of mocked experience
    """
    assert poolmanager.current_pool_size == exp_spec['min_pool_size'], (
        ">>> 'poolmanager.current_pool_size' SHOULD BE AT MIN SIZE but as {}".format(poolmanager.current_pool_size))
    
    trajectories = int(exp_spec['timestep_per_epoch'] / exp_spec['max_trj_steps'])
    for _ in range(trajectories):
        poolmanager, obs_t = mock_collect_and_sample_for_many_step(exp_spec['max_trj_steps'], poolmanager,
                                                                   experience_mocker
                                                                   )
        poolmanager.trajectory_ended()
    
    assert (poolmanager.timestep_collected_so_far() - exp_spec['min_pool_size']) == exp_spec['timestep_per_epoch'], (
        ">>> 'poolmanager.timestep_collected_so_far' SHOULD BE GREATER than exp_spec['timestep_per_epoch'] "
        "but as {}".format(poolmanager.timestep_collected_so_far))
    return poolmanager
