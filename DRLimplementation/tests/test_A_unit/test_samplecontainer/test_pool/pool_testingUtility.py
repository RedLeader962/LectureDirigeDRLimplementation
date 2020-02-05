# coding=utf-8
from typing import Union, Any

import numpy as np
from gym.wrappers import TimeLimit

from blocAndTools import PoolManager
from blocAndTools.container.trajectories_pool import TrajectoriesPool


def step_foward_and_collect(env: Union[TimeLimit, Any], obs_t: np.ndarray,
                            trajectoriespool: Union[TrajectoriesPool, PoolManager],
                            dummy_rew: float = None):
    """
    Testing utility function.
        Execute:
            1) select a action
            2) execute in the environment
            3) collect (obs_t, act_t, obs_t_prime, rew_t, done_t) in container
    :param env: gym environment
    :param obs_t: the current observation
    :param trajectoriespool: the container to pass the collected stuff
    :param dummy_rew: (optional) overwrite collected rew_t with dummy_rew
    :return: (act_t, obs_t_prime, rew_t, done_t)
    """
    act_t = env.action_space.sample()
    obs_t_prime, rew_t, done_t, _ = env.step(act_t)
    
    if dummy_rew is not None:
        rew_t = dummy_rew
    
    trajectoriespool.collect_OAnORD(obs_t,
                                    act_t,
                                    obs_t_prime,
                                    rew_t,
                                    done_t)
    return act_t, obs_t_prime, rew_t, done_t
