# coding=utf-8
from typing import List, Tuple

import numpy as np

# (nice to have) todo:implement --> Advantage computation with discount factor:

def computhe_the_Advantage(rewards: List, v_estimates: List) -> np.ndarray:
    """Compute the Advantage for the full trajectory in one shot using element wise operation

    Note: on computing the Advantage
      |
      |   Their is many way to implement Advantage computation:
      |       - directly in the computation graph (eg the single network Actor-Critic),
      |       - during trajectory (eg online Actor-Critic),
      |       - or post trajectory (eg batch Actor-Critic)
      |
      |   Which way to chose depend on your Actor-Critic algorithm architecture design
      |
      |   Here we use the post trajectory computation approach:
      |     How: compute the Advantage for the trajectory in one shot using element wise operation and array slicing
      |     Requirement: V estimate collected for every timestep
      |     PRO: give us the ability to implement Actor-Critic variant with discount fator, n_step return or GAE

    :return: the computed Advantage
    :rtype: np.ndarray
    """
    rew_t = np.array(rewards)
    V_t, V_tPrime = get_t_and_tPrime_array_view_for_element_wise_op(v_estimates)

    advantage = rew_t + V_tPrime - V_t
    return advantage


def compute_TD_target(rewards: list, v_estimates: list) -> np.ndarray:
    """Compute the Temporal Difference target for the full trajectory in one shot using element wise operation"""
    rew_t = np.array(rewards)
    _, V_tPrime = get_t_and_tPrime_array_view_for_element_wise_op(v_estimates)
    TD_target = rew_t + V_tPrime
    return TD_target


def get_t_and_tPrime_array_view_for_element_wise_op(trajectory_values: List) -> Tuple[np.ndarray, np.ndarray]:
    """Utility for trajectory array view alignement: timestep t vs t+1"""

    # Note: Trick requirement
    #   |   - Extend the array by one blank space
    #   |   - Than performing comparaison element wise:
    #   |       [:-1] is the collected value at timestep t
    #   |       [1:] is the collected value at timestep t+1
    local_trj_v = trajectory_values.copy()
    local_trj_v.append(0)
    view = np.array(local_trj_v)
    view_t = view[:-1]
    view_tPrime = view[1:]
    return view_t, view_tPrime


