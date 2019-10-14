# coding=utf-8
import numpy as np

from blocAndTools.buildingbloc import ExperimentSpec


def reward_to_go(rewards: list) -> list:
    """
    Compute the reward to go (Python list version)
    :param rewards: a array of rewards
    :type rewards: list
    :return: the computed reward to go
    :rtype: list
    """
    assert isinstance(rewards, list)
    np_backward_rewards = np.array(rewards[::-1])
    rtg = np.cumsum(np_backward_rewards)
    return list(rtg[::-1])


def reward_to_go_np(rewards: np.ndarray) -> np.ndarray:
    """
    Compute the reward to go (Numpy ndarray version)
    :param rewards: a array of rewards
    :type rewards: np.ndarray
    :return: the computed reward to go
    :rtype: np.ndarray
    """
    assert isinstance(rewards, np.ndarray)
    np_backward_rewards = np.flip(rewards)
    rtg = np.cumsum(np_backward_rewards)
    return np.flip(rtg)


def discounted_reward_to_go(rewards: list, experiment_spec: ExperimentSpec) -> list:
    """
    Compute the discounted reward to go iteratively

    (Priority) todo:refactor --> refactor using a gamma mask and matrix product & sum, instead of a loop
    (nice to have) todo:refactor --> fct signature: pass gamma (the discount factor) explicitely
    """
    gamma = experiment_spec.discout_factor
    assert (0 <= gamma) and (gamma <= 1)
    assert isinstance(rewards, list)

    backward_rewards = rewards[::-1]
    discounted_reward_to_go = np.zeros_like(rewards)

    for r in range(len(rewards)):
        exp = 0
        for i in range(r, len(rewards)):
            discounted_reward_to_go[i] += gamma**exp * backward_rewards[r]
            exp += 1

    return list(discounted_reward_to_go[::-1])


def discounted_reward_to_go_np(rewards: np.ndarray, experiment_spec: ExperimentSpec) -> np.ndarray:
    """
    Compute the discounted reward to go iteratively (Numpy ndarray version)

    (Priority) todo:refactor --> refactor using a gamma mask and matrix product & sum, instead of a loop
    (nice to have) todo:refactor --> fct signature: pass gamma (the discount factor) explicitely
    """
    gamma = experiment_spec.discout_factor
    assert (0 <= gamma) and (gamma <= 1)
    assert rewards.ndim == 1, "Current implementation only support array of rank 1"
    assert isinstance(rewards, np.ndarray)

    np_backward_rewards = np.flip(rewards)
    discounted_rtg = np.zeros_like(rewards)

    # (Ice-Boxed) todo:assessment --> Since flip return a view, test if iterating on a pre-fliped ndarray
    #                                                           and than post flip before return would be cleaner:
    for r in range(len(rewards)):
        exp = 0
        for i in range(r, len(rewards)):
            discounted_rtg[i] += gamma**exp * np_backward_rewards[r]
            exp += 1

    return np.flip(discounted_rtg)
