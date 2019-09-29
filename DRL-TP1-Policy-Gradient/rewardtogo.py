# coding=utf-8
import numpy as np

from buildingbloc import ExperimentSpec


def reward_to_go(rewards: list) -> list:
    assert isinstance(rewards, list)
    np_backward_rewards = np.array(rewards[::-1])
    reward_to_go = np.cumsum(np_backward_rewards)
    return list(reward_to_go[::-1])


def reward_to_go_np(rewards: np.ndarray) -> np.ndarray:
    assert isinstance(rewards, np.ndarray)
    np_backward_rewards = np.flip(rewards)
    reward_to_go = np.cumsum(np_backward_rewards)
    return np.flip(reward_to_go)


def discounted_reward_to_go(rewards: list, experiment_spec: ExperimentSpec) -> list:
    """
    Compute the discounted reward to go iteratively

    (Priority) todo:refactor --> refactor using a gamma mask and matrix product & sum, instead of loop
    (nice to have) todo:refactor --> fct signature: pass gamma (the discount factor) explicitely

    :param rewards:
    :type rewards:
    :param experiment_spec:
    :type experiment_spec:
    :return:
    :rtype:
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
    gamma = experiment_spec.discout_factor
    assert (0 <= gamma) and (gamma <= 1)
    assert rewards.ndim == 1, "Current implementation only support array of rank 1"
    assert isinstance(rewards, np.ndarray)

    np_backward_rewards = np.flip(rewards)
    discounted_reward_to_go = np.zeros_like(rewards)

    # todo --> Since flip return a view, test if iterate on a pre flip ndarray and than post flip before return would be cleaner

    # refactor --> using a gamma mask and matrix product & sum, instead of loop
    for r in range(len(rewards)):
        exp = 0
        for i in range(r, len(rewards)):
            discounted_reward_to_go[i] += gamma**exp * np_backward_rewards[r]
            exp += 1

    return np.flip(discounted_reward_to_go)