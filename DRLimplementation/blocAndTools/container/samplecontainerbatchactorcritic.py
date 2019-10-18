# coding=utf-8
from typing import List

import numpy as np

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground

from blocAndTools.container.samplecontainer import TrajectoryContainer, TrajectoryCollector
from blocAndTools.container.samplecontainer import UniformeBatchContainer


class TrajectoryContainerBatchActorCritic(TrajectoryContainer):
    def __init__(self, observations: list, actions: list, rewards: list, Q_values: list, trajectory_return: list,
                 trajectory_id, V_estimates: list, Advantages: list) -> None:

        super().__init__(observations, actions, rewards, Q_values, trajectory_return, trajectory_id)

        self.V_estimates = V_estimates
        self.Advantages = Advantages

    def cut(self, max_lenght):
        super().cut(max_lenght)

        self.V_estimates = self.V_estimates[:max_lenght]
        self.Advantages = self.Advantages[:max_lenght]

    def unpack(self) -> (list, list, list, list, float, int, list, list):
        tc = super().unpack()

        return (*tc, self.V_estimates, self.Advantages)

    def __repr__(self):
        myRep = super().__repr__()
        myRep += ".V_estimates=\n{}\n\n".format(self.V_estimates)
        myRep += ".Advantages=\n{}\n\n".format(self.Advantages)
        return myRep


class TrajectoryCollectorBatchActorCritic(TrajectoryCollector):
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground, discounted: bool = True):
        self._TD_target = []
        self._Advantages = []
        super().__init__(experiment_spec, playground, discounted)

    def collect(self, observation: np.ndarray, action, reward: float, V_estimate: float = None) -> None:
        """ Collect observation, action, reward and V estimate for one timestep

        :type observation: np.ndarray
        :type action: int or float
        :type reward: float
        :type V_estimate: float
        """
        self._V_estimates.append(V_estimate)
        super().collect(observation, action, reward)

    def trajectory_ended(self) -> float:
        """ Must be called at each trajectory end

        Compute:
            1. the trajectory lenght base on collected samples
            2. the Q-values
            3. the trajectory return
            4. the Advantages


        :return: the trajectory return
        :rtype: float
        """

        advantage = self.computage_Advantage()

        self._Advantages = advantage

        return super().trajectory_ended()

    def pop_trajectory_and_reset(self) -> TrajectoryContainerBatchActorCritic:
        """
            1.  Return the last sampled trajectory in a TrajectoryContainer
            2.  Reset the container ready for the next trajectory sampling.

        :return: A TrajectoryContainerBatchActorCritic with a full trajectory
        :rtype: TrajectoryContainerBatchActorCritic
        """
        assert super()._q_values_computed, ("The return and the Q-values are not computed yet!!! "
                                            "Call the method trajectory_ended() before pop_trajectory_and_reset()")
        trajectory_containerBatchAC = TrajectoryContainerBatchActorCritic(super()._observations.copy(),
                                                                   super()._actions.copy(),
                                                                   super()._rewards.copy(),
                                                                   super()._q_values.copy(),
                                                                   super()._theReturn,
                                                                   super()._trj_collected,
                                                                   self._V_estimates.copy(),
                                                                   self._Advantages.copy())

        self._reset()
        return trajectory_containerBatchAC

    def _reset(self):
        super()._reset()
        self._V_estimates.clear()
        self._Advantages.clear()
        return None


class UniformeBatchContainerBatchActorCritic(UniformeBatchContainer):

    def __init__(self, batch_container_list: List[TrajectoryContainerBatchActorCritic], batch_constraint: int):
        self.V_estimate = []
        self.Advantage = []

        super().__init__(batch_container_list, batch_constraint)


def computage_Advantage(rewards, v_estimates):
    """
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

    :return:
    :rtype:
    """
    # Note: trick to acces timestep t+1 of the trajectory array
    #   |     [:-1] is the collected value at timestep t
    #   |     [1:] is the collected value at timestep t+1
    #   |     Requirement: extend the array by one blank space
    rew_t = np.array(rewards)
    v_estimates.append(0)
    V = np.array(v_estimates)
    V_t = V[:-1]
    V_tPrime = V[1:]

    # compute A for the full trajectory in one shot using element wise operation
    advantage = rew_t + V_tPrime - V_t
    return advantage