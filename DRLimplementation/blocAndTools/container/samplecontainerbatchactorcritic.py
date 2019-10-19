# coding=utf-8
from typing import List, Tuple, Any

import numpy as np

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground

from blocAndTools.container.samplecontainer import TrajectoryContainer, TrajectoryCollector
from blocAndTools.container.samplecontainer import UniformeBatchContainer, UniformBatchCollector
from blocAndTools.temporal_difference_computation import computhe_the_Advantage, compute_TD_target


class TrajectoryContainerBatchActorCritic(TrajectoryContainer):
    """
    Container for storage & retrieval of events collected at every timestep of a trajectories
    for Batch Actor-Critic algorithm
    """
    def __init__(self, observations: list, actions: list, rewards: list, Q_values: list, trajectory_return: list,
                 trajectory_id, V_estimates: list) -> None:

        self.V_estimates = V_estimates
        super().__init__(observations, actions, rewards, Q_values, trajectory_return, trajectory_id)

    def cut(self, max_lenght):
        """Down size the number of timestep stored in the container"""
        self.V_estimates = self.V_estimates[:max_lenght]
        super().cut(max_lenght)

    def unpack(self) -> Tuple[Any, list]:
        """
        Unpack the full trajectorie as a tuple of numpy array

        :return: (observations, actions, rewards, Q_values, trajectory_return, _trajectory_lenght, V_estimate)
        :rtype: (list, list, list, list, float, int, list)
        """
        # (nice to have) todo:refactor --> as a namedtuple
        tc = super().unpack()

        return (*tc, self.V_estimates)

    def __repr__(self):
        myRep = super().__repr__()
        myRep += ".V_estimates=\n{}\n\n".format(self.V_estimates)
        return myRep


class TrajectoryCollectorBatchActorCritic(TrajectoryCollector):
    """
    Collect timestep event of single trajectory for Batch Actor-Critic algorihm

        1. Collect sampled timestep events of a single trajectory,
        2. On trajectory end:
            a. Compute relevant information
            b. Output a TrajectoryContainer feed with collected sample
            c. Reset ready for next trajectory
    """
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground, discounted: bool = True,
                 MonteCarloTarget=True):

        self.MonteCarloTarget = MonteCarloTarget
        self._V_estimates = []
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

    def _compute_Q_values(self) -> None:
        if self.MonteCarloTarget:
            super()._compute_Q_values()
        else:
            self._q_values = list(compute_TD_target(self._rewards, self._V_estimates))
        return None

    def pop_trajectory_and_reset(self) -> TrajectoryContainerBatchActorCritic:
        """
            1.  Return the last sampled trajectory in a TrajectoryContainer
            2.  Reset the container ready for the next trajectory sampling.

        :return: A TrajectoryContainerBatchActorCritic with a full trajectory
        :rtype: TrajectoryContainerBatchActorCritic
        """
        assert self._q_values_computed, ("The return and the Q-values are not computed yet!!! "
                                            "Call the method trajectory_ended() before pop_trajectory_and_reset()")
        trajectory_containerBatchAC = TrajectoryContainerBatchActorCritic(self._observations.copy(),
                                                                   self._actions.copy(),
                                                                   self._rewards.copy(),
                                                                   self._q_values.copy(),
                                                                   self._theReturn,
                                                                   self._trj_collected,
                                                                   self._V_estimates.copy())

        self._reset()
        return trajectory_containerBatchAC

    def _reset(self):
        super()._reset()
        self._V_estimates.clear()
        return None


class UniformeBatchContainerBatchActorCritic(UniformeBatchContainer):
    def __init__(self, batch_container_list: List[TrajectoryContainerBatchActorCritic], batch_constraint: int):
        """
        Container for storage & retrieval of sampled trajectories for Batch Actor-Critic algorihm
        Is a component of the UniformBatchCollectorBatchActorCritic

        (nice to have) todo:implement --> make the container immutable: convert each list to tupple once initialized

        :param batch_constraint: max capacity measured in timestep
        :type batch_constraint: int
        :param batch_container_list: Take a list of TrajectoryContainer instance fulled with collected timestep events.
        :type batch_container_list: List[TrajectoryContainer]
        """
        self.batch_Values_estimate = []
        self.batch_Advantages = []

        super().__init__(batch_container_list, batch_constraint)

    def _container_feed_on_init_hook(self, aTrjContainer: TrajectoryContainerBatchActorCritic):
        aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght, aTrj_Values = aTrjContainer.unpack()

        self.batch_Values_estimate += aTrj_Values

        # compute the advantage
        aTrj_Advantages = list(computhe_the_Advantage(aTrj_rews, aTrj_Values))
        assert len(aTrj_Advantages) == len(aTrj_acts), "Problem with Advantage computation"
        self.batch_Advantages += aTrj_Advantages

        return aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght

    def unpack_all(self) -> Tuple[Any, list, list]:
        """
        Unpack the full epoch batch of collected trajectories in lists of numpy ndarray

        :return: (batch_observations, batch_actions, batch_Qvalues,
                    batch_returns, batch_trjs_lenghts, total_timestep_collected, nb_of_collected_trjs,
                     batch_Values_estimate, batch_Advantages)
        :rtype: (list, list, list, list, list, int, int, list, list)
        """
        unpack_super = super().unpack_all()
        return (*unpack_super, self.batch_Values_estimate.copy(), self.batch_Advantages.copy())

class UniformBatchCollectorBatchActorCritic(UniformBatchCollector):
    """
    Collect sampled trajectories and agregate them in multiple batch container of uniforme dimension
    for batch Actor-Critic algorithm.
    (!) Is responsible of batch dimension uniformity across the experiement.

    note: Optimization consideration --> why collect numpy ndarray in python list?
      |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
      |       to a long ndarray than it is to append ndarray to each other

    """

    def pop_batch_and_reset(self) -> UniformeBatchContainerBatchActorCritic:
        """
        :return: A batch of concatenated trajectories component
        :rtype: UniformeBatchContainerBatchActorCritic
        """
        container = UniformeBatchContainerBatchActorCritic(self.trajectories_list, self.CAPACITY)

        # reset
        self._reset()
        return container


