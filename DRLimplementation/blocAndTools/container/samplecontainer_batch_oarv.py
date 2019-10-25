# coding=utf-8
from typing import List, Tuple, Any, Iterable

import numpy as np

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground

from blocAndTools.container.samplecontainer import TrajectoryContainer, TrajectoryCollector
from blocAndTools.container.samplecontainer import UniformeBatchContainer, UniformBatchCollector
from blocAndTools.temporal_difference_computation import computhe_the_Advantage, compute_TD_target


class TrajectoryContainerBatchOARV(TrajectoryContainer):
    """
    Container for storage & retrieval of events collected at every timestep of a trajectories
    for Batch Actor-Critic algorithm
    """
    __slots__ = ['obs_t',
                 'actions',
                 'rewards',
                 'Q_values',
                 'trajectory_return',
                 '_trajectory_lenght',
                 'trajectory_id',
                 'V_estimates',]

    def __init__(self, obs_t: list, actions: list, rewards: list, Q_values: list, trajectory_return: list,
                 trajectory_id, V_estimates: list = None) -> None:

        self.V_estimates = V_estimates
        super().__init__(obs_t, actions, rewards, Q_values, trajectory_return, trajectory_id)

    def cut(self, max_lenght):
        """Down size the number of timestep stored in the container"""
        self.V_estimates = self.V_estimates[:max_lenght]
        super().cut(max_lenght)

    def unpack(self) -> Tuple[list, list, list, list, float, int, list]:
        """
        Unpack the full trajectorie as a tuple of numpy array

        :return: (obs_t, actions, rewards, Q_values, trajectory_return, _trajectory_lenght, V_estimate)
        :rtype: (list, list, list, list, float, int, list)
        """
        # (nice to have) todo:refactor --> as a namedtuple
        unpacked_super = super().unpack()

        observations, actions, rewards, Q_values, trajectory_return, _trajectory_lenght = unpacked_super

        return observations, actions, rewards, Q_values, trajectory_return, _trajectory_lenght, self.V_estimates

    def __repr__(self):
        myRep = super().__repr__()
        myRep += ".V_estimates=\n{}\n\n".format(self.V_estimates)
        return myRep


class TrajectoryCollectorBatchOARV(TrajectoryCollector):
    """
    Collect timestep event of single trajectory for Batch Actor-Critic algorihm

        1. Collect sampled timestep events of a single trajectory,
        2. On trajectory end:
            a. Compute relevant information
            b. Output a TrajectoryContainer feed with collected sample
            c. Reset ready for next trajectory
    """
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground, discounted: bool = True):
        self.V_estimates = []
        super().__init__(experiment_spec, playground, discounted)

    def collect_OARV(self, observation: np.ndarray, action, reward: float, V_estimate: float) -> None:
        """ Collect observation, action, reward and V estimate for one timestep
        """
        self.V_estimates.append(V_estimate)
        super().collect_OAR(observation, action, reward)

    def compute_Qvalues_as_BootstrapEstimate(self) -> None:
        """
        Qvalues must be computed explicitely before pop_trajectory_and_reset
        using eiter methode:
                - set_Qvalues,
                - compute_Qvalues_as_rewardToGo
                - or compute_Qvalues_as_BootstrapEstimate
        """
        # (Priority) todo:unit-test --> the stored result and cascading behavior:
        TD_target: list = compute_TD_target(self.rewards, self.V_estimates).tolist()
        self.set_Qvalues(TD_target)
        return None

    def pop_trajectory_and_reset(self) -> TrajectoryContainerBatchOARV:
        """
            1.  Return the last sampled trajectory in a TrajectoryContainerBatchOARV
            2.  Reset the container ready for the next trajectory sampling.

        :return: A TrajectoryContainerBatchOARV with a full trajectory
        :rtype: TrajectoryContainerBatchOARV
        """
        assert self._q_values_computed, ("The return and the Q-values are not computed yet!!! "
                                            "Call the method trajectory_ended() before pop_trajectory_and_reset()")
        trajectory_containerBatchAC = TrajectoryContainerBatchOARV(obs_t=self.observations.copy(),
                                                                   actions=self.actions.copy(),
                                                                   rewards=self.rewards.copy(),
                                                                   Q_values=self.q_values.copy(),
                                                                   trajectory_return=self.theReturn,
                                                                   trajectory_id=self._trj_collected,
                                                                   V_estimates=self.V_estimates.copy())

        self._reset()
        return trajectory_containerBatchAC

    def _reset(self):
        super()._reset()
        self.V_estimates.clear()
        return None


class UniformeBatchContainerBatchOARV(UniformeBatchContainer):
    def __init__(self, trj_container_batch: List[TrajectoryContainerBatchOARV], batch_constraint: int, batch_id: int):
        """
        Container for storage & retrieval of sampled trajectories for Batch Actor-Critic algorihm
        Is a component of the UniformBatchCollectorBatchOARV

        (nice to have) todo:implement --> make the container immutable: convert each list to tupple once initialized

        :param id:
        :type id:
        :param batch_constraint: max capacity measured in timestep
        :type batch_constraint: int
        :param trj_container_batch: Take a list of TrajectoryContainer instance fulled with collected timestep events.
        :type trj_container_batch: List[TrajectoryContainer]
        """
        self.batch_Values_estimate = []
        super().__init__(trj_container_batch, batch_constraint, batch_id)

    def _container_feed_on_init_hook(self, aTrjContainer: TrajectoryContainerBatchOARV):
        aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght, aTrj_Values = aTrjContainer.unpack()

        self.batch_Values_estimate += aTrj_Values

        return aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght

    def unpack_all(self) -> Tuple[Any, list]:
        """
        Unpack the full epoch batch of collected trajectories in lists of numpy ndarray

        :return: (batch_observations, batch_actions, batch_Qvalues,
                    batch_returns, batch_trjs_lenghts, total_timestep_collected, nb_of_collected_trjs,
                     batch_Values_estimate)
        :rtype: (list, list, list, list, list, int, int, list)
        """
        unpack_super = super().unpack_all()
        return (*unpack_super, self.batch_Values_estimate.copy())

class UniformBatchCollectorBatchOARV(UniformBatchCollector):
    """
    Collect sampled trajectories and agregate them in multiple batch container of uniforme dimension
    for batch Actor-Critic algorithm.
    (!) Is responsible of batch dimension uniformity across the experiement.

    note: Optimization consideration --> why collect numpy ndarray in python list?
      |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
      |       to a long ndarray than it is to append ndarray to each other

    """

    def pop_batch_and_reset(self) -> UniformeBatchContainerBatchOARV:
        """
        :return: A batch of concatenated trajectories component
        :rtype: UniformeBatchContainerBatchOARV
        """
        container = UniformeBatchContainerBatchOARV(self.trajectories_list, self.CAPACITY, self.batch_idx)

        # reset
        self._reset()
        return container


