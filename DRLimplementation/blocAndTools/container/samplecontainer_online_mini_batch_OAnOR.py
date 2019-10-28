# coding=utf-8
from typing import List, Tuple, Any, Iterable

import numpy as np
from collections import namedtuple
from dataclasses import dataclass

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground

from blocAndTools.container.samplecontainer import TrajectoryContainer, TrajectoryCollector
from blocAndTools.container.samplecontainer import UniformeBatchContainer, UniformBatchCollector
from blocAndTools.temporal_difference_computation import computhe_the_Advantage, compute_TD_target

@dataclass()
class MiniBatch:
    __slots__ = ['obs_t', 'act_t', 'obs_tPrime', 'rew_t']

    obs_t: list
    act_t: list
    obs_tPrime: list
    rew_t: list


class TrajectoryContainerMiniBatchOnlineOAnOR(TrajectoryContainer):
    """
    Container for storage & retrieval of events collected at every timestep of a trajectories
    for Batch Actor-Critic algorithm
    """
    __slots__ = ['obs_t',
                 'actions',
                 'rewards',
                 'trajectory_return',
                 '_trajectory_lenght',
                 'trajectory_id',
                 'actor_losses',
                 'critic_losses']

    def __init__(self, obs_t: list, actions: list, rewards: list, trajectory_return: float,
                 trajectory_id, actor_losses: list = None, critic_losses: list = None) -> None:

        self.actor_losses = actor_losses
        self.critic_losses = critic_losses
        super().__init__(obs_t, actions, rewards, None, trajectory_return, trajectory_id)

    def cut(self, max_lenght):
        """Down size the number of timestep stored in the container"""
        raise UserWarning("This method is not supose to be used with this TrajectoryContainer subclass")

    def unpack(self) -> Tuple[list, list, list, float, int, list, list]:
        """
        Unpack the full trajectorie as a tuple of numpy array

        :return: (obs_t, actions, rewards, trajectory_return, _trajectory_lenght, actor_losses, critic_losses)
        :rtype: (list, list, list, float, int, list, list)
        """
        # (nice to have) todo:refactor --> as a namedtuple

        tc = (self.obs_t, self.actions, self.rewards,
              self.trajectory_return, self._trajectory_lenght, self.actor_losses, self.critic_losses)

        return tc

    def __repr__(self):
        myRep = super().__repr__()
        myRep += ".actor_losses=\n{}\n\n".format(self.actor_losses)
        myRep += ".critic_losses=\n{}\n\n".format(self.critic_losses)
        return myRep


class TrajectoryCollectorMiniBatchOnlineOAnOR(TrajectoryCollector):
    """
    Collect timestep event of single trajectory for Batch Actor-Critic algorihm

        1. Collect sampled timestep events of a single trajectory,
        2. Output on demande minibatch of collected timestep for learning with computed Bootstrap estimate target
        2. On trajectory end:
            a. Output a TrajectoryContainerMiniBatchOnlineOAnORV feed with collected sample
            b. Reset and ready for next trajectory
    """
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground, discounted: bool = True,
                 mini_batch_capacity: int = 10):
        self.mini_batch_capacity = mini_batch_capacity
        self.obs_tPrime = []

        self._trjCollector_minibatch_runing_idx = 0
        self._current_minibatch_size = 0
        super().__init__(experiment_spec, playground, discounted)
        self.actor_losses = []
        self.critic_losses = []

    def minibatch_is_full(self) -> bool:
        return self._current_minibatch_size >= self.mini_batch_capacity

    def collect_OAnOR(self, obs_t: np.ndarray, act_t, obs_tPrime: np.ndarray, rew_t: float) -> None:
        """ Collect obs_t, act_t, obs_tPrime, rew_t and V estimate for one timestep
        """

        assert not self.minibatch_is_full(), ("The minibatch is full: {} timesteps collected! "
                                              "Execute compute_Qvalues_as_BootstrapEstimate() "
                                              "than get_minibatch()").format(self._current_minibatch_size)

        self.obs_tPrime.append(obs_tPrime)
        super().collect_OAR(obs_t, act_t, rew_t)
        self._current_minibatch_size += 1
        return None

    def set_Qvalues(self, Qvalues: list) -> None:
        raise UserWarning("This method is not supose to be used with this TrajectoryContainer subclass")

    def get_minibatch(self):
        # (!) Dont assert if minibatch is full. The last minibatch of the trajectory will be smaller than the other

        mb_idx = self._trjCollector_minibatch_runing_idx

        mini_batch = MiniBatch(obs_t=self.observations[mb_idx:], act_t=self.actions[mb_idx:],
                               obs_tPrime=self.obs_tPrime[mb_idx:], rew_t=self.rewards[mb_idx:])

        self._reset_minibatch_internal_state()
        return mini_batch

    def _reset_minibatch_internal_state(self):
        self._current_minibatch_size = 0
        self._trjCollector_minibatch_runing_idx = len(self.actions)
        return None

    def compute_Qvalues_as_BootstrapEstimate(self) -> None:
        raise UserWarning("This method is not supose to be used with this TrajectoryContainer subclass")

    def compute_Qvalues_as_rewardToGo(self) -> None:
        raise UserWarning("This method is not supose to be used with this TrajectoryContainer subclass")

    def collect_loss(self, actor_loss, critic_loss):
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        return None

    def pop_trajectory_and_reset(self) -> TrajectoryContainerMiniBatchOnlineOAnOR:
        """
            1.  Return the last sampled trajectory in a TrajectoryContainerMiniBatchOnlineOAnORV
            2.  Reset the container ready for the next trajectory sampling.

        :return: A TrajectoryContainerMiniBatchOnlineOAnORV with a full trajectory
        :rtype: TrajectoryContainerMiniBatchOnlineOAnORV
        """
        trajectory_containerBatchAC = TrajectoryContainerMiniBatchOnlineOAnOR(obs_t=self.observations.copy(),
                                                                              actions=self.actions.copy(),
                                                                              rewards=self.rewards.copy(),
                                                                              trajectory_return=self.theReturn,
                                                                              trajectory_id=self._trj_collected,
                                                                              actor_losses=self.actor_losses.copy(),
                                                                              critic_losses=self.critic_losses.copy())

        self._reset()
        return trajectory_containerBatchAC

    def _reset(self):
        super()._reset()

        self.obs_tPrime.clear()

        self._trjCollector_minibatch_runing_idx = 0
        self._current_minibatch_size = 0

        self.actor_losses.clear()
        self.critic_losses.clear()
        return None


class UnconstrainedExperimentStageContainerOnlineAACnoV(UniformeBatchContainer):
    def __init__(self, trj_container_batch: List[TrajectoryContainerMiniBatchOnlineOAnOR], batch_constraint: int, batch_id: int):
        """
        Container for storage & retrieval of sampled trajectories history & statistic
        (!) Be advise, this container do not inforce uniformity constraint over collected timestep.
        All container will be of uneven timestep lenght & they keep all collected trajectory complete (uncut)
        Used with Online Actor-Critic algorihm

        Purpose: statistic book-keeping.

        It's a component of the ExperimentStageCollectorOnlineAAC

        :param batch_constraint: max capacity measured in timestep
        :type batch_constraint: int
        :param trj_container_batch: Take a list of TrajectoryContainer instance fulled with collected timestep events.
        :type trj_container_batch: List[TrajectoryContainer]
        :param batch_id: the batch idx number
        :type batch_id: int
        """
        self.actor_losses = []
        self.critic_losses = []
        super().__init__(trj_container_batch, batch_constraint, batch_id)

    def _container_feed_on_init_hook(self, aTrjContainer: TrajectoryContainerMiniBatchOnlineOAnOR):
        aTrj_obss, aTrj_acts, aTrj_rews, aTrj_return, aTrj_lenght, aTrj_actor_losses, aTrj_critic_losses = aTrjContainer.unpack()

        self.actor_losses += aTrj_actor_losses
        self.critic_losses += aTrj_critic_losses

        return aTrj_obss, aTrj_acts, aTrj_rews, [], aTrj_return, aTrj_lenght

    def _check_uniformity_constraint(self, batch_constraint: Any) -> None:
        """
        The function muted since this subclass does not inforce uniformity across stages
        """
        pass

    def unpack_all(self) -> Tuple[Any, list, list]:
        """
        Unpack the full epoch batch of collected trajectories in lists of numpy ndarray

        :return: (batch_observations, batch_actions, batch_Qvalues,
                    batch_returns, batch_trjs_lenghts, total_timestep_collected, nb_of_collected_trjs,
                     stage_Values_estimate, actor_losses, critic_losses)
        :rtype: (list, list, list, list, list, int, int, list, list)
        """
        # unpack_super = super().unpack_all()
        trajectories_copy = (self.batch_observations.copy(), self.batch_actions.copy(),
                             self.batch_returns.copy(), self.batch_trjs_lenghts, self._timestep_count, self.__len__(),
                             self.actor_losses.copy(), self.critic_losses.copy())
        return trajectories_copy

    def get_stage_mean_loss(self) -> (float, float):
        """
        Compute mean loss over stage for actor and critic

        :return: (actor_mean_loss, critic_mean_loss)
        :rtype: (float, float)
        """
        actor_mean_loss = float(np.array(self.actor_losses).mean())
        critic_mean_loss = float(np.array(self.critic_losses).mean())
        return actor_mean_loss, critic_mean_loss

class ExperimentStageCollectorOnlineAACnoV(UniformBatchCollector):
    """
    Collect sampled trajectories and agregate them in multiple stage container of even number of trajectories
    for online Actor-Critic algorithm

    Purpose: statistic book-keeping.
    capacity: is on a trajectory count scale

    (!) Keep in mind that the size of containers produced on a timestep scale will be UNEVEN )

    note: Optimization consideration --> why collect numpy ndarray in python list?
      |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
      |       to a long ndarray than it is to append ndarray to each other

    """

    def is_not_full(self) -> bool:
        return self._trajectory_count < self.CAPACITY

    def internal_state(self) -> namedtuple:
        """Testing utility"""
        ExperimentStatsAndHistoryCollectorInternalState = namedtuple('ExperimentStatsAndHistoryCollectorInternalState',
                                                        ['trajectories_list', 'timestep_count',
                                                         'trajectory_count', 'remaining_stage_space'])

        return ExperimentStatsAndHistoryCollectorInternalState(self.trajectories_list, self._timestep_count,
                                                               self._trajectory_count, self.remaining_stage_space)

    def __call__(self, trajectory: TrajectoryContainerMiniBatchOnlineOAnOR, *args, **kwargs) -> None:
        assert self.is_not_full(), ("The batch is full: {} timesteps collected! "
                                    "Execute pop_batch_and_reset()").format(self._timestep_count)

        self.trajectories_list.append(trajectory)
        self._trajectory_count += 1
        self._timestep_count += trajectory.__len__()
        self.remaining_stage_space -= 1
        return None

    def pop_batch_and_reset(self) -> UnconstrainedExperimentStageContainerOnlineAACnoV:
        """
        :return: A batch of concatenated trajectories component
        :rtype: UnconstrainedExperimentStageContainerOnlineAAC
        """
        container = UnconstrainedExperimentStageContainerOnlineAACnoV(self.trajectories_list, self.CAPACITY, self.batch_idx)

        # reset
        self._reset()
        return container

    def _reset(self):
        self.trajectories_list = []
        self._timestep_count = 0
        self._trajectory_count = 0
        self.remaining_stage_space = self.CAPACITY


