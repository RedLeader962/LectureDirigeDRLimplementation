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
    __slots__ = ['obs_t', 'act_t', 'obs_tPrime', 'rew_t', 'q_values_t']

    obs_t: list
    act_t: list
    obs_tPrime: list
    rew_t: list
    q_values_t: list


class TrajectoryContainerMiniBatchOnlineOAnORV(TrajectoryContainer):
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
                 'V_estimates',
                 'actor_losses',
                 'critic_losses']

    def __init__(self, obs_t: list, actions: list, rewards: list, Q_values: list, trajectory_return: float,
                 trajectory_id, V_estimates: list = None, actor_losses: list = None, critic_losses: list = None) -> None:

        self.V_estimates = V_estimates
        self.actor_losses = actor_losses
        self.critic_losses = critic_losses
        super().__init__(obs_t, actions, rewards, Q_values, trajectory_return, trajectory_id)

    def cut(self, max_lenght):
        """Down size the number of timestep stored in the container"""
        raise UserWarning("This method is not supose to be used with this TrajectoryContainer subclass")
        # self.V_estimates = self.V_estimates[:max_lenght]
        # super().cut(max_lenght)

    def unpack(self) -> Tuple[list, list, list, list, float, int, list, list, list]:
        """
        Unpack the full trajectorie as a tuple of numpy array

        :return: (obs_t, actions, rewards, Q_values, trajectory_return, _trajectory_lenght, V_estimate, actor_losses, critic_losses)
        :rtype: (list, list, list, list, float, int, list, list, list)
        """
        # (nice to have) todo:refactor --> as a namedtuple
        unpacked_super = super().unpack()

        observations, actions, rewards, Q_values, trajectory_return, _trajectory_lenght = unpacked_super

        return (observations, actions, rewards, Q_values, trajectory_return, _trajectory_lenght,
                self.V_estimates, self.actor_losses, self.critic_losses)

    def __repr__(self):
        myRep = super().__repr__()
        myRep += ".V_estimates=\n{}\n\n".format(self.V_estimates)
        myRep += ".actor_losses=\n{}\n\n".format(self.actor_losses)
        myRep += ".critic_losses=\n{}\n\n".format(self.critic_losses)
        return myRep


class TrajectoryCollectorMiniBatchOnlineOAnORV(TrajectoryCollector):
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
        self.V_estimates = []

        self._trjCollector_minibatch_runing_idx = 0
        self._current_minibatch_size = 0
        super().__init__(experiment_spec, playground, discounted)
        self.q_values = []
        self.actor_losses = []
        self.critic_losses = []

    def minibatch_is_full(self) -> bool:
        return self._current_minibatch_size >= self.mini_batch_capacity

    def collect_OAnORV(self, obs_t: np.ndarray, act_t, obs_tPrime: np.ndarray, rew_t: float, V_estimate: float) -> None:
        """ Collect obs_t, act_t, obs_tPrime, rew_t and V estimate for one timestep
        """

        assert not self.minibatch_is_full(), ("The minibatch is full: {} timesteps collected! "
                                              "Execute compute_Qvalues_as_BootstrapEstimate() "
                                              "than get_minibatch()").format(self._current_minibatch_size)

        self.obs_tPrime.append(obs_tPrime)
        self.V_estimates.append(V_estimate)
        super().collect_OAR(obs_t, act_t, rew_t)
        self._current_minibatch_size += 1
        return None

    def set_Qvalues(self, Qvalues: list) -> None:
        """
        Qvalues must be computed explicitely before minibatch_pop using eiter methode:
                - set_Qvalues,
                - or compute_Qvalues_as_rewardToGo
        """
        assert not self._q_values_computed
        assert isinstance(Qvalues, list)

        assert len(Qvalues) == len(self.actions) - len(self.q_values)
        self.q_values += Qvalues

        self._q_values_computed = True
        return None

    def get_minibatch(self):
        # (!) Dont assert if minibatch is full. The last minibatch of the trajectory will be smaller than the other

        assert self._q_values_computed, ("The Q-values are not computed yet!!! "
                                         "Call the method set_Qvalues() or compute_Qvalues_as_BootstrapEstimate()"
                                         " before get_minibatch()")

        mb_idx = self._trjCollector_minibatch_runing_idx

        mini_batch = MiniBatch(obs_t=self.observations[mb_idx:], act_t=self.actions[mb_idx:],
                               obs_tPrime=self.obs_tPrime[mb_idx:], rew_t=self.rewards[mb_idx:],
                               q_values_t=self.q_values[mb_idx:])

        self._reset_minibatch_internal_state()
        return mini_batch

    def _reset_minibatch_internal_state(self):
        self._q_values_computed = False
        self._current_minibatch_size = 0
        self._trjCollector_minibatch_runing_idx = len(self.actions)
        return None

    def compute_Qvalues_as_BootstrapEstimate(self) -> None:
        """
        Qvalues must be computed explicitely before minibatch_pop using eiter methode:
                - set_Qvalues,
                - or compute_Qvalues_as_BootstrapEstimate
        """
        mb_idx = self._trjCollector_minibatch_runing_idx
        TD_target: list = compute_TD_target(self.rewards[mb_idx:], self.V_estimates[mb_idx:],
                                            self._exp_spec.discout_factor).tolist()
        self.set_Qvalues(TD_target)
        return None

    def collect_loss(self, actor_loss, critic_loss):
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        return None

    def pop_trajectory_and_reset(self) -> TrajectoryContainerMiniBatchOnlineOAnORV:
        """
            1.  Return the last sampled trajectory in a TrajectoryContainerMiniBatchOnlineOAnORV
            2.  Reset the container ready for the next trajectory sampling.

        :return: A TrajectoryContainerMiniBatchOnlineOAnORV with a full trajectory
        :rtype: TrajectoryContainerMiniBatchOnlineOAnORV
        """
        assert not self._q_values_computed, ("The return and the Q-values are not computed yet!!! "
                                             "Call the method trajectory_ended() before pop_trajectory_and_reset()")
        trajectory_containerBatchAC = TrajectoryContainerMiniBatchOnlineOAnORV(obs_t=self.observations.copy(),
                                                                               actions=self.actions.copy(),
                                                                               rewards=self.rewards.copy(),
                                                                               Q_values=self.q_values.copy(),
                                                                               trajectory_return=self.theReturn,
                                                                               trajectory_id=self._trj_collected,
                                                                               V_estimates=self.V_estimates.copy(),
                                                                               actor_losses=self.actor_losses.copy(),
                                                                               critic_losses=self.critic_losses.copy())

        self._reset()
        return trajectory_containerBatchAC

    def _reset(self):
        super()._reset()

        self.obs_tPrime.clear()
        self.V_estimates.clear()

        self._trjCollector_minibatch_runing_idx = 0
        self._current_minibatch_size = 0

        self.q_values = []  # is required since q_values are not handle the same way as in the parent class
        self.actor_losses.clear()
        self.critic_losses.clear()
        return None


class UnconstrainedExperimentStageContainerOnlineAAC(UniformeBatchContainer):
    def __init__(self, trj_container_batch: List[TrajectoryContainerMiniBatchOnlineOAnORV], batch_constraint: int, batch_id: int):
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
        self.stage_Values_estimate = []
        self.actor_losses = []
        self.critic_losses = []
        super().__init__(trj_container_batch, batch_constraint, batch_id)

    def _container_feed_on_init_hook(self, aTrjContainer: TrajectoryContainerMiniBatchOnlineOAnORV):
        aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght, aTrj_Values, aTrj_actor_losses, aTrj_critic_losses = aTrjContainer.unpack()

        self.stage_Values_estimate += aTrj_Values
        self.actor_losses += aTrj_actor_losses
        self.critic_losses += aTrj_critic_losses

        return aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght

    def _check_uniformity_constraint(self, batch_constraint: Any) -> None:
        """
        The function muted since this subclass does not inforce uniformity across stages
        """
        pass

    def unpack_all(self) -> Tuple[Any, list, list, list]:
        """
        Unpack the full epoch batch of collected trajectories in lists of numpy ndarray

        :return: (batch_observations, batch_actions, batch_Qvalues,
                    batch_returns, batch_trjs_lenghts, total_timestep_collected, nb_of_collected_trjs,
                     stage_Values_estimate, actor_losses, critic_losses)
        :rtype: (list, list, list, list, list, int, int, list, list, list)
        """
        unpack_super = super().unpack_all()
        return (*unpack_super, self.stage_Values_estimate.copy(), self.actor_losses.copy(), self.critic_losses.copy())

    def get_stage_mean_loss(self) -> (float, float):
        """
        Compute mean loss over stage for actor and critic

        :return: (actor_mean_loss, critic_mean_loss)
        :rtype: (float, float)
        """
        actor_mean_loss = float(np.array(self.actor_losses).mean())
        critic_mean_loss = float(np.array(self.critic_losses).mean())
        return actor_mean_loss, critic_mean_loss

class ExperimentStageCollectorOnlineAAC(UniformBatchCollector):
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

    def __call__(self, trajectory: TrajectoryContainerMiniBatchOnlineOAnORV, *args, **kwargs) -> None:
        assert self.is_not_full(), ("The batch is full: {} timesteps collected! "
                                    "Execute pop_batch_and_reset()").format(self._timestep_count)

        self.trajectories_list.append(trajectory)
        self._trajectory_count += 1
        self._timestep_count += trajectory.__len__()
        self.remaining_stage_space -= 1
        return None

    def pop_batch_and_reset(self) -> UnconstrainedExperimentStageContainerOnlineAAC:
        """
        :return: A batch of concatenated trajectories component
        :rtype: UnconstrainedExperimentStageContainerOnlineAAC
        """
        container = UnconstrainedExperimentStageContainerOnlineAAC(self.trajectories_list, self.CAPACITY, self.batch_idx)

        # reset
        self._reset()
        return container

    def _reset(self):
        self.trajectories_list = []
        self._timestep_count = 0
        self._trajectory_count = 0
        self.remaining_stage_space = self.CAPACITY


