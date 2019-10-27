# coding=utf-8
from typing import List, Tuple, Any, Iterable

import numpy as np

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground

from blocAndTools.container.samplecontainer import TrajectoryContainer, TrajectoryCollector
from blocAndTools.container.samplecontainer import UniformeBatchContainer, UniformBatchCollector
from blocAndTools.temporal_difference_computation import computhe_the_Advantage, compute_TD_target


class TrajectoryContainerBatchAdvantage(TrajectoryContainer):
    """
    Container for storage & retrieval of events collected at every timestep of a trajectories
    for Batch Actor-Critic algorithm
    """
    __slots__ = ['obs_t',
                 'actions',
                 'rewards',
                 'obs_tPrime',
                 'Q_values',
                 'trajectory_return',
                 '_trajectory_lenght',
                 'trajectory_id',
                 'V_estimates',
                 'Advantage']

    def __init__(self, obs_t: list, actions: list, rewards: list, Q_values: list, trajectory_return: list,
                 trajectory_id, obs_tPrime: list = None, V_estimates: list = None, Advantage: list = None) -> None:

        self.obs_tPrime = obs_tPrime
        self.V_estimates = V_estimates
        self.Advantage = Advantage
        super().__init__(obs_t, actions, rewards, Q_values, trajectory_return, trajectory_id)

    def cut(self, max_lenght):
        """Down size the number of timestep stored in the container"""
        self.obs_tPrime = self.obs_tPrime[:max_lenght]
        self.V_estimates = self.V_estimates[:max_lenght]
        self.Advantage = self.Advantage[:max_lenght]
        super().cut(max_lenght)

    def unpack(self) -> Tuple[list, list, list, list, float, int, list, list, list]:
        """
        Unpack the full trajectorie as a tuple of numpy array

        :return: (obs_t, actions, rewards, Q_values, trajectory_return, _trajectory_lenght, V_estimate, obs_tPrime, Advantage )
        :rtype: (list, list, list, list, float, int, list, list, list)
        """
        # (nice to have) todo:refactor --> as a namedtuple
        unpacked_super = super().unpack()

        observations, actions, rewards, Q_values, trajectory_return, _trajectory_lenght = unpacked_super

        return observations, actions, rewards, Q_values, trajectory_return, _trajectory_lenght, \
               self.V_estimates, self.obs_tPrime, self.Advantage

    def __repr__(self):
        myRep = super().__repr__()
        myRep += ".obs_tPrime=\n{}\n\n".format(self.obs_tPrime)
        myRep += ".V_estimates=\n{}\n\n".format(self.V_estimates)
        myRep += ".Advantage=\n{}\n\n".format(self.Advantage)
        return myRep


class TrajectoryCollectorBatchAdvantage(TrajectoryCollector):
    """
    Collect timestep event of single trajectory for Batch Actor-Critic algorihm

        1. Collect sampled timestep events of a single trajectory,
        2. On trajectory end:
            a. Compute relevant information
            b. Output a TrajectoryContainer feed with collected sample
            c. Reset ready for next trajectory
    """
    q_values: list

    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground, discounted: bool = True):

        self.obs_tPrime = []
        self.V_estimates = []
        self.Advantage = []
        self._advantages_computed = False

        super().__init__(experiment_spec, playground, discounted)

    def collect_tPrime(self, obs_t: np.ndarray, action_t, reward_t: float, obs_tPrime: np.ndarray) -> None:
        """ Collect observation, action, reward and V estimate for one timestep
        """
        self.obs_tPrime = obs_tPrime
        super().collect_OAR(obs_t, action_t, reward_t)

    def collect_V(self, observation: np.ndarray, action, reward: float, V_estimate: float = None) -> None:
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
        self.set_Qvalues(compute_TD_target(self.rewards, self.V_estimates, self._exp_spec.discout_factor).tolist())
        return None

    def set_advantage(self, advantages):

        # (Priority) todo:refactor --> pull the Advantage computation outside of the container:
        #                    - sepration of concern
        #                    - computation done on the last trajectory collected is bogus because it is most likely cut

        # (CRITICAL) todo:validate --> mute temporarely for reference implementation:
        # # compute the advantage
        # aTrj_Advantages = computhe_the_Advantage(aTrj_rews, aTrj_Values).tolist()
        # assert len(aTrj_Advantages) == len(aTrj_acts), "Problem with Advantage computation"
        # self.batch_Advantages += aTrj_Advantages

        assert self._advantages_computed is False
        assert isinstance(advantages, list)
        assert len(advantages) == len(self.rewards)
        self.Advantage = advantages
        self._advantages_computed = True
        return None


    def pop_trajectory_and_reset(self) -> TrajectoryContainerBatchAdvantage:
        """
            1.  Return the last sampled trajectory in a TrajectoryContainer
            2.  Reset the container ready for the next trajectory sampling.

        :return: A TrajectoryContainerBatchOARV with a full trajectory
        :rtype: TrajectoryContainerBatchOARV
        """
        assert self._q_values_computed, ("The return and the Q-values are not computed yet!!! "
                                            "Call the method trajectory_ended() before pop_trajectory_and_reset()")
        trajectory_containerBatchAC = TrajectoryContainerBatchAdvantage(obs_t=self.observations.copy(),
                                                                        actions=self.actions.copy(),
                                                                        rewards=self.rewards.copy(),
                                                                        Q_values=self._q_values.copy(),
                                                                        trajectory_return=self.theReturn,
                                                                        trajectory_id=self._trj_collected,
                                                                        obs_tPrime=self.obs_tPrime.copy(),
                                                                        V_estimates=self.V_estimates.copy(),
                                                                        Advantage=self.Advantage.copy())

        self._reset()
        return trajectory_containerBatchAC

    def _reset(self):
        super()._reset()
        self.V_estimates.clear()
        self.obs_tPrime.clear()
        self.Advantage.clear()
        self._advantages_computed = False
        return None


class UniformeBatchContainerBatchAdvantage(UniformeBatchContainer):
    def __init__(self, trj_container_batch: List[TrajectoryContainerBatchAdvantage], batch_constraint: int, id):
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
        self.batch_obs_tPrime = []
        self.batch_Values_estimate = []
        self.batch_Advantages = []

        super().__init__(trj_container_batch, batch_constraint, self.batch_idx)

        # (CRITICAL) todo:validate --> mute temporarely for reference implementation:
        # # normalize advantage
        # tmp_Adv: np.ndarray = np.array(self.batch_Advantages)
        # Adv_mean = tmp_Adv.mean()
        # Adv_std = tmp_Adv.std()
        #
        # self.batch_Advantages = ((tmp_Adv - Adv_mean) / Adv_std).tolist()

    def _container_feed_on_init_hook(self, aTrjContainer: TrajectoryContainerBatchAdvantage):
        aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght, aTrj_Values, aTrj_obs_tPrime, aTrj_Adv = aTrjContainer.unpack()

        self.batch_obs_tPrime += aTrj_obs_tPrime
        self.batch_Values_estimate += aTrj_Values
        self.batch_Advantages += aTrj_Adv

        return aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght

    def unpack_all(self) -> Tuple[Any, list, list, list]:
        """
        Unpack the full epoch batch of collected trajectories in lists of numpy ndarray

        :return: (batch_observations, batch_actions, batch_Qvalues,
                    batch_returns, batch_trjs_lenghts, total_timestep_collected, nb_of_collected_trjs,
                     batch_obs_tPrime, batch_Values_estimate, batch_Advantages)
        :rtype: (list, list, list, list, list, int, int, list, list, list)
        """
        unpack_super = super().unpack_all()
        return (*unpack_super, self.batch_obs_tPrime.copy(), self.batch_Values_estimate.copy(), self.batch_Advantages.copy())

class UniformBatchCollectorBatchAdvantage(UniformBatchCollector):
    """
    Collect sampled trajectories and agregate them in multiple batch container of uniforme dimension
    for batch Actor-Critic algorithm.
    (!) Is responsible of batch dimension uniformity across the experiement.

    note: Optimization consideration --> why collect numpy ndarray in python list?
      |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
      |       to a long ndarray than it is to append ndarray to each other

    """

    def pop_batch_and_reset(self) -> UniformeBatchContainerBatchAdvantage:
        """
        :return: A batch of concatenated trajectories component
        :rtype: UniformeBatchContainerBatchOARV
        """
        container = UniformeBatchContainerBatchAdvantage(self.trajectories_list, self.CAPACITY, self.batch_idx)

        # reset
        self._reset()
        return container


