# coding=utf-8
import numpy as np
import pandas as pd
from collections import namedtuple
from typing import List, Tuple, Any

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from blocAndTools.rewardtogo import reward_to_go, discounted_reward_to_go
from blocAndTools.container.batchstats import BatchStats


class TrajectoryContainer(object):
    """
    Container for storage & retrieval of events collected at every timestep of a single batch of trajectories
    """
    __slots__ = ['obs_t',
                 'actions',
                 'rewards',
                 'Q_values',
                 'trajectory_return',
                 '_trajectory_lenght',
                 'trajectory_id',
                 ]

    def __init__(self, obs_t: list, actions: list, rewards: list, Q_values: list, trajectory_return: float,
                 trajectory_id) -> None:
        assert isinstance(obs_t, list) and isinstance(actions, list) and isinstance(rewards, list), "wrong argument type"
        assert len(obs_t) == len(actions) == len(rewards), "{} vs {} vs {} !!!".format(obs_t, actions, rewards)
        self.obs_t = obs_t
        self.actions = actions
        self.rewards = rewards
        self.Q_values = Q_values                        # Computed via reward to go or the discouted reward to go
        self.trajectory_return = trajectory_return
        self._trajectory_lenght = len(self.actions)

        # Internal state
        self.trajectory_id = trajectory_id

    def __len__(self):
        """In timestep"""
        return self._trajectory_lenght

    def cut(self, max_lenght):
        """Down size the number of timestep stored in the container"""
        self.obs_t = self.obs_t[:max_lenght]
        self.actions = self.actions[:max_lenght]
        self.rewards = self.rewards[:max_lenght]
        self.Q_values = self.Q_values[:max_lenght]

        # update trajectory lenght
        self._trajectory_lenght = len(self.actions)

    def unpack(self) -> Tuple[list, list, list, list, float, int]:
        """
        Unpack the full trajectorie as a tuple of numpy array

        :return: (obs_t, actions, rewards, Q_values, trajectory_return, _trajectory_lenght)
        :rtype: (list, list, list, list, float, int)
        """
        # (nice to have) todo:refactor --> as a namedtuple
        tc = self.obs_t, self.actions, self.rewards, self.Q_values, self.trajectory_return, self._trajectory_lenght
        return tc

    def __repr__(self):
        myRep = "\n::trajectory_container/\n"
        myRep += ".obs_t=\n{}\n\n".format(self.obs_t)
        myRep += ".actions=\n{}\n\n".format(self.actions)
        myRep += ".rewards=\n{}\n\n".format(self.rewards)
        myRep += ".Q_values=\n{}\n\n".format(self.Q_values)
        myRep += ".trajectory_return=\n{}\n\n".format(self.trajectory_return)
        myRep += "._trajectory_lenght=\n{}\n\n".format(self._trajectory_lenght)
        return myRep


class TrajectoryCollector(object):
    """
    Collect timestep event of single trajectory

        1. Collect sampled timestep events of a single trajectory,
        2. On trajectory end:
            a. Compute relevant information
            b. Output a TrajectoryContainer feed with collected sample
            c. Reset ready for next trajectory
    """
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground, discounted: bool = True):
        self._exp_spec = experiment_spec
        self._playground_spec = playground.get_environment_spec()
        self.discounted = discounted

        self.observations = []
        self.actions = []
        self.rewards = []

        self.q_values = None
        self.theReturn = None
        self.lenght = None

        # Internal state
        # (nice to have) todo:refactor --> using the namedtuple InertnalState:
        self._step_count_since_begining_of_training = 0
        self._trj_collected = 0
        self._q_values_computed = False

    def internal_state(self) -> namedtuple:
        """Testing utility"""
        TrajectoryCollectorInternalState = namedtuple('TrajectoryCollectorInternalState',
                                                      ['step_count_since_begining_of_training',
                                                       'trj_collected',
                                                       'q_values_computed_on_current_trj'], )

        return TrajectoryCollectorInternalState(self._step_count_since_begining_of_training,
                                                self._trj_collected,
                                                self._q_values_computed)

    def collect_OAR(self, observation: np.ndarray, action, reward: float) -> None:
        """ Collect observation, action, reward for one timestep

        :type observation: np.ndarray
        :type action: int or float
        :type reward: float
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self._step_count_since_begining_of_training += 1
        return None

    def trajectory_ended(self) -> float:
        """ Must be called at each trajectory end

        Compute:
            1. the trajectory lenght base on collected samples
            2. the Q-values
            3. the trajectory return

        :return: the trajectory return
        :rtype: float
        """
        self.lenght = len(self.actions)
        self._trj_collected += 1
        return self._compute_trajectory_return()

    def _compute_trajectory_return(self) -> float:
        trj_return = float(np.sum(self.rewards, axis=None))
        self.theReturn = trj_return
        return trj_return

    def set_Qvalues(self, Qvalues: list) -> None:
        """
        Qvalues must be computed explicitely before pop_trajectory_and_reset
        using eiter methode:
                - set_Qvalues,
                - or compute_Qvalues_as_rewardToGo
        """
        assert not self._q_values_computed
        assert isinstance(Qvalues, list)
        assert len(Qvalues) == len(self.rewards)
        self.q_values = Qvalues
        self._q_values_computed = True
        return None

    def compute_Qvalues_as_rewardToGo(self) -> None:
        """
        Qvalues must be computed explicitely before pop_trajectory_and_reset
        using eiter methode:
                - set_Qvalues,
                - or compute_Qvalues_as_rewardToGo
        """
        if self.discounted:
            self.set_Qvalues(discounted_reward_to_go(self.rewards, experiment_spec=self._exp_spec))
        else:
            self.set_Qvalues(reward_to_go(self.rewards))

        return None

    def pop_trajectory_and_reset(self) -> TrajectoryContainer:
        """
            1.  Return the last sampled trajectory in a TrajectoryContainer
            2.  Reset the container ready for the next trajectory sampling.

        :return: A TrajectoryContainer with a full trajectory
        :rtype: TrajectoryContainer
        """
        assert self._q_values_computed, ("The return and the Q-values are not computed yet!!! "
                                         "Call the method trajectory_ended() before pop_trajectory_and_reset()")
        trajectory_container = TrajectoryContainer(obs_t=self.observations.copy(),
                                                   actions=self.actions.copy(),
                                                   rewards=self.rewards.copy(),
                                                   Q_values=self.q_values.copy(),
                                                   trajectory_return=self.theReturn,
                                                   trajectory_id=self._trj_collected)

        self._reset()
        return trajectory_container

    def _reset(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()

        self.q_values = None
        self.theReturn = None
        self.lenght = None

        self._q_values_computed = False
        return None

    def __del__(self):
        self._reset()


class UniformeBatchContainer(object):
    def __init__(self, trj_container_batch: List[TrajectoryContainer], batch_constraint: int, batch_id: int):
        """
        Container for storage & retrieval of sampled trajectories
        Is a component of the UniformBatchCollector

        (nice to have) todo:implement --> make the container immutable: convert each list to tupple once initialized

        :param trj_container_batch: Take a list of TrajectoryContainer instance fulled of collected timestep events.
        :type trj_container_batch: List[TrajectoryContainer]
        :param batch_constraint: max capacity measured in timestep
        :type batch_constraint: int
        :param batch_id: the batch idx number
        :type batch_id: int
        """
        self.batch_observations = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_Qvalues = []
        self.batch_returns = []
        self.batch_trjs_lenghts = []
        self._timestep_count = 0
        self._trjs_count = len(trj_container_batch)
        self.batch_id = batch_id

        for aTrjContainer in trj_container_batch:
            assert isinstance(aTrjContainer, TrajectoryContainer), ("The trj_container_batch list must contain object "
                                                                    "of type TrajectoryContainer")

            aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght = self._container_feed_on_init_hook(
                aTrjContainer)

            # merge list
            self.batch_observations += aTrj_obss
            self.batch_actions += aTrj_acts
            self.batch_rewards += aTrj_rews
            self.batch_Qvalues += aTrj_Qs

            self.batch_returns.append(aTrj_return)
            self.batch_trjs_lenghts.append(aTrj_lenght)

            self._timestep_count += len(aTrjContainer)

        assert self._timestep_count == batch_constraint, ("The sum of each TrajectoryContainer lenght does not"
                                                          " respect the size contraint: "
                                                          "Expected {}, got {} !!! "
                                                          ).format(batch_constraint, self._timestep_count)

        # Note: Quick fix since data collected in batch stats are used in 2 separate methode
        #   |     already widely use over the code base
        self._batch_stats = self._compute_batch_stats()

    def _compute_batch_stats(self):
        stats = BatchStats(batch_id=self.batch_id, step_collected=self.__len__(),
                           trajectory_collected=self.trajectories_count(), batch_trjs_returns=self.batch_returns,
                           batch_trjs_lenghts=self.batch_trjs_lenghts)
        return stats

    def get_batch_stats(self) -> BatchStats:
        """
        Get statistic about the collected trajectories return and lenght such as mean, max, min and standard deviation

        :return: A BatchStats dataclass with computed batch statistic
        :rtype: BatchStats
        """
        return self._batch_stats

    def get_basic_metric(self) -> (float, float):
        """
        Compute batch relevant metric over this container stored sample

        :return: (batch_average_trjs_return, batch_average_trjs_lenght)
        :rtype: (float, float)
        """
        return self._batch_stats.mean_return, self._batch_stats.mean_trj_lenght

    def _container_feed_on_init_hook(self, aTrjContainer: TrajectoryContainer):
        """Utility fct: Expose the init process for subclassing"""
        aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght = aTrjContainer.unpack()

        # Note: add computation here in subclass methode >>>

        return aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght

    def __len__(self) -> int:
        return self._timestep_count

    def trajectories_count(self):
        return self._trjs_count

    def unpack_all(self) -> Tuple[list, list, list, list, list, int, int]:
        """
        Unpack the full epoch batch of collected trajectories in lists of numpy ndarray

        :return: (batch_observations, batch_actions, batch_Qvalues,
                    batch_returns, batch_trjs_lenghts, total_timestep_collected, nb_of_collected_trjs )
        :rtype: (list, list, list, list, list, int, int)
        """

        # (icebox) todo:assessment --> if the copy method still required?: it does only if the list content are numpy ndarray
        trajectories_copy = (self.batch_observations.copy(), self.batch_actions.copy(), self.batch_Qvalues.copy(),
                             self.batch_returns.copy(), self.batch_trjs_lenghts, self._timestep_count, self.__len__())
        return trajectories_copy

class UniformBatchCollector(object):
    """
    Collect sampled trajectories and agregate them in multiple batch container of uniforme dimension.
    Is also responsible of batch dimension uniformity across the experiement.

    (!) Note: Optimization consideration --> why collect sample in python list instead of numpy ndarray?
          |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
          |   to a long ndarray than it is to append ndarray to each other

    """
    batch_stats: List[BatchStats]

    def __init__(self, capacity: int):
        self.CAPACITY = capacity
        self._reset()
        self.batch_stats = []
        self.batch_idx = 0

    def internal_state(self) -> namedtuple:
        """Testing utility"""
        UniformBatchCollectorInternalState = namedtuple('UniformBatchCollectorInternalState',
                                                        ['trajectories_list', 'timestep_count',
                                                         'trajectory_count', 'remaining_batch_space'])

        return UniformBatchCollectorInternalState(self.trajectories_list, self._timestep_count,
                                                  self._trajectory_count, self.remaining_batch_space)

    def __call__(self, trajectory: TrajectoryContainer, *args, **kwargs) -> None:
        assert self.is_not_full(), "The batch is full: {} timesteps collected! Execute pop_batch_and_reset()".format(self._timestep_count)

        if self.remaining_batch_space < len(trajectory):
            """ Cut the trajectory and append to batch """
            trajectory.cut(max_lenght=self.remaining_batch_space)
            assert len(trajectory) - self.remaining_batch_space == 0, ("The trajectory to collect should be downsized but it's not. "
                                                                       "Actual downsized len: {} Expected: {}").format(len(trajectory), self.remaining_batch_space)

        self.trajectories_list.append(trajectory)
        self._trajectory_count += 1
        self._timestep_count += trajectory.__len__()
        self.remaining_batch_space -= trajectory.__len__()
        return None

    def collect(self, trajectory: TrajectoryContainer) -> None:
        self.__call__(trajectory)
        return None

    def is_not_full(self) -> bool:
        return self._timestep_count < self.CAPACITY

    def trj_collected_so_far(self) -> int:
        return self._trajectory_count

    def timestep_collected_so_far(self) -> int:
        return self._timestep_count

    def pop_batch_and_reset(self) -> UniformeBatchContainer:
        """
        :return: A batch of concatenated trajectories component
        :rtype: UniformeBatchContainer
        """
        container = UniformeBatchContainer(self.trajectories_list, self.CAPACITY, self.batch_idx)

        self.batch_stats.append(container.get_batch_stats())
        self.batch_idx += 1
        self._reset()
        return container

    def compute_experiment_stats(self):

        # batch_id: int
        # step_collected: int
        # trajectory_collected: int
        _mean_return = np.zeros(self.batch_idx)
        _max_return = np.zeros(self.batch_idx)
        _min_return = np.zeros(self.batch_idx)
        _std_return = np.zeros(self.batch_idx)
        _mean_trj_lenght = np.zeros(self.batch_idx)
        _max_trj_lenght = np.zeros(self.batch_idx)
        _min_trj_lenght = np.zeros(self.batch_idx)
        _std_trj_lenght = np.zeros(self.batch_idx)

        for i, each in enumerate(self.batch_stats):
            _mean_return[i] = each.mean_return
            _max_return[i] = each.max_return
            _min_return[i] = each.min_return
            _std_return[i] = each.std_return
            _mean_trj_lenght[i] = each.mean_trj_lenght
            _max_trj_lenght[i] = each.max_trj_lenght
            _min_trj_lenght[i] = each.min_trj_lenght
            _std_trj_lenght[i] = each.std_trj_lenght

        # (Priority) todo:unit-test --> validate stats computation:
        # (Priority) todo:implement --> build CSV or panda dataframe:
        # (Priority) todo:implement --> log to file:



    def _reset(self):
        self.trajectories_list = []
        self._timestep_count = 0
        self._trajectory_count = 0
        self.remaining_batch_space = self.CAPACITY
