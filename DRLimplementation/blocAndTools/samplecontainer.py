# coding=utf-8
import numpy as np
from collections import namedtuple

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from blocAndTools.rewardtogo import reward_to_go, discounted_reward_to_go


class TrajectoryContainer(object):
    def __init__(self, observations: list, actions: list, rewards: list, Q_values: list, trajectory_return: list,
                 trajectory_id) -> None:
        """
        Container for storage & retrieval of events collected at every timestep of a single batch of trajectories

        todo:assessment --> validate dtype for discrete case:

        Note: about dtype (source: Numpy doc)
         |      "This argument can only be used to 'upcast' the array.
         |          For downcasting, use the .astype(t) method."

        """
        assert isinstance(observations, list) and isinstance(actions, list) and isinstance(rewards, list), "wrong argument type"
        assert len(observations) == len(actions) == len(rewards), "{} vs {} vs {} !!!".format(observations, actions, rewards)
        self.observations = observations
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
        self.observations = self.observations[:max_lenght]
        self.actions = self.actions[:max_lenght]
        self.rewards = self.rewards[:max_lenght]
        self.Q_values = self.Q_values[:max_lenght]

        # update trajectory lenght
        self._trajectory_lenght = len(self.actions)

    def unpack(self) -> (list, list, list, list, float, int):
        """
        Unpack the full trajectorie as a tuple of numpy array

            Note: Q_values is a numpy ndarray view

        :return: (observations, actions, rewards, Q_values, trajectory_return, _trajectory_lenght)
        :rtype: (list, list, list, list, float, int)
        """
        # (nice to have) todo:refactor --> as a namedtuple
        tc = self.observations, self.actions, self.rewards, self.Q_values, self.trajectory_return, self._trajectory_lenght
        return tc

    def __repr__(self):
        myRep = "\n::trajectory_container/\n"
        myRep += ".observations=\n{}\n\n".format(self.observations)
        myRep += ".actions=\n{}\n\n".format(self.actions)
        myRep += ".rewards=\n{}\n\n".format(self.rewards)
        myRep += ".Q_values=\n{}\n\n".format(self.Q_values)
        myRep += ".trajectory_return=\n{}\n\n".format(self.trajectory_return)
        myRep += "._trajectory_lenght=\n{}\n\n".format(self._trajectory_lenght)
        return myRep


class TrajectoryCollector(object):
    """
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

        self._observations = []
        self._actions = []
        self._rewards = []

        self._q_values = None
        self._theReturn = None
        self._lenght = None

        # Internal state
        # (nice to have) todo:refactor --> using the namedtuple InetrnalState:
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
                                                self._q_values_computed, )

    def __call__(self, observation: np.ndarray, action, reward: float, *args, **kwargs) -> None:
        """ Collect observation, action, reward for one timestep

        :type observation: np.ndarray
        :type action: int or float
        :type reward: float
        """
        self._observations.append(observation)
        self._actions.append(action)
        self._rewards.append(reward)
        self._step_count_since_begining_of_training += 1

    def collect(self, observation: np.ndarray, action, reward: float) -> None:
        """ Collect observation, action, reward for one timestep

        :type observation: np.ndarray
        :type action: int or float
        :type reward: float
        """
        self.__call__(observation, action, reward)
        return None

    def trajectory_ended(self) -> float:
        """ Must be call at each trajectory end

        Compute:
            1. the trajectory lenght base on collected samples
            2. the Q-values
            3. the trajectory return

        :param: the trajectory return
        :rtype: float
        """
        self._lenght = len(self._actions)
        self._compute_Q_values()
        self._trj_collected += 1
        self._q_values_computed = True
        return self._compute_trajectory_return()

    def _compute_trajectory_return(self) -> float:
        trj_return = float(np.sum(self._rewards, axis=None))
        self._theReturn = trj_return
        return trj_return

    def _compute_Q_values(self) -> None:
        if self.discounted:
            self._q_values = discounted_reward_to_go(self._rewards, experiment_spec=self._exp_spec)
        else:
            self._q_values = reward_to_go(self._rewards)
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
        trajectory_container = TrajectoryContainer(self._observations.copy(), self._actions.copy(),
                                                   self._rewards.copy(), self._q_values.copy(), self._theReturn,
                                                   self._trj_collected)

        self._reset()
        return trajectory_container

    def _reset(self):
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()

        self._q_values = None
        self._theReturn = None
        self._lenght = None

        self._q_values_computed = False
        return None

    def __del__(self):
        self._reset()


class UniformeBatchContainer(object):
    def __init__(self, batch_container_list: list, batch_constraint: int):
        """
        Container for storage & retrieval of sampled trajectories
        Is a component of the UniformBatchCollector

        (nice to have) todo:implement --> make the container immutable: convert each list to tupple once initialized

        :param batch_constraint:
        :type batch_constraint:
        :param batch_container_list: Take a list of TrajectoryContainer instance fulled with collected timestep events.
        :type batch_container_list: [TrajectoryContainer, ...]
        """
        self.batch_observations = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_Qvalues = []
        self.batch_returns = []
        self.batch_trjs_lenghts = []
        self._timestep_count = 0
        self._trjs_count = len(batch_container_list)

        for aTrjContainer in batch_container_list:
            assert isinstance(aTrjContainer, TrajectoryContainer), "The list must contain object of type TrajectoryContainer"

            aTrj_obss, aTrj_acts, aTrj_rews, aTrj_Qs, aTrj_return, aTrj_lenght = aTrjContainer.unpack()

            # merge list
            self.batch_observations += aTrj_obss
            self.batch_actions += aTrj_acts
            self.batch_rewards += aTrj_rews
            self.batch_Qvalues += aTrj_Qs

            self.batch_returns.append(aTrj_return)
            self.batch_trjs_lenghts.append(aTrj_lenght)

            self._timestep_count += len(aTrjContainer)

        assert self._timestep_count == batch_constraint, ("The sum of each TrajectoryContainer lenght does not respect the size contraint: "
                                                          "Exepcted {}, got {} !!! ").format(batch_constraint, self._timestep_count)

    def __len__(self) -> int:
        return self._timestep_count

    def trajectories_count(self):
        return self._trjs_count

    def unpack_all(self) -> (list, list, list, list, list, int, int):
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

    def compute_metric(self) -> (float, float):
        """
        Compute batch relevant metric over this container stored sample

        :return: (batch_average_trjs_return, batch_average_trjs_lenght)
        :rtype: (float, float)
        """
        assert len(self.batch_returns) == self.trajectories_count(), "Nb of trajectories_returns collected differ from the container trj_count"
        batch_average_trjs_return = float(np.mean(self.batch_returns))
        batch_average_trjs_lenght = float(np.mean(self.batch_trjs_lenghts))
        return batch_average_trjs_return, batch_average_trjs_lenght


class UniformBatchCollector(object):
    """
    Collect sampled trajectories and agregate them in multiple batch container of uniforme dimension.
    (!) Is responsible of batch dimension uniformity across the experiement.

    note: Optimization consideration --> why collect numpy ndarray in python list?
      |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
      |       to a long ndarray than it is to append ndarray to each other

    """
    def __init__(self, capacity: ExperimentSpec):
        self.CAPACITY = capacity
        self._reset()

    def internal_state(self) -> namedtuple:
        """Testing utility"""
        UniformBatchCollectorInternalState = namedtuple('UniformBatchCollectorInternalState',
                                                        ['trajectories_list', 'timestep_count',
                                                         'trajectory_count', 'remaining_space'])

        return UniformBatchCollectorInternalState(self.trajectories_list, self._timestep_count,
                                                  self._trajectory_count, self.remaining_space)

    def __call__(self, trajectory: TrajectoryContainer, *args, **kwargs) -> None:
        assert self.is_not_full(), "The batch is full: {} timesteps collected! Execute pop_batch_and_reset()".format(self._timestep_count)

        if self.remaining_space < len(trajectory):
            """ Cut the trajectory and append to batch """
            trajectory.cut(max_lenght=self.remaining_space)
            assert len(trajectory) - self.remaining_space == 0, ("The trajectory to collect should be downsized but it's not. "
                                                                 "Actual downsized len: {} Expected: {}").format(len(trajectory),
                                                                                                                 self.remaining_space)

        self.trajectories_list.append(trajectory)
        self._trajectory_count += 1
        self._timestep_count += trajectory.__len__()
        self.remaining_space -= trajectory.__len__()
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
        container = UniformeBatchContainer(self.trajectories_list, self.CAPACITY)

        # reset
        self._reset()
        return container

    def _reset(self):
        self.trajectories_list = []
        self._timestep_count = 0
        self._trajectory_count = 0
        self.remaining_space = self.CAPACITY
