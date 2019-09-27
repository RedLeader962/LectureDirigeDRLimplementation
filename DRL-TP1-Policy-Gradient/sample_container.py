#!/usr/bin/env python
import numpy as np

from DRL_building_bloc import ExperimentSpec, GymPlayground, discounted_reward_to_go, reward_to_go


class TrajectoryContainer(object):
    def __init__(self, observations: list, actions: list, rewards: list, Q_values: list, trajectory_return: list) -> None:
        """
        Container for storage & retrieval of events collected at every timestep of a single batch of trajectories

        todo --> validate dtype for discrete case

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
        self.trajectory_lenght = len(self.actions)

    def __len__(self):
        """In timestep"""
        return self.trajectory_lenght


    def cut(self, max_lenght):
        self.observations = self.observations[:max_lenght]
        self.actions = self.actions[:max_lenght]
        self.rewards = self.rewards[:max_lenght]
        self.Q_values = self.Q_values[:max_lenght]

    def unpack(self) -> (list, list, list, list, float, int):
        """
        Unpack the full trajectorie as a tuple of numpy array

            Note: Q_values is a numpy ndarray view

        :return: (observations, actions, rewards, Q_values, trajectory_return, trajectory_lenght)
        :rtype: (list, list, list, list, float, int)
        """
        tc = self.observations, self.actions, self.rewards, self.Q_values, self.trajectory_return, self.trajectory_lenght
        return tc

    def __repr__(self):
        myRep = "\n::trajectory_container/\n"
        myRep += ".observations=\n{}\n\n".format(self.observations)
        myRep += ".actions=\n{}\n\n".format(self.actions)
        myRep += ".rewards=\n{}\n\n".format(self.rewards)
        myRep += ".Q_values=\n{}\n\n".format(self.Q_values)
        myRep += ".trajectory_return=\n{}\n\n".format(self.trajectory_return)
        myRep += ".trajectory_lenght=\n{}\n\n".format(self.trajectory_lenght)
        return myRep



class TrajectoryCollector(object):
    """
    Collect sampled timestep events and compute relevant information
    """
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground, discounted: bool = True):
        self._exp_spec = experiment_spec
        self._playground_spec = playground.get_environment_spec()
        self._observations = []
        self._actions = []
        self._rewards = []

        self._curent_trj_rewards = []
        self._trajectories_returns = []
        self._q_values = []
        self._trajectories_lenght = []

        self.step_count = 0
        self.trj_count = 0
        self._capacity = experiment_spec.batch_size_in_ts

        # self.trajectories_returns = np.sum(self.rewards, axis=None)
        self.discounted = discounted

    def __call__(self, observation: np.ndarray, action, reward: float, *args, **kwargs) -> None:
        """ Collect observation, action, reward for one timestep

        :type observation: np.ndarray
        :type action: int or float
        :type reward: float
        """
        try:
            assert not self.full()
            self._observations.append(observation)
            self._actions.append(action)
            self._rewards.append(reward)
            self._curent_trj_rewards.append(reward)
            self.step_count += 1

        except AssertionError as ae:
            raise ae
        return None

    def collect(self, observation: np.ndarray, action, reward: float) -> None:
        """ Collect observation, action, reward for one timestep

        :type observation: np.ndarray
        :type action: int or float
        :type reward: float
        """
        self.__call__(observation, action, reward)
        return None

    # def full(self) -> bool:
    #     return not self.step_count < self._capacity

    def trajectory_ended(self) -> float:
        """ Must be call at each trajectory end

                1. Compute the return
                2. compute the reward to go for the full trajectory
                3. Flush curent trj rewars acumulator
                :param adruptly:
                :type adruptly:
        """
        trj_return = self._compute_trajectory_return()
        self._compute_reward_to_go()
        self._trajectories_lenght.append(len(self._curent_trj_rewards))
        self._curent_trj_rewards = []                                       # flush curent trj rewars acumulator
        self.trj_count += 1
        return trj_return

    def _compute_trajectory_return(self) -> float:
        trj_return = float(np.sum(self._curent_trj_rewards, axis=None))
        self._trajectories_returns.append(trj_return)
        return trj_return

    def _compute_reward_to_go(self) -> None:
        if self.discounted:
            self._q_values += discounted_reward_to_go(self._curent_trj_rewards, experiment_spec=self._exp_spec)
        else:
            self._q_values += reward_to_go(self._curent_trj_rewards)
        return None



    def pop_trajectory_and_reset(self) -> TrajectoryContainer:
        """
            1.  Return the sampled trajectory in a TrajectoryContainer
            2.  Reset the container ready for the next trajectorie sampling.

        :return: A TrajectoryContainer with the full trajectory
        :rtype: TrajectoryContainer
        """


        trajectory_container = TrajectoryContainer(self._observations.copy(), self._actions.copy(),
                                                   self._rewards.copy(), self._q_values.copy(),
                                                   self._trajectories_returns.copy())

        self._reset()
        return trajectory_container

    def _reset(self):
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._curent_trj_rewards.clear()
        self._q_values.clear()
        self._trajectories_lenght.clear()
        self.step_count = 0
        self.trj_count = 0
        return None

    def __del__(self):
        self._reset()


class UniformeBatchContainer(object):
    def __init__(self, batch_container_list: list, batch_constraint: int):
        """
        Container for storage & retrieval of sampled trajectories
        Is a component of the UniformBatchCollector

        (nice to have) todo:implement --> convert each list to tupple
        Once initialize, what was stored in it become immutable

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
        assert len(self.batch_returns) == self.trj_count, "Nb of trajectories_returns collected differ from the container trj_count"
        batch_average_trjs_return = float(np.mean(self.batch_returns))
        batch_average_trjs_lenght = float(np.mean(self.batch_trjs_lenghts))
        return batch_average_trjs_return, batch_average_trjs_lenght


class UniformBatchCollector(object):
    """
    Collect sampled trajectories and agregate them in multiple batch container of uniforme dimension.
    (!) Is responsible of batch dimension uniformity across the experiement.

    (CRITICAL) todo:unit-test --> tensor shape must be equal across trajectory for loss optimization:

    note: Optimization consideration --> why collect numpy ndarray in python list?
      |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
      |       to a long ndarray than it is to append ndarray to each other

    """
    def __init__(self, capacity: ExperimentSpec):
        self.CAPACITY = capacity
        self._reset()

    def __call__(self, trajectory: TrajectoryContainer, *args, **kwargs) -> None:
        assert self.is_not_full(), "The batch is full: {} timesteps collected! Execute pop_batch_and_reset()".format(self._timestep_count)

        if self.remaining_space < len(trajectory):              # (Priority) todo:unit-test
            """ Cut the trajectory and append to batch """
            trajectory.cut(max_lenght=self.remaining_space)
            assert len(trajectory) + self.remaining_space == self.CAPACITY

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
        container = UniformeBatchContainer(self.trajectories_list, exp_spec)

        assert container
        # reset
        self._reset()
        return container

    def _reset(self):
        self.trajectories_list = []
        self._timestep_count = 0
        self._trajectory_count = 0
        self.remaining_space = self.CAPACITY