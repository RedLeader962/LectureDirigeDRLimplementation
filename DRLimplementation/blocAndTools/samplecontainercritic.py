# coding=utf-8

import numpy as np
from collections import namedtuple

from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from blocAndTools.rewardtogo import reward_to_go, discounted_reward_to_go

from blocAndTools.samplecontainer import TrajectoryContainer, TrajectoryCollector
from blocAndTools.samplecontainer import  UniformeBatchContainer, UniformBatchCollector


class TrajectoryContainerActorCritic(TrajectoryContainer):
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


class TrajectoryCollectorActorCritic(TrajectoryCollector):
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground, discounted: bool = True):
        super().__init__(experiment_spec, playground, discounted)

        self.V_estimates = []

    def __call__(self, observation: np.ndarray, action, reward: float, V_estimate: float) -> None:
        TrajectoryCollector.__call__(observation, action, reward)



