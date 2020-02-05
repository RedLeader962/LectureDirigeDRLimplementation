# coding=utf-8

from blocAndTools.rl_vocabulary import rl_name
from blocAndTools import buildingbloc as bloc
from blocAndTools.agent import Agent
from blocAndTools.buildingbloc import ExperimentSpec
from blocAndTools.visualisationtools import ConsolPrintLearningStats
from blocAndTools.container.trajectories_pool import PoolManager
from blocAndTools.container.FAST_trajectories_pool import Fast_PoolManager
from blocAndTools.container.samplecontainer import (
    TrajectoryContainer, TrajectoryCollector, UniformBatchCollector,
    UniformeBatchContainer,
    )
from blocAndTools.container.samplecontainer_batch_OARV import TrajectoryContainerBatchOARV, TrajectoryCollectorBatchOARV, UniformeBatchContainerBatchOARV, UniformBatchCollectorBatchOARV
from blocAndTools.temporal_difference_computation import computhe_the_Advantage, compute_TD_target, get_t_and_tPrime_array_view_for_element_wise_op
