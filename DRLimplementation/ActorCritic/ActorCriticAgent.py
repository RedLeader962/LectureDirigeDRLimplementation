# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import numpy as np

from ActorCritic.ActorCriticBrain import actor_policy, critic
from blocAndTools import buildingbloc as bloc
from blocAndTools.agent import Agent
from blocAndTools.buildingbloc import ExperimentSpec
from blocAndTools.visualisationtools import ConsolPrintLearningStats
from blocAndTools.samplecontainer import TrajectoryCollector, UniformBatchCollector, UniformeBatchContainer
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1   # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()
# endregion

class ActorCriticAgent(Agent):
    def _use_hardcoded_agent_root_directory(self):
        self.agent_root_dir = 'ActorCritic'
        return None

    def _build_computation_graph(self):
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                               Actor network                                               *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        """ ---- Placeholder ---- """
        observation_ph, action_ph, Advantage_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
            self.playground, None, obs_shape_constraint=None)
        self.observation_ph = observation_ph
        self.action_ph = action_ph
        self.Q_values_ph = Advantage_ph

        """ ---- The policy and is neural net theta ---- """
        actor_network = actor_policy(observation_ph, action_ph, Advantage_ph, self.exp_spec, self.playground)
        (actor_policy_action_sampler, theta_mlp, actor_pseudo_loss) = actor_network
        self.policy_action_sampler = actor_policy_action_sampler
        self.theta_mlp = theta_mlp
        self.pseudo_loss = actor_pseudo_loss

        """ ---- Optimizer ---- """
        self.policy_optimizer_op = bloc.policy_optimizer(self.pseudo_loss, self.exp_spec.learning_rate)

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                               Critic Network                                              *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # todo:implement --> build Critic network:

        pass

    def _instantiate_data_collector(self):
        """
        Data collector utility

        :return: Collertor utility
        :rtype: (TrajectoryCollector, UniformBatchCollector)
        """

        the_TRAJECTORY_COLLECTOR = TrajectoryCollector(self.exp_spec, self.playground)
        the_UNI_BATCH_COLLECTOR = UniformBatchCollector(self.exp_spec.batch_size_in_ts)
        return the_TRAJECTORY_COLLECTOR, the_UNI_BATCH_COLLECTOR

    def _training_epoch_generator(self, consol_print_learning_stats, render_env):
        pass

