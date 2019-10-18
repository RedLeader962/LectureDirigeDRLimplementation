# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import numpy as np

from ActorCritic.ActorCriticBrain import build_actor_policy_graph, build_critic_graph
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
        """
        Build the Policy_theta & V_phi computation graph with theta and phi as multi-layer perceptron

        """

        """ ---- Placeholder ---- """
        self.observation_ph, self.action_ph, self.Advantage_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
            self.playground, obs_shape_constraint=None, action_shape_constraint=None)

        self.target_placeholder = tf_cv1.placeholder(tf.float32, shape=self.Advantage_ph.shape, name='target_placeholder')

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                         Actor computation graph                                           *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        """ ---- The policy and is neural net theta ---- """
        actor_graph = build_actor_policy_graph(self.observation_ph, self.action_ph, self.Advantage_ph,
                                                 self.exp_spec, self.playground)
        (self.policy_action_sampler, self.theta_mlp, self.actor_loss) = actor_graph

        """ ---- Actor optimizer ---- """
        self.actor_policy_optimizer_op = bloc.policy_optimizer(self.actor_loss, self.exp_spec.learning_rate)

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                         Critic computation graph                                          *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        """ ---- The value function estimator ---- """
        self.V_phi_estimator, self.V_phi_loss = build_critic_graph(self.observation_ph,
                                                                   self.target_placeholder, self.exp_spec)
        """ ---- Critic optimizer ---- """
        self.V_phi_optimizer

        return None

    def _instantiate_data_collector(self):
        """
        Data collector utility

        # (Priority) todo:implement --> collect Value estimate V_t for each timestep:

        :return: Collertor utility
        :rtype: (TrajectoryCollector, UniformBatchCollector)
        """

        the_TRAJECTORY_COLLECTOR = TrajectoryCollector(self.exp_spec, self.playground)
        the_UNI_BATCH_COLLECTOR = UniformBatchCollector(self.exp_spec.batch_size_in_ts)
        return the_TRAJECTORY_COLLECTOR, the_UNI_BATCH_COLLECTOR

    def _compute_the_advantage(self, epoch_batch) -> list:
        """
        AT EPOCH END, Compute the advantage for every timestep of every trajectory collected

                            A_t = r_t + V_tPrime - V_t
         numpy version:
                - compute A for the full trajectory in one shot using element wise operation
                - graph computation on V is done only for timestep t
                - V_tPrime is compute at epoch end
                - require V collected for every timestep + a extra space to appended value:
                    - 0 for Done trajectory
                    - or the actual t+1 for those cut because of force terminaison of max container capacity reached

                Computed element wise:
                    A = rewards[:-1] + V[1:] - V[:-1]

                Note: trick to acces timestep t+1 of the full array
                  |     [:-1] is the collected value at timestep t
                  |     [1:] is the collected value at timestep t+1

        """
        raise NotImplementedError   # todo: implement
        return None

    def _training_epoch_generator(self, consol_print_learning_stats, render_env):
        # todo:implement --> critic training variation: target y = Monte Carlo target
        # todo:implement --> critic training variation: target y = Bootstrap estimate target
        pass

