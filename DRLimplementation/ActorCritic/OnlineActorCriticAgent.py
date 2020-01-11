# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Tuple

import numpy as np
# region ::Import statement ...
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import blocAndTools.tensorflowbloc
from ActorCritic.ActorCriticBrainSharedNetwork import build_actor_critic_shared_graph, actor_shared_train, critic_shared_train
from ActorCritic.ActorCriticBrainSplitNetwork import (build_actor_policy_graph, build_critic_graph, critic_train,
                                                      actor_train, )
from blocAndTools import buildingbloc as bloc, ConsolPrintLearningStats
from blocAndTools.agent import Agent
from blocAndTools.container.samplecontainer_online_mini_batch_OAnORV import (TrajectoryCollectorMiniBatchOnlineOAnORV,
                                                                             ExperimentStageCollectorOnlineAAC, )
from blocAndTools.rl_vocabulary import rl_name, NetworkType

tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()
# endregion

class OnlineActorCriticAgent(Agent):
    def _use_hardcoded_agent_root_directory(self):
        self.agent_root_dir = 'ActorCritic'
        return None

    def _build_computation_graph(self):
        """
        Build the Policy_theta & V_phi computation graph with theta and phi as multi-layer perceptron
        """
        assert isinstance(self.exp_spec['Network'], NetworkType), ("exp_spec['Network'] must be explicitely defined "
                                                                   "with a NetworkType enum")

        if self.exp_spec.random_seed == 0:
            print(":: Random seed control is turned OFF")
        else:
            tf_cv1.random.set_random_seed(self.exp_spec.random_seed)
            np.random.seed(self.exp_spec.random_seed)
            print(":: Random seed control is turned ON")

        """ ---- Placeholder ---- """
        self.obs_t_ph, self.action_ph, self.Qvalues_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
            self.playground, obs_shape_constraint=None, action_shape_constraint=None, Q_name=vocab.Qvalues_ph)

        if self.exp_spec['Network'] is NetworkType.Split:
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # *                                                                                                       *
            # *                                         Critic computation graph                                      *
            # *                                              (Split network)                                          *
            # *                                                                                                       *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            self.V_phi_estimator = build_critic_graph(self.obs_t_ph, self.exp_spec)

            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # *                                                                                                       *
            # *                                         Actor computation graph                                       *
            # *                                             (Split network)                                           *
            # *                                                                                                       *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            self.policy_pi, log_pi, _ = build_actor_policy_graph(self.obs_t_ph, self.exp_spec,
                                                                 self.playground)

            print(":: SPLIT network constructed")

        elif self.exp_spec['Network'] is NetworkType.Shared:
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # *                                                                                                       *
            # *                                   Shared Actor-Critic computation graph                               *
            # *                                                                                                       *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            self.policy_pi, log_pi, _, self.V_phi_estimator = build_actor_critic_shared_graph(
                self.obs_t_ph, self.exp_spec, self.playground)

            print(":: SHARED network constructed")

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                                 Advantage                                                 *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # # alternate architecture with element wise computed advantage
        # self.Advantage_ph = tf_cv1.placeholder(tf.float32, shape=self.Qvalues_ph.shape, name=vocab.advantage_ph)

        with tf_cv1.name_scope(vocab.Advantage):
            # (!) note: Advantage computation
            #       |       no squeeze      ==>     SLOWER computation
            #       |               eg: Advantage = self.Qvalues_ph - self.V_phi_estimator
            #       |
            #       |       with squeeze    ==>     RACING CAR FAST computation
            #
            # (Nice to have) todo:investigate?? --> why it's much faster?: hypothese --> broadcasting slowdown computation
            Advantage = self.Qvalues_ph - tf_cv1.squeeze(self.V_phi_estimator)

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                           Actor & Critic Train                                            *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        self.actor_loss, self.actor_policy_optimizer = actor_shared_train(self.action_ph, log_pi=log_pi, advantage=Advantage,
                                                                          experiment_spec=self.exp_spec,
                                                                          playground=self.playground)

        self.V_phi_loss, self.V_phi_optimizer = critic_shared_train(Advantage, self.exp_spec)

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *
        # *                                                                                                            *
        # *                                                 Summary ops                                                *
        # *                                                                                                            *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *

        """ ---- By Epoch summary ---- """
        self.summary_stage_avg_trjs_actor_loss_ph = tf_cv1.placeholder(tf.float32, name='Actor_loss_ph')
        self.summary_stage_avg_trjs_critic_loss_ph = tf_cv1.placeholder(tf.float32, name='Critic_loss_ph')
        tf_cv1.summary.scalar('Actor_loss', self.summary_stage_avg_trjs_actor_loss_ph, family=vocab.loss)
        tf_cv1.summary.scalar('Critic_loss', self.summary_stage_avg_trjs_critic_loss_ph, family=vocab.loss)

        self.summary_stage_avg_trjs_return_ph = tf_cv1.placeholder(tf.float32, name='summary_stage_avg_trjs_return_ph')
        tf_cv1.summary.scalar('Batch average return', self.summary_stage_avg_trjs_return_ph, family=vocab.G)

        self.summary_epoch_op = tf_cv1.summary.merge_all()

        """ ---- By Trajectory summary ---- """
        self.Summary_trj_return_ph = tf_cv1.placeholder(tf.float32, name='Summary_trj_return_ph')
        self.summary_trj_return_op = tf_cv1.summary.scalar('Trajectory return', self.Summary_trj_return_ph,
                                                           family=vocab.G)

        self.Summary_trj_lenght_ph = tf_cv1.placeholder(tf.float32, name='Summary_trj_lenght_ph')
        self.summary_trj_lenght_op = tf_cv1.summary.scalar('Trajectory lenght', self.Summary_trj_lenght_ph,
                                                           family=vocab.Trajectory_lenght)

        self.summary_trj_op = tf_cv1.summary.merge([self.summary_trj_return_op, self.summary_trj_lenght_op])


        return None

    def _instantiate_data_collector(self) -> Tuple[TrajectoryCollectorMiniBatchOnlineOAnORV,
                                                   ExperimentStageCollectorOnlineAAC]:
        """
        Data collector utility

        :return: Collertor utility
        :rtype: (TrajectoryCollectorBatchOARV, UniformBatchCollectorBatchOARV)
        """
        trjCOLLECTOR = TrajectoryCollectorMiniBatchOnlineOAnORV(self.exp_spec, self.playground,
                                                                discounted=self.exp_spec.discounted_reward_to_go,
                                                                mini_batch_capacity=self.exp_spec.batch_size_in_ts)
        experimentCOLLECTOR = ExperimentStageCollectorOnlineAAC(self.exp_spec['stage_size_in_trj'])
        return trjCOLLECTOR, experimentCOLLECTOR

    def _training_epoch_generator(self, consol_print_learning_stats: ConsolPrintLearningStats, render_env: bool):
        """
        Training epoch generator

        :param consol_print_learning_stats:
        :type consol_print_learning_stats:
        :param render_env:
        :type render_env: bool
        :yield: (epoch, epoch_loss, stage_average_trjs_return, stage_average_trjs_lenght)
        """

        self.trjCOLLECTOR, experimentCOLLECTOR = self._instantiate_data_collector()

        print(":: ONline ActorCritic agent reporting for training ")

        """ ---- Warm-up the computation graph and start learning! ---- """
        with tf_cv1.Session() as sess:
            self.sess = sess
            self.sess.run(tf_cv1.global_variables_initializer())  # initialize random variable in the computation graph

            consol_print_learning_stats.start_the_crazy_experiment()
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # *                                                                                                       *
            # *                                             Training loop                                             *
            # *                                                                                                       *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

            """ ---- Simulator: Epochs ---- """
            global_timestep_idx = 0
            for epoch in range(self.exp_spec.max_epoch):
                # (Ice-Boxed) todo:implement --> finish lr sheduler for online shared algo:
                # (Ice-Boxed) todo:implement --> add 'global_timestep_max' to hparam:
                # if global_timestep_idx >= self.exp_spec['global_timestep_max']:
                #     break

                consol_print_learning_stats.next_glorious_epoch()

                """ ---- Simulator: trajectories ---- """
                while experimentCOLLECTOR.is_not_full():
                    obs_t = self.playground.env.reset()
                    consol_print_learning_stats.next_glorious_trajectory()

                    """ ---- Simulator: time-steps ---- """
                    local_step_t = 0
                    while True:
                        global_timestep_idx += 1
                        local_step_t += 1
                        self._render_trajectory_on_condition(epoch, render_env,
                                                             experimentCOLLECTOR.trj_collected_so_far())
    
                        """ ---- Run Graph computation ---- """
                        obs_t_flat = bloc.format_single_step_observation(obs_t)
                        action, V_t = self.sess.run([self.policy_pi, self.V_phi_estimator],
                                                    feed_dict={self.obs_t_ph: obs_t_flat})
    
                        action = blocAndTools.tensorflowbloc.to_scalar(action)
                        V_t = blocAndTools.tensorflowbloc.to_scalar(V_t)
    
                        """ ---- Agent: act in the environment ---- """
                        obs_tPrime, reward, done, _ = self.playground.env.step(action)
    
                        """ ---- Agent: Collect current timestep events ---- """
                        self.trjCOLLECTOR.collect_OAnORV(obs_t=obs_t, act_t=action, obs_tPrime=obs_tPrime,
                                                         rew_t=reward, V_estimate=V_t)
    
                        obs_t = obs_tPrime

                        if done:
                            """ ---- Simulator: trajectory as ended ---- """
                            trj_return = self.trjCOLLECTOR.trajectory_ended()
                            self._train_on_minibatch(consol_print_learning_stats, local_step_t)


                            """ ---- Agent: Collect the sampled trajectory  ---- """
                            trj_container = self.trjCOLLECTOR.pop_trajectory_and_reset()
                            experimentCOLLECTOR.collect(trj_container)

                            # trj_summary = self.sess.run(self.summary_trj_return_op, {self.Summary_trj_return_ph: trj_return})
                            trj_len = len(trj_container)

                            trj_summary = sess.run(self.summary_trj_op,
                                                   {self.Summary_trj_return_ph: trj_return,
                                                    self.Summary_trj_lenght_ph: trj_len
                                                    })

                            self.writer.add_summary(trj_summary, global_step=global_timestep_idx)

                            consol_print_learning_stats.trajectory_training_stat(the_trajectory_return=trj_return,
                                                                                 timestep=trj_len)
                            break

                        elif self.trjCOLLECTOR.minibatch_is_full():
                            self._train_on_minibatch(consol_print_learning_stats, local_step_t)

                """ ---- Simulator: epoch as ended, it's time to learn! ---- """
                stage_trj_collected = experimentCOLLECTOR.trj_collected_so_far()
                stage_timestep_collected = experimentCOLLECTOR.timestep_collected_so_far()

                """ ---- Prepare data for backpropagation in the neural net ---- """
                experiment_container = experimentCOLLECTOR.pop_batch_and_reset()
                stage_average_trjs_return, stage_average_trjs_lenght = experiment_container.get_basic_metric()
                stage_actor_mean_loss, stage_critic_mean_loss = experiment_container.get_stage_mean_loss()

                epoch_feed_dictionary = blocAndTools.tensorflowbloc.build_feed_dictionary(
                    [self.summary_stage_avg_trjs_actor_loss_ph,
                     self.summary_stage_avg_trjs_critic_loss_ph,
                     self.summary_stage_avg_trjs_return_ph],
                    [stage_actor_mean_loss,
                     stage_critic_mean_loss,
                     stage_average_trjs_return])

                epoch_summary = self.sess.run(self.summary_epoch_op, feed_dict=epoch_feed_dictionary)

                self.writer.add_summary(epoch_summary, global_step=global_timestep_idx)

                consol_print_learning_stats.epoch_training_stat(
                    epoch_loss=stage_actor_mean_loss,
                    epoch_average_trjs_return=stage_average_trjs_return,
                    epoch_average_trjs_lenght=stage_average_trjs_lenght,
                    number_of_trj_collected=stage_trj_collected,
                    total_timestep_collected=stage_timestep_collected)

                self._save_learned_model(stage_average_trjs_return, epoch, self.sess)

                """ ---- Expose current epoch computed information for integration test ---- """
                yield (epoch, stage_actor_mean_loss, stage_average_trjs_return, stage_average_trjs_lenght)

            print("\n\n\n:: Global timestep collected: {}".format(global_timestep_idx), end="")
        return None

    def _train_on_minibatch(self, consol_print_learning_stats, local_step_t):
        self.trjCOLLECTOR.compute_Qvalues_as_BootstrapEstimate()
        minibatch = self.trjCOLLECTOR.get_minibatch()
    
        minibatch_feed_dictionary = blocAndTools.tensorflowbloc.build_feed_dictionary(
            [self.obs_t_ph, self.action_ph, self.Qvalues_ph],
            [minibatch.obs_t, minibatch.act_t, minibatch.q_values_t])
    
        """ ---- Compute metric and collect ---- """
        minibatch_actor_loss, minibatch_V_loss = self.sess.run([self.actor_loss, self.V_phi_loss],
                                                               feed_dict=minibatch_feed_dictionary)
    
        self.trjCOLLECTOR.collect_loss(actor_loss=minibatch_actor_loss, critic_loss=minibatch_V_loss)
    
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *
        # *                                                                                                  *
        # *                    Update policy_theta & critic V_phi over the minibatch                         *
        # *                                                                                                  *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *
        # consol_print_learning_stats.track_progress(progress=local_step_t, message="Agent training")

        """ ---- Train actor ---- """
        self.sess.run(self.actor_policy_optimizer, feed_dict=minibatch_feed_dictionary)

        """ ---- Train critic ---- """
        critic_feed_dictionary = blocAndTools.tensorflowbloc.build_feed_dictionary(
            [self.obs_t_ph, self.Qvalues_ph],
            [minibatch.obs_t, minibatch.q_values_t])

        for c_loop in range(self.exp_spec['critique_loop_len']):      # <-- (!) most likely 1 iteration
            self.sess.run(self.V_phi_optimizer, feed_dict=critic_feed_dictionary)

        return None

