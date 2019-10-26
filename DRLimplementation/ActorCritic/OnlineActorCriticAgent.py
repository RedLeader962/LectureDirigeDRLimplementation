# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import numpy as np
from typing import List, Tuple, Any

from ActorCritic.ActorCriticBrain import build_actor_policy_graph, build_critic_graph, critic_train, actor_train
from ActorCritic.ActorCriticBrainSharedNetwork import build_actor_critic_shared_graph
from blocAndTools.agent import Agent
from blocAndTools.rl_vocabulary import rl_name, TargetType, NetworkType
from blocAndTools import buildingbloc as bloc, ConsolPrintLearningStats
# from blocAndTools.container.samplecontainer import TrajectoryCollector, UniformBatchCollector
from blocAndTools.container.samplecontainer_online_mini_batch_OAnORV import (TrajectoryContainerMiniBatchOnlineOAnORV,
                                                                             TrajectoryCollectorMiniBatchOnlineOAnORV,
                                                                             UnconstrainedExperimentStageContainerOnlineAAC,
                                                                             ExperimentStageCollectorOnlineAAC)
from blocAndTools.temporal_difference_computation import (computhe_the_Advantage, compute_TD_target,
                                                          get_t_and_tPrime_array_view_for_element_wise_op, )

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

        """ ---- Placeholder ---- """
        self.observation_ph, self.action_ph, self.Qvalues_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
            self.playground, obs_shape_constraint=None, action_shape_constraint=None, Q_name=vocab.Qvalues_ph)

        if self.exp_spec['Network'] is NetworkType.Split:
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # *                                                                                                       *
            # *                                         Critic computation graph                                      *
            # *                                              (Split network)                                          *
            # *                                                                                                       *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            self.V_phi_estimator = build_critic_graph(self.observation_ph, self.exp_spec)

            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # *                                                                                                       *
            # *                                         Actor computation graph                                       *
            # *                                             (Split network)                                           *
            # *                                                                                                       *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            self.policy_action_sampler, log_pi, _ = build_actor_policy_graph(self.observation_ph, self.exp_spec,
                                                                             self.playground)

            print(":: SPLIT network constructed")

        elif self.exp_spec['Network'] is NetworkType.Shared:
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # *                                                                                                       *
            # *                                   Shared Actor-Critic computation graph                               *
            # *                                                                                                       *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            self.policy_action_sampler, log_pi, _, self.V_phi_estimator = build_actor_critic_shared_graph(
                self.observation_ph, self.exp_spec, self.playground)

            print(":: SHARED network constructed")



        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                                 Advantage                                                 *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # # \\\\\\    My bloc    \\\\\\
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
        self.actor_loss, self.actor_policy_optimizer = actor_train(self.action_ph, log_pi=log_pi, advantage=Advantage,
                                                                   experiment_spec=self.exp_spec,
                                                                   playground=self.playground)

        self.V_phi_loss, self.V_phi_optimizer = critic_train(Advantage, self.exp_spec)

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *
        # *                                                                                                            *
        # *                                                 Summary ops                                                *
        # *                                                                                                            *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *

        """ ---- Episode summary ---- """
        tf_cv1.summary.scalar('Actor_loss', self.actor_loss, family=vocab.loss)
        tf_cv1.summary.scalar('Critic_loss', self.V_phi_loss, family=vocab.loss)

        self.Summary_batch_avg_trjs_return_ph = tf_cv1.placeholder(tf.float32, name='Summary_batch_avg_trjs_return_ph')
        tf_cv1.summary.scalar('Batch average return', self.Summary_batch_avg_trjs_return_ph, family=vocab.G)

        self.summary_epoch_op = tf_cv1.summary.merge_all()

        """ ---- Trajectory summary ---- """
        self.Summary_trj_return_ph = tf_cv1.placeholder(tf.float32, name='Summary_trj_return_ph')
        self.summary_trj_op = tf_cv1.summary.scalar('Trajectory return', self.Summary_trj_return_ph,
                                                    family=vocab.G)

        return None

    def _instantiate_data_collector(self) -> Tuple[TrajectoryCollectorMiniBatchOnlineOAnORV, ExperimentStageCollectorOnlineAAC]:
        """
        Data collector utility

        :return: Collertor utility
        :rtype: (TrajectoryCollectorBatchOARV, UniformBatchCollectorBatchOARV)
        """
        trjCOLLECTOR = TrajectoryCollectorMiniBatchOnlineOAnORV(self.exp_spec, self.playground)
        experimentCOLLECTOR = ExperimentStageCollectorOnlineAAC(self.exp_spec.batch_size_in_ts)
        return trjCOLLECTOR, experimentCOLLECTOR

    def _training_epoch_generator(self, consol_print_learning_stats: ConsolPrintLearningStats, render_env: bool):
        """
        Training epoch generator

        :param consol_print_learning_stats:
        :type consol_print_learning_stats:
        :param render_env:
        :type render_env: bool
        :yield: (epoch, epoch_loss, batch_average_trjs_return, batch_average_trjs_lenght)
        """

        trjCOLLECTOR, experimentCOLLECTOR = self._instantiate_data_collector()

        """ ---- Warm-up the computation graph and start learning! ---- """
        tf_cv1.random.set_random_seed(self.exp_spec.random_seed)
        np.random.seed(self.exp_spec.random_seed)
        with tf_cv1.Session() as sess:
            sess.run(tf_cv1.global_variables_initializer())  # initialize random variable in the computation graph

            consol_print_learning_stats.start_the_crazy_experiment()
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # *                                                                                                       *
            # *                                             Training loop                                             *
            # *                                                                                                       *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

            """ ---- Simulator: Epochs ---- """
            global_step_i = 0
            for epoch in range(self.exp_spec.max_epoch):
                consol_print_learning_stats.next_glorious_epoch()

                """ ---- Simulator: trajectories ---- """
                while experimentCOLLECTOR.is_not_full():                         # <-- (!) BATCH collector control over sampling from batch capacity
                    obs_t = self.playground.env.reset()
                    consol_print_learning_stats.next_glorious_trajectory()

                    """ ---- Simulator: time-steps ---- """
                    while True:
                        global_step_i += 1
                        self._render_trajectory_on_condition(epoch, render_env,
                                                             experimentCOLLECTOR.trj_collected_so_far())

                        """ ---- Run Graph computation ---- """
                        obs_t_flat = bloc.format_single_step_observation(obs_t)
                        action, V_t = sess.run([self.policy_action_sampler, self.V_phi_estimator],
                                               feed_dict={self.observation_ph: obs_t_flat})

                        """ ---- Agent: act in the environment ---- """
                        action = bloc.to_scalar(action)
                        obs_tPrime, reward, done, _ = self.playground.env.step(action)

                        """ ---- Agent: Collect current timestep events ---- """
                        trjCOLLECTOR.collect_OAnORV(obs_t=obs_t, act_t=action, obs_tPrime=obs_tPrime,
                                                    rew_t=reward, V_estimate=V_t)                                    # <-- (!) TRJ collector control

                        obs_t = obs_tPrime

                        # (CRITICAL) todo:finish --> minibatch TRAINING & UPDATE V_phi
                        if trjCOLLECTOR.minibatch_is_full():
                            self.train_on_minibatch(trjCOLLECTOR, sess, global_step_i, consol_print_learning_stats)

                        if done:
                            """ ---- Simulator: trajectory as ended ---- """
                            trj_return = trjCOLLECTOR.trajectory_ended()

                            trj_summary = sess.run(self.summary_trj_op, {self.Summary_trj_return_ph: trj_return})
                            self.writer.add_summary(trj_summary, global_step=global_step_i)

                            self.train_on_minibatch(trjCOLLECTOR, sess, global_step_i, consol_print_learning_stats)

                            """ ---- Agent: Collect the sampled trajectory  ---- """
                            trj_container = trjCOLLECTOR.pop_trajectory_and_reset()
                            experimentCOLLECTOR.collect(trj_container)                                             # <-- (!) BATCH collector control

                            consol_print_learning_stats.trajectory_training_stat(
                                the_trajectory_return=trj_return, timestep=len(trj_container))                     # <-- (!) TRJ container ACCESS
                            break

                """ ---- Simulator: epoch as ended, it's time to learn! ---- """
                batch_trj_collected = experimentCOLLECTOR.trj_collected_so_far()
                batch_timestep_collected = experimentCOLLECTOR.timestep_collected_so_far()

                # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *
                # *                                                                                                  *
                # *                                    Update policy_theta                                           *
                # *                                                                                                  *
                # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *

                """ ---- Prepare data for backpropagation in the neural net ---- """
                experiment_container = experimentCOLLECTOR.pop_batch_and_reset()                                     # <-- (!) BATCH collector control
                batch_average_trjs_return, batch_average_trjs_lenght = experiment_container.get_basic_metric()       # <-- (!) BATCH container ACCESS

                # batch_observations = experiment_container.batch_observations                                 # BATCH container ACCESS  # =Muted=
                # batch_actions = experiment_container.batch_actions                                           # BATCH container ACCESS  # =Muted=
                # batch_Qvalues = experiment_container.batch_Qvalues                                           # BATCH container ACCESS  # =Muted=

                # self._data_shape_is_compatibility_with_graph(batch_Qvalues, batch_actions, batch_observations) # =Muted=

                epoch_summary = sess.run(self.summary_epoch_op,
                                         feed_dict={self.Summary_batch_avg_trjs_return_ph: batch_average_trjs_return})

                self.writer.add_summary(epoch_summary, global_step=global_step_i)

                # todo:refactor --> move :
                consol_print_learning_stats.epoch_training_stat(
                    epoch_loss=0,                                                                    # <-- (!)epoch stats
                    epoch_average_trjs_return=batch_average_trjs_return,
                    epoch_average_trjs_lenght=batch_average_trjs_lenght,
                    number_of_trj_collected=batch_trj_collected,
                    total_timestep_collected=batch_timestep_collected
                    )

                self._save_learned_model(batch_average_trjs_return, epoch, sess)

                """ ---- Expose current epoch computed information for integration test ---- """
                yield (epoch, e_actor_loss, batch_average_trjs_return, batch_average_trjs_lenght)

        return None

    def train_on_minibatch(self, trjCOLLECTOR, sess, global_step_i, consol_print_learning_stats):
        trjCOLLECTOR.compute_Qvalues_as_BootstrapEstimate()  # <-- (!) TRJ collector COMPUTE
        minibatch = trjCOLLECTOR.get_minibatch()

        """ ---- Agent: Compute gradient & update policy ---- """
        epoch_feed_dictionary = bloc.build_feed_dictionary([self.observation_ph, self.action_ph, self.Qvalues_ph],
                                                           [minibatch.obs_t, minibatch.act_t, minibatch.q_values_t])

        e_actor_loss, e_V_phi_loss, epoch_summary = sess.run([self.actor_loss, self.V_phi_loss, self.summary_epoch_op],
                                                             feed_dict=epoch_feed_dictionary)

        self.writer.add_summary(epoch_summary, global_step=global_step_i)

        """ ---- Train actor ---- """
        sess.run(self.actor_policy_optimizer, feed_dict=epoch_feed_dictionary)

        critic_feed_dictionary = bloc.build_feed_dictionary(
            [self.observation_ph, self.Qvalues_ph],
            [minibatch.obs_t, minibatch.q_values_t])

        """ ---- Train critic ---- """
        for c_loop in range(self.exp_spec['critique_loop_len']):
            consol_print_learning_stats.track_progress(progress=c_loop, message="Critic training")
            sess.run(self.V_phi_optimizer, feed_dict=critic_feed_dictionary)
