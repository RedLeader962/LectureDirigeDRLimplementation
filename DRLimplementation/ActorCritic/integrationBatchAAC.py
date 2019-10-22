# coding=utf-8
"""
Integration test based on 'Actro-Critic' implementation from Lil-log at:
    https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#actor-critic

Use in conjunction with: ../tests/test_Z_integration/... .py

Code from referenced implementation are marked with ////// Original bloc ////// on line end or at code bloc begining
"""


from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import numpy as np
from typing import List, Tuple, Any

from ActorCritic.ActorCriticBrain import build_actor_policy_graph, build_critic_graph
from blocAndTools.agent import Agent
from blocAndTools.rl_vocabulary import rl_name
from blocAndTools import buildingbloc as bloc, ConsolPrintLearningStats
# from blocAndTools.container.samplecontainer import TrajectoryCollector, UniformBatchCollector
from blocAndTools.container.samplecontainerbatchactorcritic import (TrajectoryContainerBatchActorCritic,
                                                                    TrajectoryCollectorBatchActorCritic,
                                                                    UniformeBatchContainerBatchActorCritic,
                                                                    UniformBatchCollectorBatchActorCritic, )
from blocAndTools.temporal_difference_computation import (computhe_the_Advantage, compute_TD_target,
                                                          get_t_and_tPrime_array_view_for_element_wise_op, )

tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()
# endregion



class IntegrationActorCriticAgent(Agent):
    def _use_hardcoded_agent_root_directory(self):
        self.agent_root_dir = 'ActorCritic'
        return None

    def _build_computation_graph(self):
        """
        Build the Policy_theta & V_phi computation graph with theta and phi as multi-layer perceptron

        """

        """ ---- Placeholder ---- """
        # \\\\\\    My bloc    \\\\\\
        self.observation_ph, self.action_ph, self.target_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
            self.playground, obs_shape_constraint=None, action_shape_constraint=None, Q_name=vocab.target_ph)

        # \\\\\\    My bloc    \\\\\\
        # self.Advantage_ph = tf_cv1.placeholder(tf.float32, shape=self.target_ph.shape, name=vocab.advantage_ph)
        # # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # # *                                                                                                           *
        # # *                                         Actor computation graph                                           *
        # # *                                                                                                           *
        # # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # actor_graph = build_actor_policy_graph(self.observation_ph, self.action_ph, self.Advantage_ph,
        #                                        self.exp_spec, self.playground)
        # self.policy_action_sampler, _, self.actor_loss, self.actor_policy_optimizer = actor_graph
        # # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # # *                                                                                                           *
        # # *                                         Critic computation graph                                          *
        # # *                                                                                                           *
        # # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # """ ---- The value function estimator ---- """
        # self.V_phi_estimator, self.V_phi_loss, self.V_phi_optimizer = build_critic_graph(self.observation_ph,
        #                                                                                  self.target_ph, self.exp_spec)

        # ////// Original bloc //////
        def dense_nn(inputs, layers_sizes, name):
            """Creates a densely connected multi-layer neural network.
            inputs: the input tensor
            layers_sizes (list<int>): defines the number of units in each layer. The output
                layer has the size layers_sizes[-1].
            """
            with tf_cv1.variable_scope(name):
                for i, size in enumerate(layers_sizes):
                    inputs = tf.layers.dense(
                        inputs,
                        size,
                        # Add relu activation only for internal layers.
                        activation=tf.nn.relu if i < len(layers_sizes) - 1 else None,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name=name + '_l' + str(i)
                        )
            return inputs

        # ////// Original bloc //////
        # Inputs
        # states = tf.placeholder(tf.float32, shape=(None, observation_size), name='state')
        # actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        # td_targets = tf.placeholder(tf.float32, shape=(None,), name='td_target')

        # note: Design architecture
        #   |    collect:
        #    |      - Target: MonteCarlo or Bootstrap

        # Actor: action probabilities
        actor = dense_nn(self.observation_ph, [32, 32, self.playground.env.action_space.n],                     # ////// Original bloc //////
                         name=vocab.theta_NeuralNet)

        self.policy_action_sampler = tf.squeeze(tf.multinomial(actor, 1))                                       # ////// Original bloc //////


        # Critic: action value (Q-value)
        critic = dense_nn(self.observation_ph, [32, 32, 1],                                                     # ////// Original bloc //////
                          name=vocab.phi_NeuralNet)

        action_ohe = tf.one_hot(self.action_ph, self.playground.ACTION_CHOICES, 1.0, 0.0, name='action_one_hot')# ////// Original bloc //////

        V_estimate = tf.reduce_sum(critic * action_ohe, reduction_indices=-1, name='q_acted')                   # ////// Original bloc //////
        flatten_V_estimate = tf.reshape(V_estimate, [-1])
        Advantage = self.target_ph - flatten_V_estimate  # ////// Original bloc //////

        # ////// Original bloc //////
        with tf_cv1.variable_scope(vocab.actor_network):
            self.actor_loss = tf.reduce_mean(
                tf.stop_gradient(Advantage) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=actor, labels=self.action_ph),
                name='loss_actor')
            self.actor_policy_optimizer = tf_cv1.train.AdamOptimizer(0.01).minimize(self.actor_loss)

        # \\\\\\    My bloc    \\\\\\
        tf_cv1.summary.scalar('Actor_loss', self.actor_loss, family=vocab.loss)                                          # =HL=

        # ////// Original bloc //////
        with tf_cv1.variable_scope(vocab.critic_network):
            self.V_phi_loss = tf.reduce_mean(tf.square(Advantage))
            self.V_phi_optimizer = tf_cv1.train.AdamOptimizer(0.01).minimize(self.V_phi_loss)

        # \\\\\\    My bloc    \\\\\\
        tf_cv1.summary.scalar('Critic_loss', self.V_phi_loss, family=vocab.loss)                                         # =HL=

        # train_ops = [optim_c, optim_a]

        # \\\\\\    My bloc    \\\\\\
        """ ---- Episode summary ---- """
        # those intantiated in graph will be agregate to
        self.Summary_batch_avg_trjs_return_ph = tf_cv1.placeholder(tf.float32, name='Summary_batch_avg_trjs_return_ph')  # =HL=
        tf_cv1.summary.scalar('Batch average return', self.Summary_batch_avg_trjs_return_ph, family=vocab.G)             # =HL=
        self.summary_op = tf_cv1.summary.merge_all()                                                                     # =HL=

        """ ---- Trajectory summary ---- """
        self.Summary_trj_return_ph = tf_cv1.placeholder(tf.float32, name='Summary_trj_return_ph')                        # =HL=
        self.summary_trj_op = tf_cv1.summary.scalar('Trajectory return', self.Summary_trj_return_ph, family=vocab.G)     # =HL=
        return None

    def _instantiate_data_collector(self) -> Tuple[
        TrajectoryCollectorBatchActorCritic, UniformBatchCollectorBatchActorCritic]:
        """
        Data collector utility

        :return: Collertor utility
        :rtype: (TrajectoryCollectorBatchActorCritic, UniformBatchCollectorBatchActorCritic)
        """
        the_TRAJECTORY_COLLECTOR = TrajectoryCollectorBatchActorCritic(
            self.exp_spec, self.playground, MonteCarloTarget=self.exp_spec['MonteCarloTarget'])
        the_UNI_BATCH_COLLECTOR = UniformBatchCollectorBatchActorCritic(self.exp_spec.batch_size_in_ts)
        return the_TRAJECTORY_COLLECTOR, the_UNI_BATCH_COLLECTOR

    # todo:implement --> critic training variation: target y = Monte Carlo target
    # todo:implement --> critic training variation: target y = Bootstrap estimate target
    def _training_epoch_generator(self, consol_print_learning_stats: ConsolPrintLearningStats, render_env: bool):
        """
        Training epoch generator

        :param consol_print_learning_stats:
        :type consol_print_learning_stats:
        :param render_env:
        :type render_env: bool
        :yield: (epoch, epoch_loss, batch_average_trjs_return, batch_average_trjs_lenght)
        """

        the_TRAJECTORY_COLLECTOR, the_UNI_BATCH_COLLECTOR = self._instantiate_data_collector()

        """ ---- Warm-up the computation graph and start learning! ---- """
        tf_cv1.set_random_seed(self.exp_spec.random_seed)
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
                while the_UNI_BATCH_COLLECTOR.is_not_full():
                    obs_t = self.playground.env.reset()  # <-- fetch initial observation
                    consol_print_learning_stats.next_glorious_trajectory()

                    """ ---- Simulator: time-steps ---- """
                    while True:
                        global_step_i += 1
                        self._render_trajectory_on_condition(epoch, render_env,
                                                             the_UNI_BATCH_COLLECTOR.trj_collected_so_far())

                        """ ---- Agent: act in the environment ---- """
                        obs_t_flat = bloc.format_single_step_observation(obs_t)
                        action_array = sess.run(self.policy_action_sampler,
                                                            feed_dict={self.observation_ph: obs_t_flat})

                        action = bloc.to_scalar(action_array)

                        obs_tPrime, reward, done, _ = self.playground.env.step(action)

                        """ ---- Agent: Collect current timestep events ---- """
                        the_TRAJECTORY_COLLECTOR.collect(observation=obs_t,
                                                         action=action,
                                                         reward=reward,
                                                         # V_estimate=bloc.to_scalar(V_estimate)
                                                         )
                        obs_t = obs_tPrime  # <-- (!)

                        if done:
                            """ ---- Simulator: trajectory as ended ---- """
                            trj_return = the_TRAJECTORY_COLLECTOR.trajectory_ended()

                            trj_summary = sess.run(self.summary_trj_op,
                                                   {self.Summary_trj_return_ph: trj_return})    # =HL=
                            self.writer.add_summary(trj_summary, global_step=global_step_i)     # =HL=


                            """ ---- Agent: Collect the sampled trajectory  ---- """
                            trj_container = the_TRAJECTORY_COLLECTOR.pop_trajectory_and_reset()
                            the_UNI_BATCH_COLLECTOR.collect(trj_container)

                            consol_print_learning_stats.trajectory_training_stat(
                                the_trajectory_return=trj_return, timestep=len(trj_container))
                            break

                """ ---- Simulator: epoch as ended, it's time to learn! ---- """
                batch_trj_collected = the_UNI_BATCH_COLLECTOR.trj_collected_so_far()
                batch_timestep_collected = the_UNI_BATCH_COLLECTOR.timestep_collected_so_far()

                # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *
                # *                                                                                                  *
                # *                                    Update policy_theta                                           *
                # *                                                                                                  *
                # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *

                """ ---- Prepare data for backpropagation in the neural net ---- """
                batch_container: UniformeBatchContainerBatchActorCritic = the_UNI_BATCH_COLLECTOR.pop_batch_and_reset()
                batch_average_trjs_return, batch_average_trjs_lenght = batch_container.compute_metric()

                batch_observations = batch_container.batch_observations
                batch_actions = batch_container.batch_actions
                batch_target_values = batch_container.batch_Qvalues
                batch_Advantages = batch_container.batch_Advantages

                # self._data_shape_is_compatibility_with_graph(batch_target_values, batch_actions, batch_observations)

                """ ---- Agent: Compute gradient & update policy ---- """
                feed_dictionary = bloc.build_feed_dictionary(
                    [self.observation_ph, self.action_ph, self.target_ph, self.Summary_batch_avg_trjs_return_ph],       # =HL=
                    [batch_observations, batch_actions, batch_target_values, batch_average_trjs_return])                # =HL=

                e_actor_loss, e_V_phi_loss, summary = sess.run([self.actor_loss, self.V_phi_loss, self.summary_op],
                                                      feed_dict=feed_dictionary)

                self.writer.add_summary(summary, global_step=global_step_i)                                             # =HL=

                """ ---- Train actor ---- """
                sess.run(self.actor_policy_optimizer, feed_dict=feed_dictionary)

                critic_feed_dictionary = bloc.build_feed_dictionary(
                    # [self.observation_ph, self.target_ph],                                                            # =HL=
                    # [batch_observations, batch_target_values])                                                        # =HL=
                    [self.observation_ph, self.action_ph, self.target_ph],                                              # =HL=
                    [batch_observations, batch_actions, batch_target_values])                                           # =HL=

                """ ---- Train critic ---- """
                for c_loop in range(self.exp_spec['critique_loop_len']):
                    consol_print_learning_stats.track_progress(progress=c_loop, message="Critic training")
                    sess.run(self.V_phi_optimizer, feed_dict=critic_feed_dictionary)

                consol_print_learning_stats.epoch_training_stat(
                    epoch_loss=e_actor_loss,
                    epoch_average_trjs_return=batch_average_trjs_return,
                    epoch_average_trjs_lenght=batch_average_trjs_lenght,
                    number_of_trj_collected=batch_trj_collected,
                    total_timestep_collected=batch_timestep_collected
                    )

                """ ---- Save learned model ---- """
                if batch_average_trjs_return == 200:
                    self._save_checkpoint(epoch, sess, 'REINFORCE')

                """ ---- Expose current epoch computed information for integration test ---- """
                yield (epoch, e_actor_loss, batch_average_trjs_return, batch_average_trjs_lenght)

        return None
