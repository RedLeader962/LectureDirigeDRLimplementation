# coding=utf-8

# region ::Import statement ...
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import blocAndTools.tensorflowbloc
from SoftActorCritic.SoftActorCriticBrain import (
    actor_train, apply_action_bound, build_critic_graph_q_theta, build_critic_graph_v_psi, build_gaussian_policy_graph,
    critic_learning_rate_scheduler, critic_q_theta_train, critic_v_psi_train, init_frozen_v_psi, update_frozen_v_psi_op,
    )
from blocAndTools import ConsolPrintLearningStats, buildingbloc as bloc
from blocAndTools.agent import Agent
from blocAndTools.buildingbloc import list_representation
from blocAndTools.container.trajectories_pool import PoolManager
from blocAndTools.discrete_time_counter import DiscreteTimestepCounter
from blocAndTools.experiment_clicker import ExperimentClicker
from blocAndTools.logger.basic_trajectory_logger import BasicTrajectoryLogger
from blocAndTools.logger.epoch_metric_logger import EpochMetricLogger
from blocAndTools.rl_vocabulary import rl_name
from blocAndTools.visualisationtools import CycleIndexer

tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()
# endregion


"""

   .|'''.|            .'|.   .       |               .                   '   ..|'''.|          ||    .    ||
   ||..  '    ...   .||.   .||.     |||      ....  .||.    ...   ... ..    .|'     '  ... ..  ...  .||.  ...    ....
    ''|||.  .|  '|.  ||     ||     |  ||   .|   ''  ||   .|  '|.  ||' ''   ||          ||' ''  ||   ||    ||  .|   ''
  .     '|| ||   ||  ||     ||    .''''|.  ||       ||   ||   ||  ||       '|.      .  ||      ||   ||    ||  ||
  |'....|'   '|..|' .||.    '|.' .|.  .||.  '|...'  '|.'  '|..|' .||.       ''|....'  .||.    .||.  '|.' .||.  '|...'

                                                                                 .
                                             ....     ... .   ....  .. ...   .||.
                                            '' .||   || ||  .|...||  ||  ||   ||
                                            .|' ||    |''   ||       ||  ||   ||
                                            '|..'|'  '||||.  '|...' .||. ||.  '|.'
                                                    .|....'



                                                                                                        +--- kban style
"""


class SoftActorCriticAgent(Agent):
    
    def _use_hardcoded_agent_root_directory(self):
        self.agent_root_dir = 'SoftActorCritic'
        return None
    
    def _build_computation_graph(self):
        """ Build the Policy_phi, V_psi and Q_theta computation graph as multi-layer perceptron """

        self._set_random_seed()

        # (nice to have) todo:implement --> add init hook:
        # Note: Second environment for policy evaluation
        self.evaluation_playground = bloc.GymPlayground(environment_name=self.exp_spec.prefered_environment)

        """ ---- Placeholder ---- """
        self.obs_t_ph = bloc.build_observation_placeholder(self.playground, name=vocab.obs_t_ph)
        self.obs_t_prime_ph = bloc.build_observation_placeholder(self.playground, name=vocab.obs_tPrime_ph)
        self.act_ph = bloc.build_action_placeholder(self.playground, name=vocab.act_ph)

        self.reward_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.rew_ph)
        self.trj_done_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.trj_done_ph)

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # /// Actor computation graph //////////////////////////////////////////////////////////////////////////////////
        with tf_cv1.variable_scope(vocab.actor_network):
    
            pi, pi_log_p, self.policy_mu = build_gaussian_policy_graph(self.obs_t_ph, self.exp_spec,
                                                                       self.playground)
    
            self.policy_pi, self.pi_log_likelihood = apply_action_bound(pi, pi_log_p)
    
            """ ---- Adjust policy distribution result to action range  ---- """
            if self.playground.ACTION_SPACE.bounded_above.all():
                self.policy_pi *= self.playground.ACTION_SPACE.high[0]
                self.policy_mu *= self.playground.ACTION_SPACE.high[0]

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # /// Critic computation graph /////////////////////////////////////////////////////////////////////////////////
        with tf_cv1.variable_scope(vocab.critic_network):
            self.V_psi, self.V_psi_frozen = build_critic_graph_v_psi(self.obs_t_ph, self.obs_t_prime_ph, self.exp_spec)

            """ ---- Q_theta {1,2} according to sampled action & according to the reparametrized policy---- """
            self.Q_act_1, self.Q_pi_1 = build_critic_graph_q_theta(self.obs_t_ph, self.act_ph, self.policy_pi,
                                                                   self.exp_spec, name=vocab.Q_theta_1)
            self.Q_act_2, self.Q_pi_2 = build_critic_graph_q_theta(self.obs_t_ph, self.act_ph, self.policy_pi,
                                                                   self.exp_spec, name=vocab.Q_theta_2)

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # /// Actor & Critic Training ops //////////////////////////////////////////////////////////////////////////////
        with tf_cv1.variable_scope(vocab.critic_training):
            critic_lr_schedule, critic_global_grad_step = critic_learning_rate_scheduler(self.exp_spec)

            self.V_psi_loss, self.V_psi_optimizer = critic_v_psi_train(self.V_psi,
                                                                       self.Q_pi_1,
                                                                       self.Q_pi_2,
                                                                       self.pi_log_likelihood,
                                                                       self.exp_spec,
                                                                       critic_lr_schedule,
                                                                       critic_global_grad_step)

            q_theta_train_ops = critic_q_theta_train(self.V_psi_frozen, self.Q_act_1, self.Q_act_2,
                                                     self.reward_t_ph,
                                                     self.trj_done_t_ph, self.exp_spec,
                                                     critic_lr_schedule, critic_global_grad_step)

        self.q_theta_1_loss, self.q_theta_2_loss, self.q_theta_1_optimizer, self.q_theta_2_optimizer = q_theta_train_ops

        with tf_cv1.variable_scope(vocab.policy_training):
            self.actor_kl_loss, self.actor_policy_optimizer_op = actor_train(self.pi_log_likelihood,
                                                                             self.Q_pi_1, self.Q_pi_2,
                                                                             self.exp_spec)

        """ ---- Target nework update: V_psi --> frozen_V_psi ---- """
        with tf_cv1.variable_scope(vocab.target_update):
            self.V_psi_frozen_update_ops = update_frozen_v_psi_op(self.exp_spec['target_smoothing_coefficient'])
            self.init_frozen_v_psi_op = init_frozen_v_psi()

        tr_str = list_representation(tf_cv1.get_collection_ref(tf_cv1.GraphKeys.TRAINABLE_VARIABLES),
                                     ":: TRAINABLE_VARIABLES")
        print(tr_str)

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # /// Summary ops //////////////////////////////////////////////////////////////////////////////////////////////

        # region :: Summary placholders & ops ...
        """ ---- By Epoch summary: RETURNS & LENGHT ---- """
        self.summary_avg_trjs_return_ph = tf_cv1.placeholder(
            tf.float32, name=vocab.summary_ph + 'stoPi_stage_avg_trjs_return_ph')
        tf_cv1.summary.scalar('Epoch_average_trj_return_stochastic_pi)', self.summary_avg_trjs_return_ph,
                              family=vocab.G)

        self.summary_avg_trjs_len_ph = tf_cv1.placeholder(
            tf.float32, name=vocab.summary_ph + 'stoPi_stage_avg_trjs_len_ph')
        tf_cv1.summary.scalar('Epoch_average_trj_lenght_stochastic_pi)', self.summary_avg_trjs_len_ph,
                              family=vocab.Trajectory_lenght)

        self.summary_eval_avg_trjs_return_ph = tf_cv1.placeholder(
            tf.float32, name=vocab.summary_ph + 'detPi_stage_avg_trjs_return_ph')
        tf_cv1.summary.scalar('Epoch_average_trj_return_deterministic_pi)', self.summary_eval_avg_trjs_return_ph,
                              family=vocab.G)

        self.summary_eval_avg_trjs_len_ph = tf_cv1.placeholder(
            tf.float32, name=vocab.summary_ph + 'detPi_stage_avg_trjs_len_ph')
        tf_cv1.summary.scalar('Epoch_average_trj_lenght_deterministic_pi)', self.summary_eval_avg_trjs_len_ph,
                              family=vocab.Trajectory_lenght)

        """ ---- By Epoch summary: LOSS ---- """
        self.summary_avg_trjs_Vloss_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'Critic_V_loss_ph')
        tf_cv1.summary.scalar('critic_v_loss', self.summary_avg_trjs_Vloss_ph, family=vocab.loss)

        self.summary_avg_trjs_Q1loss_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'Critic_Q1_loss_ph')
        tf_cv1.summary.scalar('critic_q_1_loss', self.summary_avg_trjs_Q1loss_ph, family=vocab.loss)

        self.summary_avg_trjs_Q2loss_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'Critic_Q2_loss_ph')
        tf_cv1.summary.scalar('critic_q_2_loss', self.summary_avg_trjs_Q2loss_ph, family=vocab.loss)

        self.summary_avg_trjs_pi_loss_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'policy_loss_ph')
        tf_cv1.summary.scalar('policy_loss', self.summary_avg_trjs_pi_loss_ph, family=vocab.loss)

        """ ---- By Epoch summary: POLICY & VALUE fct ---- """

        self.summary_avg_pi_log_likelihood_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'pi_log_p_ph')
        tf_cv1.summary.scalar('policy_log_likelihood', self.summary_avg_pi_log_likelihood_ph, family=vocab.policy)

        # self.summary_avg_policy_pi_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'policy_pi_ph')
        # tf_cv1.summary.scalar('policy_py', self.summary_avg_policy_pi_ph, family=vocab.policy)
        #
        # self.summary_avg_policy_mu_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'policy_mu_ph')
        # tf_cv1.summary.scalar('policy_mu', self.summary_avg_policy_mu_ph, family=vocab.policy)

        self.summary_avg_V_value_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'V_values_ph')
        tf_cv1.summary.scalar('V_values', self.summary_avg_V_value_ph, family=vocab.values)

        self.summary_avg_frozen_V_value_ph = tf_cv1.placeholder(tf.float32,
                                                                name=vocab.summary_ph + 'frozen_V_values_ph')
        tf_cv1.summary.scalar('frozen_V_values', self.summary_avg_frozen_V_value_ph, family=vocab.values)

        self.summary_avg_Q1_value_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'Q1_values_ph')
        tf_cv1.summary.scalar('Q1_values', self.summary_avg_Q1_value_ph, family=vocab.values)

        self.summary_avg_Q2_value_ph = tf_cv1.placeholder(tf.float32, name=vocab.summary_ph + 'Q2_values_ph')
        tf_cv1.summary.scalar('Q2_values', self.summary_avg_Q2_value_ph, family=vocab.values)

        self.summary_epoch_op = tf_cv1.summary.merge_all()

        """ ---- Distribution summary ---- """
        self.summary_hist_policy_pi = tf_cv1.summary.histogram('policy_py_tensor', self.policy_pi, family=vocab.policy)

        """ ---- By Trajectory summary ---- """
        # self.summary_sto_pi_TRJ_return_ph = tf_cv1.placeholder(tf.float32,
        #                                                        name=vocab.summary_ph + 'summary_stoPi_trj_return_ph')
        # self.summary_sto_pi_TRJ_return_op = tf_cv1.summary.scalar('Trajectory_return_stochastic_pi',
        #                                                           self.summary_sto_pi_TRJ_return_ph, family=vocab.G)
        #
        # self.summary_sto_pi_TRJ_lenght_ph = tf_cv1.placeholder(tf.float32,
        #                                                        name=vocab.summary_ph + 'summary_stoPi_trj_lenght_ph')
        # self.summary_sto_pi_TRJ_lenght_op = tf_cv1.summary.scalar('Trajectory_lenght_stochastic_pi',
        #                                                           self.summary_sto_pi_TRJ_lenght_ph,
        #                                                           family=vocab.Trajectory_lenght)
        #
        # self.summary_TRJ_op = tf_cv1.summary.merge([self.summary_sto_pi_TRJ_return_op,
        #                                             self.summary_sto_pi_TRJ_lenght_op])

        # endregion
        return None

    def _select_action_given_policy(self, obs_t: Any, deterministic: bool = True, **kwargs: bool):
        """ Make the final policy deterministic at training end for best performance
        
            deterministic policy  -->  'exploiTation'
            stochastic policy     -->  'exploRation'
            
        :param obs_t: a environment observation
        :return: a selected action
        """
        if deterministic:
            """ ---- The 'exploiTation' policy ---- """
            the_policy = self.policy_mu
        else:
            """ ---- The 'exploRation' policy ---- """
            the_policy = self.policy_pi

        obs_t_flat = bloc.format_single_step_observation(obs_t)
        act_t = self.sess.run(the_policy, feed_dict={self.obs_t_ph: obs_t_flat})
        act_t = act_t.ravel()  # for continuous action space.
        # Use 'act_t = blocAndTools.tensorflowbloc.to_scalar(act_t)' for discrete action space
        return act_t
    
    def _instantiate_data_collector(self) -> PoolManager:
        """ Data collector utility """
        return PoolManager(self.exp_spec, playground=self.playground)
    
    def _training_epoch_generator(self, consol_print_learning_stats: ConsolPrintLearningStats, render_env: bool):
        """ Training epoch generator

        :param consol_print_learning_stats:
        :param render_env:
        :yield: (epoch, epoch_loss, stage_stochas_pi_mean_trjs_return, stage_average_trjs_lenght)
        """

        self.pool_manager = self._instantiate_data_collector()
        timecounter = DiscreteTimestepCounter()
        self.experiment_counter = ExperimentClicker()
        self.epoch_metric_logger = EpochMetricLogger()
        consol_print_learning_stats.change_progress_bar_lenght(4)

        """ ---- Build a small fixed obs sample for TF summary distribution of policy_py  ---- """
        small_fixed_obs_sample = []
        obs = self.playground.env.reset()
        for sample in range(100):
            act_t = self.playground.env.action_space.sample()
            obs, _, _, _ = self.playground.env.step(act_t)
            small_fixed_obs_sample.append(obs)
        self.small_fixed_obs_sample_feed_dict = {self.obs_t_ph: small_fixed_obs_sample}

        print(":: SoftActorCritic agent reporting for training ")

        """ ---- Warm-up the computation graph and start learning! ---- """
        with tf_cv1.Session() as sess:
            self.sess = sess
            self.sess.run(tf_cv1.global_variables_initializer())  # initialize random variable in the computation graph

            consol_print_learning_stats.start_the_crazy_experiment()

            # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
            #    Training loop
            # //////////////////////////////////////////////////////////////////////////////////////////////////////////

            """ ---- copy V_psi_frozen parameter to V_psi ---- """
            self.sess.run(self.init_frozen_v_psi_op)

            """ ---- Simulator: Epochs ---- """
            timecounter.reset_global_count()
            for epoch in range(self.exp_spec.max_epoch):
                timecounter.reset_per_epoch_count()
                timecounter.reset_local_count()

                consol_print_learning_stats.next_glorious_epoch()
                self.epoch_metric_logger.new_epoch(epoch)

                """ ---- Simulator: trajectories ---- """
                while timecounter.per_epoch_count < self.exp_spec['timestep_per_epoch']:
                    timecounter.reset_local_count()
                    consol_print_learning_stats.next_glorious_trajectory()

                    obs_t = self.playground.env.reset()
                    """ ---- Simulator: time-steps ---- """
                    while True:
                        timecounter.step_all()

                        if timecounter.global_count > self.exp_spec['min_pool_size']:
                            """ ---- Agent: act in the environment using the stochastic policy---- """
                            act_t = self._select_action_given_policy(obs_t, deterministic=False)
                        else:
                            """ ---- Agent: act randomly at first for better exploration ---- """
                            act_t = self.playground.env.action_space.sample()

                        obs_t_prime, reward_t, trj_done, _ = self.playground.env.step(act_t)

                        """ ---- Agent: collect current timestep events ---- """
                        self.pool_manager.collect_OAnORD(obs_t, act_t, obs_t_prime, reward_t, trj_done)

                        obs_t = obs_t_prime

                        if (timecounter.local_count % self.exp_spec['gradient_step_interval'] == 0
                                and self.pool_manager.current_pool_size > self.exp_spec['min_pool_size']):
                            """ ---- 'Soft Policy Evaluation' & 'Policy Improvement' step ---- """
                            self._perform_gradient_step(consol_print_learning_stats, timecounter=timecounter)

                        if trj_done or timecounter.local_count >= self.exp_spec.max_trj_steps:
                            """ ---- Simulator: trajectory as ended --> compute training stats ---- """
                            trj_return, trj_lenght = self.pool_manager.trajectory_ended()
                            self.epoch_metric_logger.append_trajectory_metric(trj_return, trj_lenght)
    
                            # trj_summary = sess.run(self.summary_TRJ_op, {
                            #     self.summary_sto_pi_TRJ_return_ph: trj_return,
                            #     self.summary_sto_pi_TRJ_lenght_ph: trj_lenght
                            #     })
                            #
                            # self.writer.add_summary(trj_summary, global_step=timecounter.global_count)
    
                            # Muted for speed improvment
                            # consol_print_learning_stats.trajectory_training_stat(the_trajectory_return=trj_return,
                            #                                                      timestep=trj_lenght)
    
                            break

                # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
                # /// Epoch end ////////////////////////////////////////////////////////////////////////////////////////


                """ ---- Evaluate trainning of the agent policy ---- """
                if self.experiment_counter.gradient_step_count > 0:
                    self._run_agent_evaluation(sess, epoch=epoch, render_env=render_env,
                                               max_trajectories=self.exp_spec['max_eval_trj'])

                if ((not self.epoch_metric_logger.is_empty())
                        and self.pool_manager.current_pool_size > self.exp_spec['min_pool_size']):
                    """ ---- Simulator: epoch as ended, fetch training stats and evaluate agent ---- """
                    epoch_eval_mean_trjs_return = self.epoch_metric_logger.agent_eval_mean_trjs_return
                    epoch_eval_mean_trjs_lenght = self.epoch_metric_logger.agent_eval_mean_trjs_lenght
                    epoch_mean_policy_loss = self.epoch_metric_logger.mean_pi_loss

                    epoch_metric = [
                        epoch_eval_mean_trjs_return,
                        epoch_eval_mean_trjs_lenght,
                        self.epoch_metric_logger.mean_trjs_return,
                        self.epoch_metric_logger.mean_trjs_lenght,
                        self.epoch_metric_logger.mean_v_loss,
                        self.epoch_metric_logger.mean_q1_loss,
                        self.epoch_metric_logger.mean_q2_loss,
                        epoch_mean_policy_loss,
                        self.epoch_metric_logger.mean_pi_log_likelihood,
                        # self.epoch_metric_logger.mean_policy_pi,
                        # self.epoch_metric_logger.mean_policy_mu,
                        self.epoch_metric_logger.mean_v_values,
                        self.epoch_metric_logger.mean_frozen_v_values,
                        self.epoch_metric_logger.mean_q1_values,
                        self.epoch_metric_logger.mean_q2_values
                        ]

                    summary_epoch_ph = [
                        self.summary_eval_avg_trjs_return_ph,
                        self.summary_eval_avg_trjs_len_ph,
                        self.summary_avg_trjs_return_ph,
                        self.summary_avg_trjs_len_ph,
                        self.summary_avg_trjs_Vloss_ph,
                        self.summary_avg_trjs_Q1loss_ph,
                        self.summary_avg_trjs_Q2loss_ph,
                        self.summary_avg_trjs_pi_loss_ph,
                        self.summary_avg_pi_log_likelihood_ph,
                        # self.summary_avg_policy_pi_ph,
                        # self.summary_avg_policy_mu_ph,
                        self.summary_avg_V_value_ph,
                        self.summary_avg_frozen_V_value_ph,
                        self.summary_avg_Q1_value_ph,
                        self.summary_avg_Q2_value_ph
                        ]

                    epoch_feed_dictionary = blocAndTools.tensorflowbloc.build_feed_dictionary(summary_epoch_ph,
                                                                                              epoch_metric)

                    epoch_summary = self.sess.run(self.summary_epoch_op, feed_dict=epoch_feed_dictionary)

                    self.writer.add_summary(epoch_summary, global_step=timecounter.global_count)

                    consol_print_learning_stats.epoch_training_stat(
                        epoch_loss=epoch_mean_policy_loss,
                        epoch_average_trjs_return=epoch_eval_mean_trjs_return,
                        epoch_average_trjs_lenght=epoch_eval_mean_trjs_lenght,
                        number_of_trj_collected=self.epoch_metric_logger.nb_trj_collected,
                        total_timestep_collected=self.epoch_metric_logger.total_training_timestep_collected)

                    self._save_learned_model(epoch_eval_mean_trjs_return, epoch, self.sess)

                    """ ---- Expose current epoch computed information for integration test ---- """
                    yield epoch, epoch_mean_policy_loss, epoch_eval_mean_trjs_return, epoch_eval_mean_trjs_lenght
            
            print("\n\n\n:: Global timestep collected: {}".format(timecounter.global_count), end="")
        
        return None

    def _perform_gradient_step(self, consol_print_learning_stats, timecounter: DiscreteTimestepCounter):
        self.experiment_counter.gradient_step()
        replay_batch = self.pool_manager.sample_from_pool()

        full_feed_dictionary = blocAndTools.tensorflowbloc.build_feed_dictionary(
            [self.obs_t_ph, self.act_ph, self.obs_t_prime_ph, self.reward_t_ph, self.trj_done_t_ph],
            [replay_batch.obs_t, replay_batch.act_t, replay_batch.obs_t_prime, replay_batch.rew_t, replay_batch.done_t])

        if timecounter.global_count % self.exp_spec.log_metric_interval == 0:
            losses_op = [self.V_psi_loss, self.q_theta_1_loss, self.q_theta_2_loss, self.actor_kl_loss]
            policy_and_value_fct_op = [self.policy_pi, self.pi_log_likelihood, self.policy_mu,
                                       self.V_psi, self.V_psi_frozen, self.Q_act_1, self.Q_act_2]

            """ ---- Compute metric and collect ---- """
            metric = self.sess.run([*losses_op, *policy_and_value_fct_op], feed_dict=full_feed_dictionary)
            self.epoch_metric_logger.append_all_epoch_metric(*metric)

            """ ---- policy_pi summmary ---- """
            epoch_sumarry_histo = self.sess.run(self.summary_hist_policy_pi,
                                                feed_dict=self.small_fixed_obs_sample_feed_dict)
            self.writer.add_summary(epoch_sumarry_histo, global_step=timecounter.global_count)

        """ ---- 'Soft Policy Evaluation' step ---- """
        critic_optimizer = [self.V_psi_optimizer, self.q_theta_1_optimizer, self.q_theta_2_optimizer]
        self.sess.run(critic_optimizer, feed_dict=full_feed_dictionary)

        """ ---- 'Policy Improvement' step ---- """
        self.sess.run(self.actor_policy_optimizer_op, feed_dict=full_feed_dictionary)

        """ ---- 'Target update' (see SAC original paper, apendice E for result and D for hparam) ---- """
        console_print_interval = 20  # speed improvement

        if timecounter.global_count % self.exp_spec['target_update_interval'] == 0:
            self.experiment_counter.target_update_step()
            self.sess.run(self.V_psi_frozen_update_ops)
            if timecounter.global_count % console_print_interval == 0:
                consol_print_learning_stats.track_2_progress(pre_message="Gradient step",
                                                             progress_1=self.experiment_counter.gradient_step_count,
                                                             counter_str_1='',
                                                             progress_2=self.experiment_counter.target_update_count,
                                                             middle_message='Update V_psi',
                                                             counter_str_2='frozen_V_psi',
                                                             post_message='',
                                                             cursor_1_pre='>',
                                                             cursor_2_pre='-'
                                                             )
        else:
            if timecounter.global_count % console_print_interval == 0:
                consol_print_learning_stats.track_progress(message="Gradient step",
                                                           progress=self.experiment_counter.gradient_step_count,
                                                           counter_str='', post_message='  | No target update')

        return None

    def _run_agent_evaluation(self, sess: tf_cv1.Session, epoch: int, render_env, max_trajectories: int = 10) -> None:
        """
        Evaluate the agent training by forcing it to act deterministicaly.
        
        How: by using the policy mu instead of samplaing from the policy distribution
        
        :param render_env:
        :param sess: the current tf session
        :param max_trajectories: the number of trajectory to execute for evaluation
        :return: None
        """
        epoch += 1
        eval_trajectory_logger = BasicTrajectoryLogger()
        cycle_indexer = CycleIndexer(cycle_lenght=10)
        eval_trj_returns = []
        eval_trj_lenghts = []

        # print("\n:: Agent evaluation >>> \n"
        #       "           ↳ Execute {} run\n".format(max_trajectories))
        #
        # print(":: Running agent evaluation>>> ", end=" ", flush=True)

        for run in range(max_trajectories):
            observation = self.evaluation_playground.env.reset()  # fetch initial observation

            """ ---- Simulator: time-steps ---- """
            while True:
    
                self._render_eval_trj_on_condition(epoch, render_env, run)
    
                act_t = self._select_action_given_policy(observation, deterministic=True)
                observation, reward, done, _ = self.evaluation_playground.env.step(act_t)
    
                timestep = eval_trajectory_logger.lenght
                if timestep % 200 == 0:
                    print("\r     ↳ {:^3} :: Evaluation run {:>4}  |".format(epoch, run + 1),
                          ">" * cycle_indexer.i, " " * cycle_indexer.j,
                          " | reward:", reward, " | timestep:", timestep,
                          sep='', end='', flush=True)
    
                eval_trajectory_logger.push(reward)
                if done or timestep >= self.exp_spec.max_trj_steps:
                    da_return = eval_trajectory_logger.the_return
                    self.epoch_metric_logger.append_agent_eval_trj_metric(da_return,
                                                                          timestep)
                    eval_trj_returns.append(da_return)
                    eval_trj_lenghts.append(timestep)
        
                    print("\r     ↳ {:^3} :: Evaluation run {:>4}  |".format(epoch, run + 1),
                          ">" * cycle_indexer.i, " " * cycle_indexer.j,
                          "  got return {:>8.2f}   after  {:>4}  timesteps".format(da_return,
                                                                                   timestep),
                          sep='', end='', flush=True)
        
                    eval_trajectory_logger.reset()
                    break

        eval_trj_return = np.mean(eval_trj_returns)
        eval_trj_lenght = np.mean(eval_trj_lenghts)

        print("\r     ↳ {:^3} :: Evaluation runs | avg return: {:>8.4f}   avg trj lenght:  {:>4}".format(epoch,
                                                                                                         eval_trj_return,
                                                                                                         eval_trj_lenght))
        return None

    def _render_eval_trj_on_condition(self, epoch, render_env, trj_collected_in_that_epoch):
        """ Render EVALUATION playground"""
        if (render_env and (epoch % self.exp_spec.render_env_every_What_epoch == 0)
                and trj_collected_in_that_epoch % self.exp_spec['render_env_eval_interval'] == 0):
            self.evaluation_playground.env.env.render()  # keep environment rendering turned OFF during unit test
        return None

    def _save_learned_model(self, batch_average_trjs_return: float, epoch, sess: tf_cv1.Session) -> None:
        if batch_average_trjs_return >= float(self.exp_spec.expected_reward_goal):
            print("\n\n    ::  {:>4f} batch avg return reached".format(batch_average_trjs_return))
            self._save_checkpoint(epoch, sess, self.exp_spec.algo_name, batch_average_trjs_return, goal_reached=True)
        elif self.experiment_counter.gradient_step_count % 10000 == 0:
            self._save_checkpoint(epoch, sess, self.exp_spec.algo_name, batch_average_trjs_return, silent=True)
        return None

    def __del__(self):
        # (nice to have) todo:assessment --> is it linked to the 'experiment_runner' rerun error (fail at second rerun)
        tf_cv1.reset_default_graph()

        self.playground.env.env.close()
        self.evaluation_playground.env.env.close()
        print(":: SAC agent >>> CLOSED")
