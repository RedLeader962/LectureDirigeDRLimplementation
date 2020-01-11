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
    critic_q_theta_train, critic_v_psi_train,
    )
from blocAndTools import ConsolPrintLearningStats, buildingbloc as bloc
from blocAndTools.agent import Agent
from blocAndTools.logger.basic_trajectory_logger import BasicTrajectoryLogger
from blocAndTools.discrete_time_counter import DiscreteTimestepCounter
from blocAndTools.logger.epoch_metric_logger import EpochMetricLogger
from blocAndTools.container.trajectories_pool import PoolManager
from blocAndTools.rl_vocabulary import rl_name

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
        
        if self.exp_spec.random_seed == 0:
            print(":: Random seed control is turned OFF")
        else:
            tf_cv1.random.set_random_seed(self.exp_spec.random_seed)
            np.random.seed(self.exp_spec.random_seed)
            print(":: Random seed control is turned ON")

        """ ---- Placeholder ---- """
        self.obs_t_ph, self.act_ph, _ = bloc.gym_playground_to_tensorflow_graph_adapter(self.playground)
        # self.obs_t_prime_ph = bloc.continuous_space_placeholder(space=self.playground.OBSERVATION_SPACE,
        #                                                         name=vocab.obs_tPrime_ph)
        self.obs_t_prime_ph, _, _ = bloc.gym_playground_to_tensorflow_graph_adapter(self.playground,
                                                                                    obs_ph_name=vocab.obs_tPrime_ph)
        self.reward_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.rew_ph)
        self.trj_done_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.trj_done_ph)

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # /// Actor computation graph //////////////////////////////////////////////////////////////////////////////////

        pi, pi_log_p, self.policy_mu = build_gaussian_policy_graph(self.obs_t_ph, self.exp_spec, self.playground)

        self.policy_pi, self.pi_log_likelihood = apply_action_bound(pi, pi_log_p)

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # /// Critic computation graph /////////////////////////////////////////////////////////////////////////////////

        self.V_psi, self.V_psi_frozen = build_critic_graph_v_psi(self.obs_t_ph, self.obs_t_prime_ph, self.exp_spec)

        self.Q_theta_1, self.Q_theta_2 = build_critic_graph_q_theta(self.obs_t_ph, self.act_ph, self.exp_spec)

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # /// Actor & Critic Training ops //////////////////////////////////////////////////////////////////////////////

        self.V_psi_loss, self.V_psi_optimizer, self.V_psi_frozen_update_ops = critic_v_psi_train(self.V_psi,
                                                                                                 self.V_psi_frozen,
                                                                                                 self.Q_theta_1,
                                                                                                 self.Q_theta_2,
                                                                                                 self.pi_log_likelihood,
                                                                                                 self.exp_spec)

        q_theta_train_ops = critic_q_theta_train(self.V_psi_frozen, self.Q_theta_1, self.Q_theta_2,
                                                 self.reward_t_ph, self.trj_done_t_ph, self.exp_spec)

        self.q_theta_1_loss, self.q_theta_2_loss, self.q_theta_1_optimizer, self.q_theta_2_optimizer = q_theta_train_ops

        self.actor_kl_loss, self.actor_policy_optimizer_op = actor_train(self.pi_log_likelihood,
                                                                         self.Q_theta_1, self.Q_theta_2, self.exp_spec)

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # /// Summary ops //////////////////////////////////////////////////////////////////////////////////////////////

        """ ---- By Epoch summary: RETURNS & LENGHT ---- """
        self.summary_sto_pi_avg_trjs_return_ph = tf_cv1.placeholder(tf.float32,
                                                                    name='summary_stoPi_stage_avg_trjs_return_ph')
        tf_cv1.summary.scalar('Epoch average trj return (stochastic pi)', self.summary_sto_pi_avg_trjs_return_ph,
                              family=vocab.G)

        self.summary_sto_pi_avg_trjs_len_ph = tf_cv1.placeholder(tf.float32, name='summary_stoPi_stage_avg_trjs_len_ph')
        tf_cv1.summary.scalar('Epoch average trj lenght (stochastic pi)', self.summary_sto_pi_avg_trjs_len_ph,
                              family=vocab.Trajectory_lenght)

        self.summary_det_pi_avg_trjs_return_ph = tf_cv1.placeholder(tf.float32,
                                                                    name='summary_detPi_stage_avg_trjs_return_ph')
        tf_cv1.summary.scalar('Epoch average trj return (deterministic pi)', self.summary_det_pi_avg_trjs_return_ph,
                              family=vocab.G)

        self.summary_det_pi_avg_trjs_len_ph = tf_cv1.placeholder(tf.float32, name='summary_detPi_stage_avg_trjs_len_ph')
        tf_cv1.summary.scalar('Epoch average trj lenght (deterministic pi)', self.summary_det_pi_avg_trjs_len_ph,
                              family=vocab.Trajectory_lenght)

        """ ---- By Epoch summary: LOSS ---- """
        self.summary_avg_trjs_Vloss_ph = tf_cv1.placeholder(tf.float32, name='Critic_V_loss_ph')
        tf_cv1.summary.scalar('Critic V loss', self.summary_avg_trjs_Vloss_ph, family=vocab.loss)

        self.summary_avg_trjs_Q1loss_ph = tf_cv1.placeholder(tf.float32, name='Critic_Q1_loss_ph')
        tf_cv1.summary.scalar('Critic Q1 loss', self.summary_avg_trjs_Q1loss_ph, family=vocab.loss)

        self.summary_avg_trjs_Q2loss_ph = tf_cv1.placeholder(tf.float32, name='Critic_Q2_loss_ph')
        tf_cv1.summary.scalar('Critic Q2 loss', self.summary_avg_trjs_Q2loss_ph, family=vocab.loss)
        
        self.summary_avg_trjs_pi_loss_ph = tf_cv1.placeholder(tf.float32, name='policy_loss_ph')
        tf_cv1.summary.scalar('Policy loss', self.summary_avg_trjs_pi_loss_ph, family=vocab.loss)
        
        """ ---- By Epoch summary: POLICY & VALUE fct ---- """
        self.summary_avg_pi_log_likelihood_ph = tf_cv1.placeholder(tf.float32, name='pi_log_p_ph')
        tf_cv1.summary.scalar('policy log likelihood', self.summary_avg_pi_log_likelihood_ph, family=vocab.policy)
        
        self.summary_avg_policy_pi_ph = tf_cv1.placeholder(tf.float32, name='policy_pi_ph')
        tf_cv1.summary.scalar('policy py', self.summary_avg_policy_pi_ph, family=vocab.policy)
        
        self.summary_avg_policy_mu_ph = tf_cv1.placeholder(tf.float32, name='policy_mu_ph')
        tf_cv1.summary.scalar('policy mu', self.summary_avg_policy_mu_ph, family=vocab.policy)
        
        self.summary_avg_V_value_ph = tf_cv1.placeholder(tf.float32, name='V_values_ph')
        tf_cv1.summary.scalar('V values', self.summary_avg_V_value_ph, family=vocab.values)
        
        self.summary_avg_Q1_value_ph = tf_cv1.placeholder(tf.float32, name='Q1_values_ph')
        tf_cv1.summary.scalar('Q1 values', self.summary_avg_Q1_value_ph, family=vocab.values)
        
        self.summary_avg_Q2_value_ph = tf_cv1.placeholder(tf.float32, name='Q2_values_ph')
        tf_cv1.summary.scalar('Q2 values', self.summary_avg_Q2_value_ph, family=vocab.values)
        
        self.summary_epoch_op = tf_cv1.summary.merge_all()
        
        """ ---- By Trajectory summary ---- """
        self.summary_sto_pi_TRJ_return_ph = tf_cv1.placeholder(tf.float32, name='summary_stoPi_trj_return_ph')
        self.summary_sto_pi_TRJ_return_op = tf_cv1.summary.scalar('Trajectory return (stochastic pi)',
                                                                  self.summary_sto_pi_TRJ_return_ph, family=vocab.G)
        
        self.summary_sto_pi_TRJ_lenght_ph = tf_cv1.placeholder(tf.float32, name='summary_stoPi_trj_lenght_ph')
        self.summary_sto_pi_TRJ_lenght_op = tf_cv1.summary.scalar('Trajectory lenght (stochastic pi)',
                                                                  self.summary_sto_pi_TRJ_lenght_ph,
                                                                  family=vocab.Trajectory_lenght)
        
        self.summary_TRJ_op = tf_cv1.summary.merge([self.summary_sto_pi_TRJ_return_op,
                                                    self.summary_sto_pi_TRJ_lenght_op])
        return None

    def _select_action_given_policy(self, sess: tf_cv1.Session, obs_t: Any, deterministic: bool = True, **kwargs):
        """ Make the final policy deterministic at training end for best performance
        
            deterministic policy  -->  'exploiTation'
            stochastic policy     -->  'exploRation'
            
        :param sess: the curent session
        :param obs_t: a environment observation
        :param deterministic: use the policy distribution mean instead of a sample from the policy distribution
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
        act_t = blocAndTools.tensorflowbloc.to_scalar(act_t)
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
        self.epoch_metric_logger = EpochMetricLogger()
        
        # (Priority) todo:implement --> agent evaluation functionality:
        # Note: Second environment for policy evaluation
        self.evaluation_playground = bloc.GymPlayground(environment_name=self.exp_spec.prefered_environment)
        
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
            self.sess.run(self.V_psi_frozen_update_ops)

            """ ---- Simulator: Epochs ---- """
            timecounter.reset_global_count()
            for epoch in range(self.exp_spec.max_epoch):
                # (Ice-Boxed) todo:implement --> add 'global_timestep_max' to hparam:
                # if global_timestep_idx >= self.exp_spec['global_timestep_max']:
                #     break

                consol_print_learning_stats.next_glorious_epoch()
                self.epoch_metric_logger._epoch_id = epoch

                """ ---- Simulator: trajectories ---- """
                while timecounter.local_count < self.exp_spec['timestep_per_epoch']:
                    timecounter.reset_local_count()

                    obs_t = self.playground.env.reset()
                    consol_print_learning_stats.next_glorious_trajectory()

                    """ ---- Simulator: time-steps ---- """
                    while True:
                        timecounter.local_and_global_step()

                        self._render_trajectory_on_condition(epoch, render_env,
                                                             self.pool_manager.trj_collected_so_far())

                        """ ---- Agent: act in the environment using the stochastic policy---- """
                        act_t = self._select_action_given_policy(sess, obs_t, deterministic=False)

                        obs_t_prime, reward_t, trj_done, _ = self.playground.env.step(act_t)

                        """ ---- Agent: collect current timestep events ---- """
                        self.pool_manager.collect_OAnORD(obs_t, act_t, obs_t_prime, reward_t, trj_done)

                        obs_t = obs_t_prime

                        if (timecounter.local_count % self.exp_spec['gradient_step_interval'] == 0
                                and self.pool_manager.current_pool_size > self.exp_spec['min_pool_size']):
                            """ ---- 'Soft Policy Evaluation' & 'Policy Improvement' step ---- """
                            self._perform_gradient_step(consol_print_learning_stats, timecounter.local_count)

                        if trj_done:
                            """ ---- Simulator: trajectory as ended --> compute training stats ---- """
                            trj_return, trj_lenght = self.pool_manager.trajectory_ended()
                            self.epoch_metric_logger.append_trajectory_metric(trj_return, trj_lenght)

                            trj_summary = sess.run(self.summary_TRJ_op, {
                                self.summary_sto_pi_TRJ_return_ph: trj_return,
                                self.summary_sto_pi_TRJ_lenght_ph: trj_lenght
                                })

                            self.writer.add_summary(trj_summary, global_step=timecounter.global_count)

                            consol_print_learning_stats.trajectory_training_stat(the_trajectory_return=trj_return,
                                                                                 timestep=trj_lenght)
                            break

                # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
                # /// Epoch end ////////////////////////////////////////////////////////////////////////////////////////

                """ ---- Evaluate trainning of the agent policy ---- """
                self._run_agent_evaluation(sess, max_trajectories=self.exp_spec['max_eval_trj'])

                """ ---- Simulator: epoch as ended, fetch training stats and evaluate agent ---- """
                epoch_mean_policy_loss = self.epoch_metric_logger.mean_pi_loss
                epoch_eval_mean_trjs_return = self.epoch_metric_logger.agent_eval_mean_trjs_return
                epoch_eval_mean_trjs_lenght = self.epoch_metric_logger.agent_eval_mean_trjs_lenght

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
                    self.epoch_metric_logger.mean_policy_pi,
                    self.epoch_metric_logger.mean_policy_mu,
                    self.epoch_metric_logger.mean_v_values,
                    self.epoch_metric_logger.mean_q1_values,
                    self.epoch_metric_logger.mean_q2_values
                    ]

                summary_epoch_ph = [
                    self.summary_det_pi_avg_trjs_return_ph,
                    self.summary_det_pi_avg_trjs_len_ph,
                    self.summary_sto_pi_avg_trjs_return_ph,
                    self.summary_sto_pi_avg_trjs_len_ph,
                    self.summary_avg_trjs_Vloss_ph,
                    self.summary_avg_trjs_Q1loss_ph,
                    self.summary_avg_trjs_Q2loss_ph,
                    self.summary_avg_trjs_pi_loss_ph,
                    self.summary_avg_pi_log_likelihood_ph,
                    self.summary_avg_policy_pi_ph,
                    self.summary_avg_policy_mu_ph,
                    self.summary_avg_V_value_ph,
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
    
    def _perform_gradient_step(self, consol_print_learning_stats, current_local_step):
        replay_batch = self.pool_manager.sample_from_pool()

        full_feed_dictionary = blocAndTools.tensorflowbloc.build_feed_dictionary(
            [self.obs_t_ph, self.act_ph, self.obs_t_prime_ph, self.reward_t_ph, self.trj_done_t_ph],
            [replay_batch.obs_t, replay_batch.act_t, replay_batch.obs_t_prime, replay_batch.rew_t, replay_batch.done_t])
        
        losses_op = [self.V_psi_loss, self.q_theta_1_loss, self.q_theta_2_loss, self.actor_kl_loss]
        policy_and_value_fct_op = [self.policy_pi, self.pi_log_likelihood, self.policy_mu,
                                   self.V_psi, self.Q_theta_1, self.Q_theta_2]
        
        """ ---- Compute metric and collect ---- """
        metric = self.sess.run([*losses_op, *policy_and_value_fct_op], feed_dict=full_feed_dictionary)
        self.epoch_metric_logger.append_all_epoch_metric(*metric)
        
        consol_print_learning_stats.track_progress(progress=current_local_step, message="Agent training")
        
        """ ---- 'Soft Policy Evaluation' step ---- """
        critic_optimizer = [self.V_psi_optimizer, self.q_theta_1_optimizer, self.q_theta_2_optimizer]
        self.sess.run(critic_optimizer, feed_dict=full_feed_dictionary)

        """ ---- 'Policy Improvement' step ---- """
        self.sess.run(self.actor_policy_optimizer_op, feed_dict=full_feed_dictionary)

        """ ---- Target weight update interval (see SAC original paper, apendice E for result and D for hparam) ---- """
        if current_local_step == self.exp_spec['target_update_interval']:
            self.sess.run(self.V_psi_frozen_update_ops)

        return None

    def _run_agent_evaluation(self, sess: tf_cv1.Session, max_trajectories: int = 10) -> None:
        """
        Evaluate the agent training by forcing it to act deterministicaly.
        
        How: by using the policy mu instead of samplaing from the policy distribution
        
        :param sess: the current tf session
        :param max_trajectories: the number of trajectory to execute for evaluation
        :return: None
        """
        trajectory_logger = BasicTrajectoryLogger()

        print(":: Agent evaluation >>> LOCK & LOAD\n"
              "           ↳ Execute {} run\n           ↳ Test run={}".format(max_trajectories,
                                                                             self.exp_spec.isTestRun))

        print(":: Running agent evaluation>>> ", end=" ", flush=True)

        for run in range(max_trajectories):
            print(run + 1, end=" ", flush=True)
            observation = self.evaluation_playground.env.reset()  # fetch initial observation

            """ ---- Simulator: time-steps ---- """
            while True:
                act_t = self._select_action_given_policy(sess, observation, deterministic=True)
                observation, reward, done, _ = self.evaluation_playground.env.step(act_t)

                trajectory_logger.push(reward)

                if done:
                    self.epoch_metric_logger.append_agent_eval_trj_metric(trajectory_logger.the_return,
                                                                          trajectory_logger.lenght)
                    trajectory_logger.reset()
                    break
        return None
