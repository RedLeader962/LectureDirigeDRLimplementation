# coding=utf-8

# region ::Import statement ...
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

from blocAndTools.agent import Agent
from blocAndTools import buildingbloc as bloc, ConsolPrintLearningStats
from blocAndTools.container.trajectories_pool import PoolManager
from blocAndTools.rl_vocabulary import rl_name

from SoftActorCritic.SoftActorCriticBrain import (
    apply_action_bound, build_critic_graph_q_theta, build_critic_graph_v_psi, build_gaussian_policy_graph,
    actor_train, critic_q_theta_train, critic_v_psi_train,
    )

tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()


# endregion

class SoftActorCriticAgent(Agent):
    def _use_hardcoded_agent_root_directory(self):
        self.agent_root_dir = 'SoftActorCritic'
        return None


    def _build_computation_graph(self):
        """
        Build the Policy_phi, V_psi and Q_theta computation graph as multi-layer perceptron
        """

        if self.exp_spec.random_seed == 0:
            print(":: Random seed control is turned OFF")
        else:
            tf_cv1.random.set_random_seed(self.exp_spec.random_seed)
            np.random.seed(self.exp_spec.random_seed)
            print(":: Random seed control is turned ON")

        """ ---- Placeholder ---- """
        self.obs_t_ph, self.act_ph, _ = bloc.gym_playground_to_tensorflow_graph_adapter(self.playground)
        self.obs_t_prime_ph = bloc.continuous_space_placeholder(space=self.playground.OBSERVATION_SPACE,
                                                                name=vocab.obs_tPrime_ph)
        self.reward_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.rew_ph)
        self.trj_done_t_ph = tf_cv1.placeholder(dtype=tf.int32, shape=(None,), name=vocab.trj_done_ph)

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                           Actor computation graph                                         *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        pi, pi_log_p, self.policy_mu = build_gaussian_policy_graph(self.obs_t_ph, self.exp_spec, self.playground)

        self.sampled_action, self.sampled_action_log_likelihood = apply_action_bound(pi, pi_log_p)
        
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                           Critic computation graph                                        *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        self.V_psi, self.V_psi_frozen = build_critic_graph_v_psi(self.obs_t_ph, self.obs_t_prime_ph, self.exp_spec)
        
        self.Q_theta_1, self.Q_theta_2 = build_critic_graph_q_theta(self.obs_t_ph, self.act_ph, self.exp_spec)
        
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                       Actor & Critic Training ops                                         *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        self.V_psi_loss, self.V_psi_optimizer, self.V_psi_frozen_update_ops = critic_v_psi_train(self.V_psi,
                                                                                                 self.V_psi_frozen,
                                                                                                 self.Q_theta_1,
                                                                                                 self.Q_theta_2,
                                                                                                 self.sampled_action_log_likelihood,
                                                                                                 self.exp_spec)
        
        q_theta_train_ops = critic_q_theta_train(self.V_psi_frozen, self.Q_theta_1, self.Q_theta_2,
                                                 self.reward_t_ph, self.trj_done_t_ph, self.exp_spec)
        
        self.q_theta_1_loss, self.q_theta_2_loss, self.q_theta_1_optimizer, self.q_theta_2_optimizer = q_theta_train_ops
        
        self.actor_kl_loss, self.actor_policy_optimizer_op = actor_train(self.sampled_action_log_likelihood,
                                                                         self.Q_theta_1, self.Q_theta_2, self.exp_spec)
        
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                                 Summary ops                                               *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        
        """ ---- By Epoch summary ---- """
        self.summary_stage_avg_trjs_actor_KL_loss_ph = tf_cv1.placeholder(tf.float32, name='Actor_KL_loss_ph')
        self.summary_stage_avg_trjs_critic_Vloss_ph = tf_cv1.placeholder(tf.float32, name='Critic_V_loss_ph')
        self.summary_stage_avg_trjs_critic_Q1loss_ph = tf_cv1.placeholder(tf.float32, name='Critic_Q1_loss_ph')
        self.summary_stage_avg_trjs_critic_Q2loss_ph = tf_cv1.placeholder(tf.float32, name='Critic_Q2_loss_ph')
        tf_cv1.summary.scalar('Actor_KL_loss', self.summary_stage_avg_trjs_actor_KL_loss_ph, family=vocab.loss)
        tf_cv1.summary.scalar('Critic_V_loss', self.summary_stage_avg_trjs_critic_Vloss_ph, family=vocab.loss)
        tf_cv1.summary.scalar('Critic_Q1_loss', self.summary_stage_avg_trjs_critic_Q1loss_ph, family=vocab.loss)
        tf_cv1.summary.scalar('Critic_Q2_loss', self.summary_stage_avg_trjs_critic_Q2loss_ph, family=vocab.loss)
        
        self.summary_stochas_pi_stage_avg_trjs_return_ph = tf_cv1.placeholder(
            tf.float32, name='summary_stochas_pi_stage_avg_trjs_return_ph')
        tf_cv1.summary.scalar('Stage average return (stochastic pi)',
                              self.summary_stochas_pi_stage_avg_trjs_return_ph, family=vocab.G)
        
        self.summary_epoch_op = tf_cv1.summary.merge_all()
        
        """ ---- By Trajectory summary ---- """
        self.Summary_stochas_pi_trj_return_ph = tf_cv1.placeholder(tf.float32, name='Summary_stochas_pi_trj_return_ph')
        self.summary_stocahs_pi_trj_return_op = tf_cv1.summary.scalar('Trajectory return (stochastic pi)',
                                                                      self.Summary_stochas_pi_trj_return_ph,
                                                                      family=vocab.G)

        self.Summary_stochas_pi_trj_lenght_ph = tf_cv1.placeholder(tf.float32, name='Summary_stochas_pi_trj_lenght_ph')
        self.summary_stochas_pi_trj_lenght_op = tf_cv1.summary.scalar('Trajectory lenght (stochastic pi)',
                                                                      self.Summary_stochas_pi_trj_lenght_ph,
                                                                      family=vocab.Trajectory_lenght)

        self.summary_trj_op = tf_cv1.summary.merge([self.summary_stocahs_pi_trj_return_op,
                                                    self.summary_stochas_pi_trj_lenght_op])
        return None


    def define_policy_action_sampler(self, deterministic=False) -> None:
        """
        Make the final policy deterministic at training end for best performance
        
        :param deterministic:
        :return:
        """
        if deterministic:
            self.policy_action_sampler = self.policy_mu
        else:
            self.policy_action_sampler = self.sampled_action
        return None


    def select_action_given_policy(self, sess, obs, deterministic=False, **kwargs):
        """
        Make the final policy deterministic at training end for best performance

        :param deterministic: use the policy distribution mean instead of a sample from the policy distribution
        :return: a selected action
        """
        if deterministic:
            pi = self.policy_mu
        else:
            pi = self.sampled_action
    
        obs_t_flat = bloc.format_single_step_observation(obs)
        act_t = sess.run(pi,
                         feed_dict={self.obs_t_ph: obs_t_flat})
        act_t = bloc.to_scalar(act_t)
        return act_t


    def _instantiate_data_collector(self) -> PoolManager:
        """
        Data collector utility
        """
        return PoolManager(self.exp_spec, playground=self.playground)


    def _training_epoch_generator(self, consol_print_learning_stats: ConsolPrintLearningStats, render_env: bool):
        # todo:implement --> task: en cours
        """
        Training epoch generator

        :param consol_print_learning_stats:
        :param render_env:
        :yield: (epoch, epoch_loss, stage_average_trjs_return, stage_average_trjs_lenght)
        """
    
        self.pool_manager = self._instantiate_data_collector()
    
        # Note: Second environment for policy evaluation
        self.evaluation_playground = bloc.GymPlayground(environment_name=self.exp_spec.prefered_environment)
    
        print(":: SoftActorCritic agent reporting for training ")
    
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
        
            """ ---- copy V_psi_frozen parameter to V_psi ---- """
            self.sess.run(self.V_psi_frozen_update_ops)
        
            """ ---- Simulator: Epochs ---- """
            global_timestep_idx = 0
            for epoch in range(self.exp_spec.max_epoch):
                # (Ice-Boxed) todo:implement --> finish lr sheduler for online shared algo:
                # (Ice-Boxed) todo:implement --> add 'global_timestep_max' to hparam:
                # if global_timestep_idx >= self.exp_spec['global_timestep_max']:
                #     break
            
                consol_print_learning_stats.next_glorious_epoch()
                
                """ ---- Simulator: trajectories ---- """
                while True:
                    obs_t = self.playground.env.reset()
                    consol_print_learning_stats.next_glorious_trajectory()
                    
                    """ ---- Simulator: time-steps ---- """
                    local_step_t = 0
                    while True:
                        local_step_t += 1
                        global_timestep_idx += 1
                        self._render_trajectory_on_condition(epoch, render_env,
                                                             self.pool_manager.trj_collected_so_far())
    
                        """ ---- Agent: act in the environment ---- """
                        obs_t_flat = bloc.format_single_step_observation(obs_t)
                        act_t = self.sess.run(self.sampled_action,
                                              feed_dict={
                                                  self.obs_t_ph: obs_t_flat
                                                  })
                        act_t = bloc.to_scalar(act_t)
    
                        obs_t_prime, reward_t, trj_done, _ = self.playground.env.step(act_t)
    
                        """ ---- Agent: collect current timestep events ---- """
                        self.pool_manager.collect_OAnORD(obs_t, act_t, obs_t_prime, reward_t, trj_done)
    
                        obs_t = obs_t_prime
    
                        if local_step_t % self.exp_spec['gradient_step_interval'] == 0:
                            """ ---- 'Soft Policy Evaluation' & 'Policy Improvement' step ---- """
                            self._perform_gradient_step(consol_print_learning_stats, local_step_t)
    
                        if trj_done:
                            """ ---- Simulator: trajectory as ended --> compute training stats ---- """
                            trj_return, trj_lenght = self.pool_manager.trajectory_ended()
        
                            trj_summary = sess.run(self.summary_trj_op,
                                                   {
                                                       self.Summary_stochas_pi_trj_return_ph: trj_return,
                                                       self.Summary_stochas_pi_trj_lenght_ph: trj_lenght
                                                       })
        
                            self.writer.add_summary(trj_summary, global_step=global_timestep_idx)
        
                            consol_print_learning_stats.trajectory_training_stat(the_trajectory_return=trj_return,
                                                                                 timestep=trj_lenght)
                            break
            
                """ ---- Simulator: epoch as ended, fetch training stats and evaluate agent ---- """
                # stage_trj_collected = experimentCOLLECTOR.trj_collected_so_far()
                # stage_timestep_collected = experimentCOLLECTOR.timestep_collected_so_far()
                #
                # """ ---- Prepare data for backpropagation in the neural net ---- """
                # experiment_container = experimentCOLLECTOR.pop_batch_and_reset()
                # stage_average_trjs_return, stage_average_trjs_lenght = experiment_container.get_basic_metric()
                # stage_actor_mean_loss, stage_critic_mean_loss = experiment_container.get_stage_mean_loss()
                #
                # epoch_feed_dictionary = bloc.build_feed_dictionary([self.summary_stage_avg_trjs_actor_loss_ph,
                #                                                     self.summary_stage_avg_trjs_critic_loss_ph,
                #                                                     self.summary_stochas_pi_stage_avg_trjs_return_ph],
                #                                                    [stage_actor_mean_loss,
                #                                                     stage_critic_mean_loss,
                #                                                     stage_average_trjs_return])
                #
                # epoch_summary = self.sess.run(self.summary_epoch_op, feed_dict=epoch_feed_dictionary)
                #
                # self.writer.add_summary(epoch_summary, global_step=global_timestep_idx)
                #
                # consol_print_learning_stats.epoch_training_stat(
                #     epoch_loss=stage_actor_mean_loss,
                #     epoch_average_trjs_return=stage_average_trjs_return,
                #     epoch_average_trjs_lenght=stage_average_trjs_lenght,
                #     number_of_trj_collected=stage_trj_collected,
                #     total_timestep_collected=stage_timestep_collected)
                #
                # self._save_learned_model(stage_average_trjs_return, epoch, self.sess)
                #
                # """ ---- Expose current epoch computed information for integration test ---- """
                # yield (epoch, stage_actor_mean_loss, stage_average_trjs_return, stage_average_trjs_lenght)
        
            print("\n\n\n:: Global timestep collected: {}".format(global_timestep_idx), end="")
    
        return None


    def _perform_gradient_step(self, consol_print_learning_stats, current_local_step_t):
        replay_batch = self.pool_manager.sample_from_pool()
    
        full_feed_dictionary = bloc.build_feed_dictionary(
            [self.obs_t_ph, self.act_ph, self.obs_t_prime_ph, self.reward_t_ph, self.trj_done_t_ph],
            [replay_batch.obs_t, replay_batch.act_t, replay_batch.obs_t_prime, replay_batch.rew_t, replay_batch.done_t])
    
        """ ---- Compute metric and collect ---- """
        losses = self.sess.run([self.V_psi_loss, self.q_theta_1_loss, self.q_theta_2_loss, self.actor_kl_loss],
                               feed_dict=full_feed_dictionary)
        minibatch_v_loss, minibatch_q1_loss, minibatch_q2_loss, minibatch_actor_kl_loss = losses
        # (Priority) todo:implement --> collect all loss, compute mean and push to TF summary:
    
        # self.trjCOLLECTOR.collect_loss(critic_loss=minibatch_V_loss, actor_loss=minibatch_actor_KL_loss)
    
        consol_print_learning_stats.track_progress(progress=current_local_step_t, message="Agent training")
        critic_optimizer = [self.V_psi_optimizer, self.q_theta_1_optimizer, self.q_theta_2_optimizer]
    
        """ ---- 'Soft Policy Evaluation' step ---- """
        self.sess.run(critic_optimizer, feed_dict=full_feed_dictionary)
    
        """ ---- 'Policy Improvement' step ---- """
        self.sess.run(self.actor_policy_optimizer_op, feed_dict=full_feed_dictionary)
    
        """ ---- Target weight update interval (see SAC original paper, apendice E for result and D for hparam) ---- """
        if current_local_step_t == self.exp_spec['target_weight_update_interval']:
            self.sess.run(self.V_psi_frozen_update_ops)
    
        return None
