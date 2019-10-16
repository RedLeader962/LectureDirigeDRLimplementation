# coding=utf-8
"""
REINFORCE agent

Note on TensorBoard usage:

    Start TensorBoard in terminal:
        tensorboard --logdir=BasicPolicyGradient/graph/runs

    In browser, go to:
        http://0.0.0.0:6006/


Note on OpenAi Gym usage:

    For OpenAi Gym registered environment, go to:

        * Bird eye view: https://gym.openai.com/envs
        * Specification: https://github.com/openai/gym/blob/master/gym/envs/__init__.py

            eg:
                register(
                    id='CartPole-v1',
                    entry_point='gym.envs.classic_control:CartPoleEnv',
                    max_episode_steps=500,
                    reward_threshold=475.0,
                )

            'MountainCar-v0', 'MountainCarContinuous-v0',
            'CartPole-v1', 'Pendulum-v0',
            'LunarLander-v2', 'LunarLanderContinuous-v2',
            ...

"""

from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

import numpy as np
from datetime import datetime
from typing import Type

from BasicPolicyGradient.REINFORCEbrain import REINFORCE_policy
from blocAndTools import buildingbloc as bloc
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from blocAndTools.visualisationtools import ConsolPrintLearningStats
from blocAndTools.samplecontainer import TrajectoryCollector, UniformBatchCollector, UniformeBatchContainer
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1   # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()
# endregion

AGENT_ROOT_DIR = 'BasicPolicyGradient'

class REINFORCEagent(object):
    def __init__(self, exp_spec: ExperimentSpec, agent_root_dir: str = AGENT_ROOT_DIR) -> None:
        """
        Build agent computation graph

        :param exp_spec: Experiment specification regarding NN and algo training hparam plus some environment detail
        :type exp_spec: ExperimentSpec
        :param agent_root_dir: The agent root directory
        :type agent_root_dir: str
        """

        self.agent_root_dir = agent_root_dir
        self.exp_spec = exp_spec
        self.playground = GymPlayground(environment_name=exp_spec.prefered_environment)

        self._build_computation_graph(exp_spec)

        """ ---- Setup parameters saving ---- """
        self.saver = tf_cv1.train.Saver()

    def _build_computation_graph(self, exp_spec: ExperimentSpec) -> None:
        """
        Build the Policy_theta computation graph with theta as multi-layer perceptron

        :param exp_spec: Experiment specification regarding NN and algo training hparam plus some environment detail
        :type exp_spec: ExperimentSpec
        """

        """ ---- Placeholder ---- """
        observation_ph, action_ph, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
            self.playground, None, obs_shape_constraint=None)
        self.observation_ph = observation_ph
        self.action_ph = action_ph
        self.Q_values_ph = Q_values_ph

        """ ---- The policy and is neural net theta ---- """
        reinforce_policy = REINFORCE_policy(observation_ph, action_ph, Q_values_ph, exp_spec, self.playground)
        (policy_action_sampler, theta_mlp, pseudo_loss) = reinforce_policy
        self.policy_action_sampler = policy_action_sampler
        self.theta_mlp = theta_mlp
        self.pseudo_loss = pseudo_loss

        """ ---- Optimizer ---- """
        self.policy_optimizer_op = bloc.policy_optimizer(self.pseudo_loss, self.exp_spec.learning_rate)

        return None

    def train(self, render_env: bool = False) -> None:
        """
        Train a REINFORCE agent

        :param render_env: Control over trajectory rendering
        :type render_env: bool
        """

        print("\n:: Environment rendering autorised: {}\n".format(render_env))

        consol_print_learning_stats = ConsolPrintLearningStats(self.exp_spec,
                                                               self.exp_spec.print_metric_every_what_epoch)

        """ ---- setup summary collection for TensorBoard ---- """
        date_now = datetime.now()
        run_str = "Run--{}h{}--{}-{}-{}".format(date_now.hour, date_now.minute, date_now.day,
                                                date_now.month, date_now.year)
        writer = tf_cv1.summary.FileWriter("{}/graph/runs/{}".format(self.agent_root_dir, run_str),
                                           tf_cv1.get_default_graph())

        for epoch in self._training_epoch_generator(consol_print_learning_stats, render_env):
            (epoch, epoch_loss, batch_average_trjs_return, batch_average_trjs_lenght) = epoch

        consol_print_learning_stats.print_experiment_stats(print_plot=not self.exp_spec.isTestRun)
        writer.close()
        return None

    def _training_epoch_generator(self, consol_print_learning_stats: ConsolPrintLearningStats, render_env: bool):
        """
        Training epoch generator

        Mainly use for integration test

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
            for epoch in range(self.exp_spec.max_epoch):
                consol_print_learning_stats.next_glorious_epoch()

                """ ---- Simulator: trajectories ---- """
                while the_UNI_BATCH_COLLECTOR.is_not_full():
                    current_observation = self.playground.env.reset()  # <-- fetch initial observation
                    consol_print_learning_stats.next_glorious_trajectory()

                    """ ---- Simulator: time-steps ---- """
                    while True:
                        self._render_trajectory_on_condition(epoch, render_env,
                                                             the_UNI_BATCH_COLLECTOR.trj_collected_so_far())

                        """ ---- Agent: act in the environment ---- """
                        step_observation = bloc.format_single_step_observation(current_observation)
                        action_array = sess.run(self.policy_action_sampler,
                                                feed_dict={self.observation_ph: step_observation})

                        action = bloc.format_single_step_action(action_array)
                        observe_reaction, reward, done, _ = self.playground.env.step(action)

                        """ ---- Agent: Collect current timestep events ---- """
                        # (Critical) |  Collecting the right observation S_t that trigered the action A_t is critical.
                        #            |  If you collect the observe_reaction S_t+1 coupled to action A_t ...
                        #            |  the agent is doomed!

                        # (Priority) todo:refactor --> the_TRAJECTORY_COLLECTOR.collect_S_t_A_t(): remove reward param
                        # (Priority) todo:implement --> the_TRAJECTORY_COLLECTOR.collect_reward():
                        #     |                                     add assertion that .collect_S_t_A_t() was executed
                        the_TRAJECTORY_COLLECTOR.collect(current_observation, action, reward)
                        current_observation = observe_reaction  # <-- (!)

                        if done:
                            """ ---- Simulator: trajectory as ended ---- """
                            trj_return = the_TRAJECTORY_COLLECTOR.trajectory_ended()

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
                batch_container: UniformeBatchContainer = the_UNI_BATCH_COLLECTOR.pop_batch_and_reset()
                batch_average_trjs_return, batch_average_trjs_lenght = batch_container.compute_metric()

                batch_observations = batch_container.batch_observations
                batch_actions = batch_container.batch_actions
                batch_Q_values = batch_container.batch_Qvalues

                self._data_shape_is_compatibility_with_graph(batch_Q_values, batch_actions, batch_observations)

                """ ---- Agent: Compute gradient & update policy ---- """
                feed_dictionary = bloc.build_feed_dictionary([self.observation_ph, self.action_ph, self.Q_values_ph],
                                                             [batch_observations, batch_actions, batch_Q_values])
                epoch_loss, _ = sess.run([self.pseudo_loss, self.policy_optimizer_op],
                                         feed_dict=feed_dictionary)

                consol_print_learning_stats.epoch_training_stat(
                    epoch_loss=epoch_loss,
                    epoch_average_trjs_return=batch_average_trjs_return,
                    epoch_average_trjs_lenght=batch_average_trjs_lenght,
                    number_of_trj_collected=batch_trj_collected,
                    total_timestep_collected=batch_timestep_collected
                )

                """ ---- Save learned model ---- """
                if batch_average_trjs_return == 200:
                    self._save_checkpoint(epoch, sess, 'REINFORCE')

                """ ---- Expose current epoch computed information for integration test ---- """
                yield (epoch, epoch_loss, batch_average_trjs_return, batch_average_trjs_lenght)

        return None

    def _data_shape_is_compatibility_with_graph(self, batch_Q_values: list, batch_actions: list,
                                                batch_observations: list):
        """ Tensor/ndarray shape compatibility assessment """
        assert self.observation_ph.shape.is_compatible_with(np.array(batch_observations).shape), \
            "Obs: {} != {}".format(self.observation_ph.shape, np.array(batch_observations).shape)
        assert self.action_ph.shape.is_compatible_with(np.array(batch_actions).shape), \
            "Act: {} != {}".format(self.action_ph.shape, np.array(batch_actions).shape)
        assert self.Q_values_ph.shape.is_compatible_with(np.array(batch_Q_values).shape), \
            "Qval: {} != {}".format(self.Q_values_ph.shape, np.array(batch_Q_values).shape)

    def _instantiate_data_collector(self):
        the_TRAJECTORY_COLLECTOR = TrajectoryCollector(self.exp_spec, self.playground)
        the_UNI_BATCH_COLLECTOR = UniformBatchCollector(self.exp_spec.batch_size_in_ts)
        return the_TRAJECTORY_COLLECTOR, the_UNI_BATCH_COLLECTOR

    def _render_trajectory_on_condition(self, epoch, render_env, trj_collected_so_far):
        if (render_env and (epoch % self.exp_spec.render_env_every_What_epoch == 0)
                and trj_collected_so_far == 0):
            self.playground.env.render()  # keep environment rendering turned OFF during unit test

    def _save_checkpoint(self, epoch: int, sess: tf_cv1.Session, graph_name: str):
        self.saver.save(sess, '{}/graph/checkpoint_directory/{}_agent'.format(self.agent_root_dir, graph_name),
                        global_step=epoch)
        print("\n\n    :: {} network parameters were saved\n".format(graph_name))

    def play(self, run_name: str, max_trajectories=20) -> None:
        with tf_cv1.Session() as sess:

            self.load_selected_trained_agent(sess, run_name)

            print(":: Agent player >>> LOCK & LOAD\n"
                  "           ↳ Execute {} run\n           ↳ Test run={}".format(max_trajectories,
                                                                                 self.exp_spec.isTestRun)
                  )

            print(":: Running trajectory >>> ", end=" ", flush=True)
            for run in range(max_trajectories):
                print(run + 1, end=" ", flush=True)

                obs = self.playground.env.reset()  # <-- fetch initial observation
                # recorder = VideoRecorder(playground.env, '../video/cartpole_{}.mp4'.format(run))

                """ ---- Simulator: time-steps ---- """
                while True:

                    if not self.exp_spec.isTestRun:  # keep environment rendering turned OFF during unit test
                        self.playground.env.render()
                        # recorder.capture_frame()

                    """ ---- Agent: act in the environment ---- """
                    step_observation = bloc.format_single_step_observation(obs)
                    action_array = sess.run(self.policy_action_sampler,
                                            feed_dict={self.observation_ph: step_observation})

                    action = bloc.format_single_step_action(action_array)
                    obs_prime, reward, done, _ = self.playground.env.step(action)
                    obs = obs_prime  # <-- (!)

                    if done:
                        break

            print("END")
        # recorder.close()

    def load_selected_trained_agent(self, sess: tf_cv1.Session, run_name: str):
        # (nice to have) todo:implement --> capability to load the last trained agent:
        path = "{}/saved_training".format(self.agent_root_dir)
        self.saver.restore(sess, "{}/{}".format(path, run_name))

    def __del__(self):
        tf_cv1.reset_default_graph()
        self.playground.env.close()
        print(":: Agent >>> CLOSED")


