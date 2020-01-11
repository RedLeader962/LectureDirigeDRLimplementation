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

import blocAndTools.tensorflowbloc
from BasicPolicyGradient.REINFORCEbrain import REINFORCE_policy
from blocAndTools import buildingbloc as bloc
from blocAndTools.agent import Agent
from blocAndTools.visualisationtools import ConsolPrintLearningStats
from blocAndTools.container.samplecontainer import TrajectoryCollector, UniformBatchCollector, UniformeBatchContainer
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1   # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()
# endregion


class REINFORCEagent(Agent):

    def _use_hardcoded_agent_root_directory(self) -> None:
        self.agent_root_dir = 'BasicPolicyGradient'
        return None

    def _build_computation_graph(self) -> None:
        """
        Build the Policy_theta computation graph with theta as multi-layer perceptron

        """

        """ ---- Placeholder ---- """
        observation_ph, action_ph, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(self.playground,
                                                                                                 obs_shape_constraint=None,
                                                                                                 action_shape_constraint=None)
        self.obs_t_ph = observation_ph
        self.action_ph = action_ph
        self.Q_values_ph = Q_values_ph

        """ ---- The policy and is neural net theta ---- """
        reinforce_policy = REINFORCE_policy(observation_ph, action_ph, Q_values_ph, self.exp_spec, self.playground)
        (policy_action_sampler, theta_mlp, pseudo_loss) = reinforce_policy
        self.policy_pi = policy_action_sampler
        self.theta_mlp = theta_mlp
        self.pseudo_loss = pseudo_loss

        """ ---- Optimizer ---- """
        self.policy_optimizer_op = bloc.policy_optimizer(self.pseudo_loss, self.exp_spec.learning_rate)
        return None

    def _instantiate_data_collector(self):
        """
        Data collector utility

        :return: Collertor utility
        :rtype: (TrajectoryCollector, UniformBatchCollector)
        """
        the_TRAJECTORY_COLLECTOR = TrajectoryCollector(self.exp_spec, self.playground)
        the_UNI_BATCH_COLLECTOR = UniformBatchCollector(self.exp_spec.batch_size_in_ts)
        return the_TRAJECTORY_COLLECTOR, the_UNI_BATCH_COLLECTOR

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
                        action_array = sess.run(self.policy_pi,
                                                feed_dict={self.obs_t_ph: step_observation})

                        action = blocAndTools.tensorflowbloc.to_scalar(action_array)
                        observe_reaction, reward, done, _ = self.playground.env.step(action)

                        """ ---- Agent: Collect current timestep events ---- """
                        # (Critical) |  Collecting the right observation S_t that trigered the action A_t is critical.
                        #            |  If you collect_OAR the observe_reaction S_t+1 coupled to action A_t ...
                        #            |  the agent is doomed!

                        # (Priority) todo:refactor --> the_TRAJECTORY_COLLECTOR.collect_S_t_A_t(): remove reward param
                        # (Priority) todo:implement --> the_TRAJECTORY_COLLECTOR.collect_reward():
                        #     |                                     add assertion that .collect_S_t_A_t() was executed
                        the_TRAJECTORY_COLLECTOR.collect_OAR(current_observation, action, reward)
                        current_observation = observe_reaction  # <-- (!)

                        if done:
                            """ ---- Simulator: trajectory as ended ---- """
                            trj_return = the_TRAJECTORY_COLLECTOR.trajectory_ended()

                            """ ---- Agent: Collect the sampled trajectory  ---- """
                            the_TRAJECTORY_COLLECTOR.compute_Qvalues_as_rewardToGo()
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
                batch_container = the_UNI_BATCH_COLLECTOR.pop_batch_and_reset()
                batch_average_trjs_return, batch_average_trjs_lenght = batch_container.get_basic_metric()

                batch_observations = batch_container.batch_observations
                batch_actions = batch_container.batch_actions
                batch_Q_values = batch_container.batch_Qvalues

                # self._data_shape_is_compatibility_with_graph(batch_Q_values, batch_actions, batch_observations)

                """ ---- Agent: Compute gradient & update policy ---- """
                feed_dictionary = blocAndTools.tensorflowbloc.build_feed_dictionary(
                    [self.obs_t_ph, self.action_ph, self.Q_values_ph],
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

                self._save_learned_model(batch_average_trjs_return, epoch, sess)

                """ ---- Expose current epoch computed information for integration test ---- """
                yield (epoch, epoch_loss, batch_average_trjs_return, batch_average_trjs_lenght)

        return None

    # def _data_shape_is_compatibility_with_graph(self, batch_Q_values: list, batch_actions: list,
    #                                             batch_observations: list):
    #     """ Tensor/ndarray shape compatibility assessment """
    #     assert self.obs_t_ph.shape.is_compatible_with(np.array(batch_observations).shape), \
    #         "Obs: {} != {}".format(self.obs_t_ph.shape, np.array(batch_observations).shape)
    #     assert self.action_ph.shape.is_compatible_with(np.array(batch_actions).shape), \
    #         "Act: {} != {}".format(self.action_ph.shape, np.array(batch_actions).shape)
    #     assert self.Q_values_ph.shape.is_compatible_with(np.array(batch_Q_values).shape), \
    #         "Qval: {} != {}".format(self.Q_values_ph.shape, np.array(batch_Q_values).shape)
    #     return None
