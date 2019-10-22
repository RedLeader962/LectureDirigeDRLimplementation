# coding=utf-8
from abc import ABCMeta, abstractmethod
from datetime import datetime

import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
import numpy as np

from blocAndTools import buildingbloc as bloc
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from blocAndTools.visualisationtools import ConsolPrintLearningStats

tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False


class Agent(object, metaclass=ABCMeta):
    def __init__(self, exp_spec: ExperimentSpec, agent_root_dir: str = None):
        """
        Build agent computation graph

        :param exp_spec: Experiment specification regarding NN and algo training hparam plus some environment detail
        :type exp_spec: ExperimentSpec
        :param agent_root_dir: The agent root directory
        :type agent_root_dir: str
        """

        if agent_root_dir is not None:
            self.agent_root_dir = agent_root_dir
        else:
            self._use_hardcoded_agent_root_directory()

        self.exp_spec = exp_spec
        self.playground = GymPlayground(environment_name=exp_spec.prefered_environment)

        """ ---- Init computation graph ---- """
        # required placeholder for Agent.play() methode
        self.observation_ph = None
        self.policy_action_sampler = None

        self._build_computation_graph()

        not_implemented_msg = "must be set by _build_computation_graph()"
        assert self.observation_ph is not None, "self.observation_ph {}".format(not_implemented_msg)
        assert self.policy_action_sampler is not None, "self.policy_action_sampler {}".format(not_implemented_msg)

        """ ---- Setup parameters saving ---- """
        self.saver = tf_cv1.train.Saver()
        self.writer = None

    @abstractmethod
    def _use_hardcoded_agent_root_directory(self):
        raise NotImplementedError  # todo: implement
        pass

    @abstractmethod
    def _build_computation_graph(self):
        """
        Build the Policy_theta computation graph with theta as multi-layer perceptron

        Must implement property:
                self.observation_ph
                self.policy_action_sampler
        they are required for Agent.play() methode

        """
        raise NotImplementedError  # todo: implement
        pass

    @abstractmethod
    def _instantiate_data_collector(self):
        """ Data collector utility """
        raise NotImplementedError  # todo: implement
        pass

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
        cleaned_param_name = self.exp_spec.paramameter_set_name.replace(" ", "_")
        run_str = "Run--{}-{}d{}h{}m".format(cleaned_param_name, date_now.day, date_now.hour, date_now.minute,
                                                )
        self.writer = tf_cv1.summary.FileWriter("{}/graph/runs/{}".format(self.agent_root_dir, run_str),
                                           tf_cv1.get_default_graph())

        for epoch in self._training_epoch_generator(consol_print_learning_stats, render_env):
            (epoch, epoch_loss, batch_average_trjs_return, batch_average_trjs_lenght) = epoch

        if self.exp_spec.show_plot:
            consol_print_learning_stats.print_experiment_stats(print_plot=not self.exp_spec.isTestRun)

        self.writer.close()
        return None

    @abstractmethod
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
        raise NotImplementedError  # todo: implement
        pass

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

                    action = bloc.to_scalar(action_array)
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


