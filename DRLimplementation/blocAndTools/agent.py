# coding=utf-8
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import blocAndTools.tensorflowbloc
from blocAndTools import buildingbloc as bloc
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground, setup_commented_run_dir_str
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
        self.obs_t_ph = None
        self.policy_pi = None

        self._build_computation_graph()

        not_implemented_msg = "must be set by _build_computation_graph()"
        assert self.obs_t_ph is not None, "self.obs_t_ph {}".format(not_implemented_msg)
        assert self.policy_pi is not None, "self.policy_pi {}".format(not_implemented_msg)

        """ ---- Setup parameters saving ---- """
        self.saver = tf_cv1.train.Saver()
        self.writer = None
        self.this_run_dir = None

    @abstractmethod
    def _use_hardcoded_agent_root_directory(self):
        raise NotImplementedError

    @abstractmethod
    def _build_computation_graph(self):
        """
        Build the Policy_theta computation graph with theta as multi-layer perceptron

        Must implement property:
                self.obs_t_ph
                self.policy_pi
        they are required for Agent.play() methode

        """
        raise NotImplementedError

    @abstractmethod
    def _instantiate_data_collector(self):
        """ Data collector utility """
        raise NotImplementedError

    def train(self, render_env: bool = False) -> None:
        """
        Train a REINFORCE agent

        :param render_env: Control over trajectory rendering
        :type render_env: bool
        """

        print(":: Environment rendering autorised: {}".format(render_env))

        consol_print_learning_stats = ConsolPrintLearningStats(self.exp_spec,
                                                               self.exp_spec.print_metric_every_what_epoch)

        """ ---- Setup run dir name ---- """
        self.this_run_dir = setup_commented_run_dir_str(self.exp_spec, self.agent_root_dir)

        """ ---- Create run dir & setup file writer for TensorBoard ---- """
        self.writer = tf_cv1.summary.FileWriter(self.this_run_dir, tf_cv1.get_default_graph())

        """ ---- Log experiment spec in run directory ---- """
        try:
            with open("{}/config.txt".format(self.this_run_dir), "w") as f:
                f.write(self.exp_spec.__repr__())
        except IOError as e:
            raise IOError("The config file cannot be saved in the run directory!") from e

        """ ---- Start training agent ---- """
        for epoch in self._training_epoch_generator(consol_print_learning_stats, render_env):
            (epoch, epoch_loss, batch_average_trjs_return, batch_average_trjs_lenght) = epoch

        """ ---- Teardown ---- """
        consol_print_learning_stats.print_experiment_stats(print_plot=self.exp_spec.show_plot)

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
        raise NotImplementedError

    def _render_trajectory_on_condition(self, epoch, render_env, trj_collected_in_that_epoch):
        if (render_env and (epoch % self.exp_spec.render_env_every_What_epoch == 0)
                and trj_collected_in_that_epoch == 0):
            self.playground.env.env.render()  # keep environment rendering turned OFF during unit test

    def _save_learned_model(self, batch_average_trjs_return: float, epoch, sess: tf_cv1.Session) -> None:
        if batch_average_trjs_return >= float(self.exp_spec.expected_reward_goal):
            print("\n\n    ::  {} batch avg return reached".format(batch_average_trjs_return))
            self._save_checkpoint(epoch, sess, self.exp_spec.algo_name, batch_average_trjs_return)

    def _save_checkpoint(self, epoch: int, sess: tf_cv1.Session, algo_name: str, batch_avrj_trjs_return) -> None:
        cleaned_name = algo_name.replace(" ", "_")
        self.saver.save(sess, '{}/checkpoint/{}_agent-{}'.format(self.this_run_dir, cleaned_name,
                                                                 int(batch_avrj_trjs_return)),
                        global_step=epoch)
        print("    ↳ {} network parameters were saved\n".format(algo_name))

    def play(self, run_name: str, max_trajectories=20, record=False) -> None:
        # todo:implement --> hparam loading functionality : Required to make experiment management clean and bug free
        #
        with tf_cv1.Session() as sess:

            # note: Check my past implementation as ref
            #   |       - Store: Deep_RL/DQN/DQN_OpenAI_Baseline/FalconX_env/train_2_DQN_OpenAi_baseline_FalconX.py
            #   |       - Load: Deep_RL/DQN/DQN_OpenAI_Baseline/FalconX_env/enjoy_2_DQN_OpenAI_baseline_FalconX.py
            #
            #   Loading code::
            #   ''' import ast
            #          try:
            #              config = None
            #              with open("{}/{}/config.txt".format(EXPERIMENT_ROOT, run_directory), "r") as f:
            #                  s = f.readline()
            #                  config = ast.literal_eval(s)
            #          except IOError as e:
            #              raise IOError("The config file cannot be found in the run directory!") from e
            #   '''

            self.load_selected_trained_agent(sess, run_name)

            print(":: Agent player >>> LOCK & LOAD\n"
                  "           ↳ Execute {} run\n           ↳ Test run={}".format(max_trajectories,
                                                                                 self.exp_spec.isTestRun)
                  )

            print(":: Running trajectory >>> ", end=" ", flush=True)
            for run in range(max_trajectories):
                print(run + 1, end=" ", flush=True)

                obs = self.playground.env.reset()  # <-- fetch initial observation
                if record:
                    agent_name = self.exp_spec.algo_name + self.exp_spec.paramameter_set_name
                    agent_name = agent_name.replace(" ", "_")
                    recorder = VideoRecorder(self.playground.env, '../video/{}_{}.mp4'.format(agent_name, run))

                """ ---- Simulator: time-steps ---- """
                while True:
    
                    if record:
                        recorder.capture_frame()
                    elif not self.exp_spec.isTestRun:  # keep environment rendering turned OFF during unit test
                        self.playground.env.render()
    
                    """ ---- Agent: act in the environment ---- """
                    act_t = self._select_action_given_policy(sess, obs)
                    obs_prime, reward, done, _ = self.playground.env.step(act_t)

                    obs = obs_prime  # <-- (!)

                    if done:
                        break

            if record:
                recorder.close()

            print("END")

    def _select_action_given_policy(self, sess: tf_cv1.Session, obs_t: Any, **kwargs):
        obs_t_flat = bloc.format_single_step_observation(obs_t)
        act_t = sess.run(self.policy_pi,
                         feed_dict={self.obs_t_ph: obs_t_flat})
        act_t = blocAndTools.tensorflowbloc.to_scalar(act_t)
        return act_t

    def load_selected_trained_agent(self, sess: tf_cv1.Session, run_name: str):
        # (nice to have) todo:implement --> capability to load the last trained agent:
        path = "{}/saved_training".format(self.agent_root_dir)
        self.saver.restore(sess, "{}/{}".format(path, run_name))

    def _set_random_seed(self):
        if self.exp_spec.random_seed == 0:
            print(":: Random seed control is turned OFF")
        else:
            tf_cv1.random.set_random_seed(self.exp_spec.random_seed)
            np.random.seed(self.exp_spec.random_seed)
            print(":: Random seed control is turned ON")

    def __del__(self):
    
        # (nice to have) todo:assessment --> is it linked to the 'experiment_runner' rerun error (fail at second rerun)
        tf_cv1.reset_default_graph()
    
        self.playground.env.env.close()
        print(":: Agent >>> CLOSED")
