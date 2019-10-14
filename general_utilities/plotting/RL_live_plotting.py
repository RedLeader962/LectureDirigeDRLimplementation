#!/usr/bin/env python
import time

import numpy as np
from os import environ
import copy

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

plt.rcParams['toolbar'] = 'None'  # mute matplotlib toolbar
matplotlib.rcParams['backend'] = 'macosx'
# matplotlib.rcParams['backend'] = 'Qt5Agg'

BACKGROUND_COLOR = '#3C3F41'
FONT_SIZE = 9
# GRID_ALPHA = 0.1
GRID_COLOR = '#50565A'
THEME_WHITE = 'w'
THEME_OLIVE = 'olivedrab'
THEME_GOLD = 'gold'
THEME_RED = 'r'
THEME_BLUE = 'dodgerblue'


X_LIM_MIN = 320


class TrainingHistoryDataclass(object):
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.num_episodes_ = int()
        self.total_t_ = int()
        self.observation_space_shape = int()

        # Per episode data
        self._step_per_episodes = int()
        self.episodes_reward_history = [0.0, 0.0]
        self.step_lenght_history = [0]
        self.exploration_rates = [100]
        self.mean_100ep_reward = [0]

        # Per step data (max accessed via 'env._max_episode_steps')
        self.reward_delta_per_step = list()
        self.cumulated_reward_per_step = list()
        self.observations = list()

        # Keyframe
        self.episodes_start = [0]
        self.curent_episode_start = 0
        self.previous_episode_start = int()
        # self.previous_episode_2_start = int()
        # self.previous_episode_3_start = int()


class ActorTrainingPlotter(object):
    """
    Training curve plotter for reinforcement learning actor

    Note:
        - Inspired by OpenAi Lab class Grapher by kengz and lgraesser
        - source: https://github.com/kengz/openai_lab/blob/master/rl/analytics.py
    """

    def __init__(self, dataObject, config, run_directory=None, VCS_tag=None):
        assert isinstance(dataObject, TrainingHistoryDataclass)

        plt.style.use('dark_background')

        self.VCS_tag = VCS_tag
        self.dataObject = dataObject
        self.config = copy.copy(config)
        self.graph_filename = self.dataObject.experiment_id
        self.run_dir = run_directory
        self.subgraphs = {}

        self.figure = plt.figure(facecolor=BACKGROUND_COLOR, figsize=(10, 13.8))
        self.figure.suptitle(self.dataObject.experiment_id, fontweight="bold")
        self.init_figure()

    def init_figure(self):

        plot_gs = gs.GridSpec(6, 3)

        # ---- Header ---------------------------------------------------------------

        header_text = self.figure.text(x=0.04, y=0.95, s='')
        self.subgraphs['header text'] = header_text

        # --- graph 1 ---------------------------------------------------------------
        ax1 = self.figure.add_subplot(
            plot_gs[0, :],
            frame_on=False,
            ylabel='total rewards',
            xlabel="episode")
        ax1.set_title('\n\n\n\nTotal rewards per episode', fontweight="bold", size=FONT_SIZE)
        ax1.grid(True, color=GRID_COLOR)
        p1, = ax1.plot([], [], THEME_WHITE, linewidth=0.6, alpha=0.3)
        self.subgraphs['total rewards'] = (ax1, p1)

        p1mean, = ax1.plot([], [], THEME_WHITE, linewidth=1.5)
        self.subgraphs['mean rewards'] = p1mean

        ax1e = ax1.twinx()
        ax1e.set_ylabel('exploration rate').set_color(THEME_OLIVE)
        ax1e.set_frame_on(False)
        ax1e.grid(False)
        p1e, = ax1e.plot([], [], THEME_OLIVE)
        self.subgraphs['e'] = (ax1e, p1e)

        # --- graph 2 ---------------------------------------------------------------
        ax2 = self.figure.add_subplot(
            plot_gs[1, :],
            frame_on=False,
            ylabel='Step',
            xlabel="Episode")
        ax2.set_title('Step per episode', fontweight="bold", size=FONT_SIZE)
        ax2.grid(True, color=GRID_COLOR)
        p2, = ax2.plot([], [], THEME_GOLD, linewidth=0.6)
        self.subgraphs['Step per episode'] = (ax2, p2)

        # --- graph 3 ---------------------------------------------------------------
        ax3 = self.figure.add_subplot(
            plot_gs[2, :],
            frame_on=False,
            ylabel='Cumulated rewards',
            xlabel="step")
        ax3.set_title('Cumulated reward (Current & previous episode)', fontweight="bold", size=FONT_SIZE)
        ax3.grid(True, color=GRID_COLOR)
        p3a, = ax3.plot([], [], THEME_RED, linewidth=2)
        self.subgraphs['Cumulated rewards Current episode'] = (ax3, p3a)

        # graph 3b
        p3b, = ax3.plot([], [], THEME_RED, alpha=0.4, linewidth=3)
        self.subgraphs['Cumulated rewards Previous episode'] = p3b

        # # graph 3c
        # p3c, = ax3.plot([], [], THEME_RED, alpha=0.4, linewidth=3)
        # self.subgraphs['Cumulated rewards Previous episode 2'] = p3c
        #
        # # graph 3d
        # p3d, = ax3.plot([], [], THEME_RED, alpha=0.2, linewidth=4)
        # self.subgraphs['Cumulated rewards Previous episode 3'] = p3d

        # --- graph 4 ---------------------------------------------------------------
        ax4 = self.figure.add_subplot(
            plot_gs[3, :],
            frame_on=False,
            ylabel='Reward delta',
            xlabel="step")
        ax4.set_title('Reward delta (Current & previous episode)', fontweight="bold", size=FONT_SIZE)
        ax4.grid(True, color=GRID_COLOR)
        p4a, = ax4.plot([], [], THEME_BLUE, linewidth=2)
        self.subgraphs['Reward delta Current episode'] = (ax4, p4a)

        # graph 4b
        p4b, = ax4.plot([], [], THEME_BLUE, alpha=0.4, linewidth=3)
        self.subgraphs['Reward delta Previous episode'] = p4b

        # # graph 4c
        # p4c, = ax4.plot([], [], THEME_BLUE, alpha=0.4, linewidth=3)
        # self.subgraphs['Reward delta Previous episode 2'] = p4c
        #
        # # graph 4d
        # p4d, = ax4.plot([], [], THEME_BLUE, alpha=0.2, linewidth=4)
        # self.subgraphs['Reward delta Previous episode 3'] = p4d

        # --- config quadrant -------------------------------------------------------
        priority_field = ["result_folder", "run_dir", "experiment_file_name", "environment_name", "VCS_tag"]
        config_fields = ["Experiment configuration = {\n"]
        for each in priority_field:
            value = self.config.pop(each)
            config_fields.append("     {}:\n        {}\n".format(each, value))
        config_fields.append("\n")
        for key, value in self.config.items():
            no_print = ["experiment_name", "callback_class", "print_freq", "checkpoint_freq",
                        "visualize_episode_every", "render_and_plot_data", "state variable label"]
            if key not in no_print:
                config_fields.append("     {}: {}\n".format(key, value))

        config_fields.append("}")
        self.figure.text(x=0.022, y=0.036, s="".join(config_fields), size=6.5)

        # --- Observation quadrant --------------------------------------------------

        # <-- Todo en cours
        assert len(self.config['state variable label']) == self.config['observation_space_shape'], \
            "The number of 'state variable label' in the config file does not match the observation space"

        ax5 = self.figure.add_subplot(
            plot_gs[4:5, 1:3],
            frame_on=False,
            ylabel='',
            xlabel="Observations"
        )
        p5_pos = ax5.bar(range(self.config['observation_space_shape']),
                         np.zeros(self.config['observation_space_shape']),
                         color="gray", tick_label=self.config['state variable label'],
                         # animated=True,
                         # antialiased=False
                         )

        ax5.tick_params(axis='x', labelsize=5.5)

        self.subgraphs['Observations'] = (ax5, p5_pos)

        # --- Q-Values quadrant -----------------------------------------------------

        # ---------------------------------------------------------------------------

        speed_text = self.figure.text(x=0.4, y=0.1, s='')
        self.subgraphs['speed text'] = speed_text

        # ---------------------------------------------------------------------------

        plt.tight_layout()  # auto-fix spacing
        plt.ion()  # for live plot

    def plot(self, render_observations=False):
        '''do live plotting
        :param render_observations:
        '''

        # ---- Header ---------------------------------------------------------------

        header_text = self.subgraphs['header text']
        header_text.set_text("Episode: {},  "
                             "Total step: {}".format(self.dataObject.num_episodes_,
                                                     self.dataObject.total_t_))

        # --- graph 1 ---------------------------------------------------------------
        ax1, p1 = self.subgraphs['total rewards']
        p1.set_ydata(self.dataObject.episodes_reward_history)
        p1.set_xdata(np.arange(len(p1.get_ydata())))

        p1mean = self.subgraphs['mean rewards']
        mean_100ep_reward = self.dataObject.mean_100ep_reward
        p1mean_idx = list(range(-1, self.dataObject.num_episodes_, 100))
        p1mean_idx[0] = 0

        if self.dataObject.num_episodes_ % 100 == 0:
            mean_100ep_reward = mean_100ep_reward[:-1]
        else:
            p1mean_idx.append(self.dataObject.num_episodes_ - 1)

        p1mean.set_ydata(mean_100ep_reward)
        p1mean.set_xdata(p1mean_idx)

        ax1.relim()
        ax1.autoscale_view(tight=True, scalex=True, scaley=True)

        ax1e, p1e = self.subgraphs['e']
        p1e.set_ydata(self.dataObject.exploration_rates)
        p1e.set_xdata(np.arange(len(p1e.get_ydata())))
        ax1e.relim()
        ax1e.autoscale_view(tight=True, scalex=True, scaley=True)

        # --- graph 2 ---------------------------------------------------------------
        ax2, p2 = self.subgraphs['Step per episode']
        p2.set_ydata(self.dataObject.step_lenght_history)
        p2.set_xdata(np.arange(len(p2.get_ydata())))
        ax2.relim()
        ax2.autoscale_view(tight=True, scalex=True, scaley=True)

        # --- graph 3 ---------------------------------------------------------------
        ax3, p3 = self.subgraphs['Cumulated rewards Current episode']
        p3.set_ydata(self.dataObject.cumulated_reward_per_step[
                     self.dataObject.curent_episode_start:self.dataObject.total_t_ + 1])
        p3a_len = len(p3.get_ydata())
        p3.set_xdata(np.arange(p3a_len))

        p3b = self.subgraphs['Cumulated rewards Previous episode']
        p3b.set_ydata(self.dataObject.cumulated_reward_per_step[
                     self.dataObject.previous_episode_start:self.dataObject.curent_episode_start])
        p3b_len = len(p3b.get_ydata())
        p3b.set_xdata(np.arange(p3b_len))

        # Adjust scale with respect to each plot and min limit
        ax3.relim()
        ax3.autoscale_view(tight=True, scalex=False, scaley=True)
        # p3_max_len = max([p3a_len, p3b_len, p3c_len, p3d_len])
        p3_max_len = max([p3a_len, p3b_len])
        if p3_max_len < X_LIM_MIN:
            ax3.set_xlim((0, X_LIM_MIN))
        else:
            ax3.set_xlim(0, p3_max_len)

        # --- graph 4 ---------------------------------------------------------------
        """ Fastest version using precomputed max and len from ax3 """
        ax4, p4a = self.subgraphs['Reward delta Current episode']
        p4a.set_ydata(self.dataObject.reward_delta_per_step[
                      self.dataObject.curent_episode_start:self.dataObject.total_t_ + 1])
        p4a.set_xdata(np.arange(p3a_len))

        p4b = self.subgraphs['Reward delta Previous episode']
        p4b.set_ydata(self.dataObject.reward_delta_per_step[
                      self.dataObject.previous_episode_start:self.dataObject.curent_episode_start])
        p4b.set_xdata(np.arange(p3b_len))

        # Adjust scale with respect to each plot and min limit
        ax4.relim()
        ax4.autoscale_view(tight=True, scalex=False, scaley=True)
        if p3_max_len < X_LIM_MIN:
            ax4.set_xlim((0, X_LIM_MIN))
        else:
            ax4.set_xlim((0, p3_max_len))

        # --- Observation quadrant --------------------------------------------------

        if render_observations:
            ax5, p5_pos = self.subgraphs['Observations']
            max_height = 1
            for i, rectangle in enumerate(p5_pos.patches):
                height = self.dataObject.observations[-1][i]
                rectangle.set_height(height)

                if abs(height) > max_height:
                    max_height = height

            if max_height > 1:
                ax5.set_ylim((-max_height, max_height))
            else:
                ax5.set_ylim((-1, 1))

            # ax5.relim()
            # ax5.autoscale_view(tight=True, scalex=False, scaley=True)

            # print("dir(p4a):", dir(p4a))
            # print("dir(p5_pos):", dir(p5_pos))
            # print("dir(ax5):", dir(ax5))
            # time.sleep(60)

        # --- Q-Values quadrant -----------------------------------------------------

        # ---------------------------------------------------------------------------

        if render_observations:
            speed_text = self.subgraphs['speed text']
            x_vel = self.dataObject.observations[-1][5]
            y_vel = self.dataObject.observations[-1][6]
            speed_text.set_text("Speed: {}".format(np.linalg.norm(np.array([x_vel, y_vel]))))

        # ---------------------------------------------------------------------------

        plt.draw()
        plt.pause(1e-30)

    def save(self):
        '''save graph to filename'''
        if self.run_dir:
            file_name = self.graph_filename.replace(" ", "_")
            self.figure.savefig("{}/{}.png".format(self.run_dir, file_name), facecolor='#3C3F41')

    def clear(self):
        self.save()
        plt.close()
