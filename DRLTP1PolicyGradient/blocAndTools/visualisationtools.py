#!/usr/bin/env python
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from blocAndTools.buildingbloc import ExperimentSpec


class CycleIndexer(object):
    def __init__(self, cycle_lenght: int = 10):
        self.i = 0
        self.j = cycle_lenght
        self.cycle_lenght = cycle_lenght

    def __next__(self):
        if self.i == self.cycle_lenght:
            self.reset()
        else:
            self.i += 1
            self.j -= 1

        return self.i, self.j

    def reset(self):
        self.i = 0
        self.j = self.cycle_lenght


class ConsolPrintLearningStats(object):
    def __init__(self, experiment_spec, print_metric_every_what_epoch=5, consol_span=90):
        self.cycle_indexer = CycleIndexer(cycle_lenght=10)
        self.epoch = 0
        self.trj = 0
        self.the_trajectory_return = None
        self.timestep = None
        self.number_of_trj_collected = None
        self.total_timestep_collected = None
        self.epoch_loss = None
        self.average_trjs_return = None
        self.average_trjs_lenght = None
        self.print_metric_every = print_metric_every_what_epoch
        self.span = consol_span

        self.loss_smoothing_buffer = 0.0
        self.return_smoothing_buffer = 0.0
        self.lenght_smoothing_buffer = 0.0

        self.last_batch_lost = 0.0
        self.last_batch_return = 0

        self.collected_experiment_stats = {
            'smoothed_average_return': [],
            'smoothed_average_peusdo_loss': [],
            'smoothed_average_lenght': [],
        }

        self.exp_spec = experiment_spec

    def _assert_all_property_are_feed(self) -> bool:
        if ((self.number_of_trj_collected is not None) and (self.total_timestep_collected is not None) and
                (self.epoch_loss is not None) and (self.average_trjs_return is not None) and
                (self.average_trjs_lenght is not None)):
            return True

    def start_the_crazy_experiment(self, message=("3", "2", "1", "READY")) -> None:
        print("\n\n")
        self.anim_line(start_anim_at_a_new_line=True, keep_cursor_at_same_line_on_exit=False)
        self.anim_line(nb_of_cycle=1, keep_cursor_at_same_line_on_exit=True)

        for m in message:
            print("\r{:^{span}}".format(m, span=self.span), end="", flush=True)
            time.sleep(0.1)
        # print("\r{:^{span}}".format("?", span=self.span), end="", flush=True)
        # time.sleep(0.01)
        print(
            "\r{:=<{span}}\r".format("=== EXPERIMENT START ", span=self.span), end="", flush=True)
        return None

    def print_experiment_stats(self):
        print("\n\n\n{:^{span}}".format("Experiment stoped", span=self.span))
        stats_str = "Collected {} trajectories over {} epoch".format(self.trj, self.epoch)
        print("{:^{span}}".format(
            stats_str, span=self.span), end="\n\n", flush=True)
        print("{:=>{span}}".format(" EXPERIMENT END ===", span=self.span), end="\n", flush=True)
        self.anim_line(caracter=">", nb_of_cycle=1, start_anim_at_a_new_line=False)
        self.anim_line(caracter="<", nb_of_cycle=1, keep_cursor_at_same_line_on_exit=False)

        ultra_basic_ploter(self.collected_experiment_stats['smoothed_average_return'],
                           self.collected_experiment_stats['smoothed_average_peusdo_loss'],
                           self.collected_experiment_stats['smoothed_average_lenght'], self.exp_spec,
                           self.print_metric_every)

        return None

    def anim_line(self, caracter=">", nb_of_cycle=2,
                  start_anim_at_a_new_line=False,
                  keep_cursor_at_same_line_on_exit=True):
        if start_anim_at_a_new_line:
            print("\n")

        for c in range(nb_of_cycle):
            for i in range(self.span):
                print(caracter, end="", flush=True)
                time.sleep(0.0005)

            if (c == nb_of_cycle -1) and not keep_cursor_at_same_line_on_exit:
                print("\n", end="", flush=True)
            else:
                print("\r", end="", flush=True)

    def next_glorious_epoch(self) -> None:
        self.epoch += 1

        if (self.epoch - 1) % self.print_metric_every == 0:
            print("\n\n{:-<{span}}\n".format(":: Epoch ", span=self.span), end="", flush=True)
        return None

    def next_glorious_trajectory(self) -> (int, int):
        """
        Incremente the cycle_index_i and decremente the cycle_index_j.
        Both index are returned for convience.

        :return: cycle_index_i, cycle_index_j
        :rtype: int, int
        """
        self.trj += 1
        return self.cycle_indexer.__next__()

    def epoch_training_stat(self, epoch_loss, epoch_average_trjs_return, epoch_average_trjs_lenght,
                            number_of_trj_collected, total_timestep_collected) -> None:
        """
        Call after a traing update as been done, at the end of a epoch.
        """
        self.epoch_loss = epoch_loss
        self.average_trjs_return = epoch_average_trjs_return
        self.average_trjs_lenght = epoch_average_trjs_lenght
        self.number_of_trj_collected = number_of_trj_collected
        self.total_timestep_collected = total_timestep_collected

        self.return_smoothing_buffer += epoch_average_trjs_return
        self.loss_smoothing_buffer += epoch_loss
        self.lenght_smoothing_buffer += epoch_average_trjs_lenght

        if (self.epoch) % self.print_metric_every == 0:
            smoothed_batch_loss = self.loss_smoothing_buffer / self.print_metric_every
            smoothed_return = self.return_smoothing_buffer / self.print_metric_every
            smoothed_lenght = self.lenght_smoothing_buffer / self.print_metric_every
            print(
                "\r\t ↳ {:^3}".format(self.epoch),
                ":: Collected {} trajectories for a total of {} timestep.".format(
                    self.number_of_trj_collected, self.total_timestep_collected),
                "\n\t\t↳ pseudo loss: {:>6.3f} ".format(self.epoch_loss),
                "| average trj return: {:>6.3f} | average trj lenght: {:>6.3f}".format(
                    self.average_trjs_return, self.average_trjs_lenght),
                end="\n", flush=True)

            print("\n\t\t\t\t\t\tAverage return over the past {} epoch: {:>6.3f}".format(
                self.print_metric_every, smoothed_return))
            if abs(smoothed_return) < abs(self.last_batch_return):
                print("\t\t\t\t\t\t\t↳ is lowering ⬊", end="", flush=True)
            elif abs(smoothed_return) > abs(self.last_batch_return):
                print("\t\t\t\t\t\t\t↳ is rising ⬈  ...  goooood :)", end="", flush=True)

            self.collected_experiment_stats['smoothed_average_peusdo_loss'].append(smoothed_batch_loss)
            self.collected_experiment_stats['smoothed_average_return'].append(smoothed_return)
            self.collected_experiment_stats['smoothed_average_lenght'].append(smoothed_lenght)

            # reset smooting buffer
            self.loss_smoothing_buffer = 0.0
            self.return_smoothing_buffer = 0.0
            self.lenght_smoothing_buffer = 0.0

            self.last_batch_lost = smoothed_batch_loss
            self.last_batch_return = smoothed_return

            # if (self.epoch) % (self.print_metric_every * 10) == 0:
            #     ultra_basic_ploter(self.collected_experiment_stats['smoothed_average_return'],
            #                        self.collected_experiment_stats['smoothed_average_peusdo_loss'],
            #                        self.collected_experiment_stats['smoothed_average_lenght'], self.exp_spec,
            #                        self.print_metric_every)

        return None

    def trajectory_training_stat(self, the_trajectory_return, timestep) -> None:
        """
        Print formated learning metric & stat

        :param the_trajectory_return:
        :type the_trajectory_return: float
        :param timestep:
        :type timestep: int
        :return:
        :rtype: None
        """
        print("\r\t ↳ {:^3} :: Trajectory {:>4}  ".format(self.epoch, self.trj),
              ">"*self.cycle_indexer.i, " "*self.cycle_indexer.j,
              "  got return {:>8.2f}   after  {:>4}  timesteps".format(
                  the_trajectory_return, timestep),
              sep='', end='', flush=True)

        self.the_trajectory_return = the_trajectory_return
        self.timestep = timestep
        return None


class UltraBasicLivePloter(object):
    # (Ice-Boxed) todo:implement --> live ploting for trainig: (!) alternative --> use TensorBoard
    def __init__(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        self.fig = fig
        self.ax = ax
        plt.xlabel('Epoch')
        ax.grid(True)
        ax.legend(loc='best')

        plt.ion()  # for live plot

    def draw(self, epoch_average_return: list, epoch_average_loss: list) -> None:
        x_axes = range(0, len(epoch_average_return))
        self.ax.plot(x_axes, epoch_average_return, label='Average Return')
        self.ax.plot(x_axes, epoch_average_loss, label='Average loss')

        plt.draw()
        return None


def ultra_basic_ploter(epoch_average_return: list, epoch_average_loss: list, epoch_average_lenght: list,
                       experiment_spec: ExperimentSpec, metric_computed_every_what_epoch: int) -> None:

    fig, ax = plt.subplots(figsize=(8, 6))

    x_axes = np.arange(0, len(epoch_average_return)) * metric_computed_every_what_epoch
    ax.plot(x_axes, epoch_average_return, label='Average Return')
    ax.plot(x_axes, epoch_average_loss, label='Average loss')
    ax.plot(x_axes, epoch_average_lenght, label='Average lenght')

    plt.xlabel('Epoch')

    now = datetime.now()
    ax.set_title("Experiment {} finished at {}:{} {}".format(experiment_spec.paramameter_set_name,
                                                       now.hour, now.minute, now.date()), fontsize='xx-large')


    ax.grid(True)
    ax.legend(loc='best')

    plt.show()
