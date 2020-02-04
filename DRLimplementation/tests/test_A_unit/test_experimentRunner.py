# coding=utf-8
from argparse import Namespace

import pytest
import argparse
import tensorflow as tf

tf_cv1 = tf.compat.v1  # shortcut

from blocAndTools.experiment_runner import (
    run_experiment, _warmup_agent_for_playing, experiment_closing_message,
    experiment_start_message, play_agent,
    )
from ActorCritic.BatchActorCriticAgent import BatchActorCriticAgent
from blocAndTools.rl_vocabulary import TargetType, NetworkType
from blocAndTools import buildingbloc as bloc

test_AC_hparam = {
    'paramameter_set_name':           'Batch-AAC',
    'rerun_tag':                      'TEST-RUN-H',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'TestSpec',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'LunarLander-v2',
    'batch_size_in_ts':               300,
    'expected_reward_goal':           200,
    'max_epoch':                      3,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    'learning_rate':                  3e-4,
    'critic_learning_rate':           1e-3,
    'actor_lr_decay_rate':            1,  # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,  # set to 1 to swith OFF scheduler
    'critique_loop_len':              5,
    'theta_nn_h_layer_topo':          (2,),
    # 'theta_nn_h_layer_topo':          [(4, 4), (6, 6)],
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'isTestRun':                      True,
    'show_plot':                      False,
    }

test_AC_searchSet_hparam = {
    'paramameter_set_name':           'Batch-AAC',
    'rerun_tag':                      'TEST-RUN-H',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'TestSpec',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'LunarLander-v2',
    'batch_size_in_ts':               300,
    'expected_reward_goal':           200,
    'max_epoch':                      3,
    'discounted_reward_to_go':        True,
    # 'discout_factor':                 0.999,
    'discout_factor':                 [0.999, 0.91],
    'learning_rate':                  3e-4,
    'critic_learning_rate':           1e-3,
    'actor_lr_decay_rate':            1,  # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,  # set to 1 to swith OFF scheduler
    'critique_loop_len':              5,
    'theta_nn_h_layer_topo':          (2,),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'isTestRun':                      True,
    'show_plot':                      False,
    }


@pytest.fixture
def command_line_argument():
    parser = argparse.ArgumentParser(description="Pytest command line emulation")
    
    parser.add_argument('-rer', '--rerun', type=int, default=1,
                        help='Rerun training experiment with the same spec r time (default=1)')
    
    parser.add_argument('--renderTraining', action='store_true',
                        help='(Training option) Watch the agent execute trajectories while he is on traning duty')
    
    parser.add_argument('-d', '--discounted', default=None, type=bool,
                        help='(Training option) Force training execution with discounted reward-to-go')
    
    parser.add_argument('--playLunar', action='store_true',
                        help='Play on LunarLander-v2 a Batch Actor-Critic agent trained with Bootstrap target on a '
                             'split network')
    
    parser.add_argument('--play_for', type=int, default=10,
                        help='(Playing option) Max playing trajectory, default=20')
    
    parser.add_argument('--record', action='store_true',
                        help='(Playing option) Record trained agent playing in a environment')
    
    parser.add_argument('--testRun', action='store_true', help='Flag for automated continuous integration test')
    
    yield parser.parse_args


def test_full_run(command_line_argument):
    arg_parser = command_line_argument
    parser = argparse.ArgumentParser()
    args: Namespace = arg_parser(['--testRun'])
    
    experiment_start_message(consol_width=90, rerun_nb=args.rerun)
    hparam, key, values_search_set = run_experiment(test_AC_hparam, args,
                                                    test_AC_hparam, rerun_nb=1)
    experiment_closing_message(hparam, args.rerun, key, values_search_set, consol_width=90)


def test_full_with_searchSet_run(command_line_argument):
    arg_parser = command_line_argument
    parser = argparse.ArgumentParser()
    args: Namespace = arg_parser(['--testRun'])
    
    experiment_start_message(consol_width=90, rerun_nb=args.rerun)
    hparam, key, values_search_set = run_experiment(test_AC_searchSet_hparam, args,
                                                    test_AC_searchSet_hparam, rerun_nb=1)
    experiment_closing_message(hparam, args.rerun, key, values_search_set, consol_width=90)


def test_start_message(command_line_argument):
    arg_parser = command_line_argument
    parser = argparse.ArgumentParser()
    args = arg_parser(['--testRun'])
    
    experiment_start_message(consol_width=90, rerun_nb=args.rerun)


def test_end_message(command_line_argument):
    arg_parser = command_line_argument
    parser = argparse.ArgumentParser()
    args = arg_parser(['--testRun'])
    
    hparam, key, values_search_set = run_experiment(test_AC_hparam, args,
                                                    test_AC_hparam, rerun_nb=1)
    experiment_closing_message(initial_hparam=test_AC_hparam, nb_of_rerun=1, key='theta_nn_h_layer_topo',
                               values_search_set=[(4, 4), (6, 6)], consol_width=90)
