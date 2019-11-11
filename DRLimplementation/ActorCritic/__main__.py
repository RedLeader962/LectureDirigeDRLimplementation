# coding=utf-8
"""
Invoke Actor-Critic agent using

    To play:
        `python -m ActorCritic --play [--play_for] [--help] [--testRun]`

    To train:
        `python -m ActorCritic --trainExperimentSpecification [--rerun] [--renderTraining] [--discounted] [--help] [--testRun]`

        Choose `--trainExperimentSpecification` between the following:
        - CartPole-v0 environment:
            [--trainSplitMC]: Train a Batch Actor-Critic agent with Monte Carlo TD target
            [--trainSplitBootstrap]: Train a Batch Actor-Critic agent with bootstrap estimate TD target
            [--trainSharedBootstrap]: Train a Batch Actor-Critic agent with shared network
            [--trainOnlineSplit]: Train a Online Actor-Critic agent with split network
            [--trainOnlineSplit3layer]: Train a Online Actor-Critic agent with split network
            [--trainOnlineShared3layer]: Train a Online Actor-Critic agent with Shared network
            [--trainOnlineSplitTwoInputAdvantage]: Train a Online Actor-Critic agent with split Two input Advantage network
        - LunarLander-v2 environment:
            [--trainOnlineLunarLander]: Train on LunarLander a Online Actor-Critic agent with split Two input Advantage network
            [--trainBatchLunarLander]: Train on LunarLander a Batch Actor-Critic agent

Note on TensorBoard usage:
    Start TensorBoard in terminal:
        cd DRLimplementation   (!)
        tensorboard --logdir=ActorCritic/graph

    In browser, go to:
        http://0.0.0.0:6006/


"""
import argparse
import tensorflow as tf

from ActorCritic.BatchActorCriticAgent import BatchActorCriticAgent
from ActorCritic.OnlineActorCriticAgent import OnlineActorCriticAgent
from ActorCritic.OnlineTwoInputAdvantageActorCriticAgent import OnlineTwoInputAdvantageActorCriticAgent
from blocAndTools.buildingbloc import ExperimentSpec
from blocAndTools.experiment_runner import (run_experiment, _warmup_agent_for_playing, experiment_closing_message,
                                            experiment_start_message, play_agent, )
from blocAndTools.rl_vocabulary import TargetType, NetworkType

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                                                                                                                   *
# *                                   Advantage Actor-Critic (batch architecture)                                     *
# *                                                                                                                   *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# Note: About Gamma value (aka the discout factor)
#   |    Big difference between 0.9 and 0.999.
#   |    Also you need to take into account the experiment average number of step per episode
#   |
#   |        Example with experiment average step of 100:
#   |           0.9^100 = 0.000026 vs 0.99^100 = 0.366003 vs 0.999^100 = 0.904792
#   |
#   |    Meaning a agent with Gamma=0.9 is short-sighted and one with Gamma=0.9999 is farsighted or clairvoyant

BATCH_AAC_MonteCarlo_SPLIT_net_hparam = {
    'paramameter_set_name':           'Batch-AAC-Split-nn',
    'rerun_tag':                      'BMCSPL-B-G2',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'MonteCarlo-target',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               4000,
    'max_epoch':                      30,
    # 'discounted_reward_to_go':        [True, False],
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.99,
    'learning_rate':                  1e-2,
    'critic_learning_rate':           1e-2,
    'actor_lr_decay_rate':            1,  # 9e-1,
    'critic_lr_decay_rate':           1,  # 9e-1,
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (32, 32),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           ''
    # 'rerun_tag':                      'BMCSPL-C-LR-SCHEDULE',
    # 'learning_rate':                  5e-2,
    # 'critic_learning_rate':           5e-2,
    # 'actor_lr_decay_rate':            1e-1,
    # 'critic_lr_decay_rate':           1e-1,
    # 'note':                           'Learning rate scheduler work very well with a small decay rate'
    }


BATCH_AAC_bootstrap_SPLIT_net_hparam = {
    'paramameter_set_name':           'Batch-AAC-Split-nn',
    # 'rerun_tag':                      'BBSPL-A',
    'rerun_tag':                      'BBSPL-F-ActorAdam-HE',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'Discounted-Bootstrap-target Farsighted',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.Bootstrap,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               3000,
    # 'max_epoch':                      30,
    'max_epoch':                      50,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.9999,
    'learning_rate':                  1e-2,
    'critic_learning_rate':           1e-3,
    'actor_lr_decay_rate':            1,    # 9e-1,
    'critic_lr_decay_rate':           1,    # 9e-1,
    'critique_loop_len':              120,
    'theta_nn_h_layer_topo':          (16, 32, 64),
    # 'theta_nn_h_layer_topo':          (62,),    # <--(!) not learning
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.relu,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           "Both loss have a lot less variance. The algo take more time to converge. relu seams to work better"
    # 'learning_rate':                  3e-2,
    # 'critic_learning_rate':           3e-3,
    }


BATCH_AAC_Bootstrap_SHARED_net_hparam = {
    'paramameter_set_name':           'Batch-AAC-Shared-nn',
    'rerun_tag':                      'BSHA-A',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'Bootstrap-Tiny-Batch-WORKING',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.Bootstrap,
    'Network':                        NetworkType.Shared,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               200,
    'max_epoch':                      400,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-3,
    'critic_learning_rate':           1e-4,
    'actor_lr_decay_rate':            1,                                              # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,                                              # set to 1 to swith OFF scheduler
    'critique_loop_len':              100,
    'theta_nn_h_layer_topo':          (60, 60),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.leaky_relu,  # tf.nn.tanh, tf.nn.relu
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  8,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           ("Fail to learn 4 time out of 5 runs"
                                       "Converge aparently faster."
                                       "Does not learn on large batch! "
                                       "Work only on tiny batch (more or less 1 trajectory)"
                                       "Use small hlayer topo."
                                       "Require small learning rate."
                                       "Extremely sensible to hyper param tuning."
                                       "Can possibly not learn at all on different run with same hparam "
                                       "probably because of unlucky grpah initialisation or unlucky initial state")
    }

ONLINE_AAC_Bootstrap_SPLIT_net_hparam = {
    'paramameter_set_name':           'Online-AAC-Split-nn',
    'rerun_tag':                      'OSPL-A',
    'algo_name':                      'Online ActorCritic',
    'comment':                        'Discounted-Bootstrap-target',
    'AgentType':                      OnlineActorCriticAgent,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               8,
    'stage_size_in_trj':              50,
    'max_epoch':                      45,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-4,
    'critic_learning_rate':           5e-4,
    'actor_lr_decay_rate':            1,                                              # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,                                              # set to 1 to swith OFF scheduler
    'critique_loop_len':              1,
    'theta_nn_h_layer_topo':          (32, 32),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh, tf.nn.leaky_relu
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           ("Working! Difficulte to stabilitse. Very sensible hyperparameter: "
                                       "learning_rate, critic_learning_rate, discout_factor, "
                                       "critique_loop_len and batch_size_in_ts")
    }


ONLINE_AAC_Bootstrap_SPLIT_three_layer_hparam = {
    'paramameter_set_name':           'Online-AAC-Split-nn16-32-256',
    'rerun_tag':                      'OSPL3L-A',
    'algo_name':                      'Online ActorCritic',
    'comment':                        'Discounted-Bootstrap-target',
    'AgentType':                      OnlineActorCriticAgent,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               20,
    'stage_size_in_trj':              50,
    'max_epoch':                      45,
    'discout_factor':                 0.999,
    'learning_rate':                  5e-5,
    'critic_learning_rate':           5e-4,
    'actor_lr_decay_rate':            1,                                              # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,                                              # set to 1 to swith OFF scheduler
    'critique_loop_len':              1,
    'theta_nn_h_layer_topo':          (16, 32, 256),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh, tf.nn.leaky_relu
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           ""
    }


ONLINE_AAC_Bootstrap_SHARED_three_layer_hparam = {
    'paramameter_set_name':           'Online-AAC-Shared-nn16-32-256',
    'rerun_tag':                      'OSHA-A',
    'algo_name':                      'Online ActorCritic',
    'comment':                        'Discounted-Bootstrap-target',
    'AgentType':                      OnlineActorCriticAgent,
    'Network':                        NetworkType.Shared,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               10,
    'stage_size_in_trj':              50,
    'max_epoch':                      45,
    'discout_factor':                 0.95,
    'learning_rate':                  3e-4,
    'critic_learning_rate':           3e-4,
    'actor_lr_decay_rate':            1,                                              # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,                                              # set to 1 to swith OFF scheduler
    'critique_loop_len':              2,
    'theta_nn_h_layer_topo':          (32, 64, 256),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.tanh,  # tf.nn.relu, tf.nn.leaky_relu
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  5,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           "Bigger network better with shared network. Fail to learn 2 time out of 5 runs"
    }

ONLINE_AAC_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam = {
    'paramameter_set_name':           'Online-AAC-Split-TwoInputAdv-nn16-32-256',
    'rerun_tag':                      'OSTWO-H',
    'algo_name':                      'Online ActorCritic',
    'comment':                        'Discounted-Bootstrap-target',
    'AgentType':                      OnlineTwoInputAdvantageActorCriticAgent,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               5,
    'stage_size_in_trj':              50,
    'max_epoch':                      25,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-4,
    'critic_learning_rate':           5e-4,
    'actor_lr_decay_rate':            1,                                              # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,                                              # set to 1 to swith OFF scheduler
    'critique_loop_len':              1,
    'theta_nn_h_layer_topo':          (16, 32, 32),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh, tf.nn.leaky_relu
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  5,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           ""
    }

ONLINE_AAC_LunarLander_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam = {
    'paramameter_set_name':           'Online-AAC-Split-TwoInputAdv-nn62-62',
    'rerun_tag':                      'O-Lunar-B',
    'algo_name':                      'Online ActorCritic',
    'comment':                        'Discounted-Bootstrap-target',
    'AgentType':                      OnlineTwoInputAdvantageActorCriticAgent,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'LunarLander-v2',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               30,
    'stage_size_in_trj':              20,
    'max_epoch':                      65,
    'discout_factor':                 0.99,
    'learning_rate':                  1e-4,
    'critic_learning_rate':           5e-4,
    'actor_lr_decay_rate':            1,                                              # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,                                              # set to 1 to swith OFF scheduler
    'critique_loop_len':              2,
    'theta_nn_h_layer_topo':          (62, 62),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh, tf.nn.leaky_relu
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  5,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           ""
    }


BATCH_AAC_MC_LunarLander_hparam = {
    'rerun_tag':                      'BMC-Lunar-E',
    'paramameter_set_name':           'Batch-AAC-Split-nn',
    'comment':                        '',
    'AgentType':                      BatchActorCriticAgent,
    'Network':                        NetworkType.Split,
    'Target':                         TargetType.MonteCarlo,
    'algo_name':                      'Batch ActorCritic',
    'prefered_environment':           'LunarLander-v2',
    'expected_reward_goal':           195,      # trigger model save on reach
    'batch_size_in_ts':               3000,
    'max_epoch':                      120,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.9999,
    'learning_rate':                        0.01,
    'critic_learning_rate':                 0.001,
    'actor_lr_decay_rate':                  0.01,
    'critic_lr_decay_rate':                 0.01,
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (84, 84),
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'note':                           'Does not learn enough'
    }


# BATCH_AAC_LunarLander_hparam = {
#     'paramameter_set_name':           'Batch-AAC-Split-nn',
#     'rerun_tag':                      'BBOOT-Lunar-K',
#     'algo_name':                      'Batch ActorCritic',
#     'comment':                        'HE lrSchedule Bootstrap-Target LunarLander',
#     'AgentType':                      BatchActorCriticAgent,
#     'Target':                         TargetType.Bootstrap,
#     'Network':                        NetworkType.Split,
#     'prefered_environment':           'LunarLander-v2',
#     'expected_reward_goal':           195,      # trigger model save on reach
#     'batch_size_in_ts':               30000,
#     'max_epoch':                      80,
#     'discounted_reward_to_go':        True,
#     'discout_factor':                 0.9999,
#     'learning_rate':                  1e-2,                                     # BBOOT-Lunar-K
#     'critic_learning_rate':           1e-3,     # (!) best result from [1e-1, 1e-2, 1e-3]
#     'actor_lr_decay_rate':            0.01,
#     'critic_lr_decay_rate':           0.01,
#     'critique_loop_len':              80,
#     'theta_nn_h_layer_topo':          (84, 84),
#     'random_seed':                    0,
#     'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
#     'render_env_every_What_epoch':    5,
#     'print_metric_every_what_epoch':  5,
#     'note':                           'BBOOT-Lunar-K-critic_learning_rate=(0.001) --> Reached avg return ~156 for 30/80 epoch'
#     }


# BATCH_AAC_LunarLander_hparam = {
#     'paramameter_set_name':           'Batch-AAC-Split-nn',
#     'rerun_tag':                      'BBOOT-Lunar-N-REPRODUCED',
#     'algo_name':                      'Batch ActorCritic',
#     'comment':                        'HE lrSchedule Bootstrap-Target LunarLander',
#     'AgentType':                      BatchActorCriticAgent,
#     'Target':                         TargetType.Bootstrap,
#     'Network':                        NetworkType.Split,
#     'prefered_environment':           'LunarLander-v2',
#     'expected_reward_goal':           195,      # trigger model save on reach
#     # 'batch_size_in_ts':               [2500, 5000, 10000, 30000, 60000],
#     'batch_size_in_ts':               2500,     # (!) best result
#     'max_epoch':                      120,
#     # 'random_seed': 42,
#     'discounted_reward_to_go':        True,
#     'discout_factor':                 0.9999,
#     'learning_rate':                        0.01,
#     'critic_learning_rate':                 0.001,
#     'actor_lr_decay_rate':                  0.01,
#     'critic_lr_decay_rate':                 0.01,
#     'critique_loop_len':              80,
#     'theta_nn_h_layer_topo':          (84, 84),
#     'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
#     'theta_output_layers_activation': None,
#     'render_env_every_What_epoch':    5,
#     'print_metric_every_what_epoch':  5,
#     'note':                           'BBOOT-Lunar-N-batch_size_in_ts=2500 --> reached ~200 in 40/120 epoch'
#                                       '(!) CAN NOT REPRODUCE'
#     }


BATCH_AAC_LunarLander_hparam = {
    'rerun_tag':                      'BBOOT-Lunar-T',
    'paramameter_set_name':           'Batch-AAC-Split-nn',
    'comment':                        '',
    'AgentType':                      BatchActorCriticAgent,
    'Network':                        NetworkType.Split,
    'Target':                         TargetType.Bootstrap,
    'algo_name':                      'Batch ActorCritic',
    'prefered_environment':           'LunarLander-v2',
    'expected_reward_goal':           195,      # trigger model save on reach
    'batch_size_in_ts':               4000,
    # 'max_epoch':                      [220, 170, 120],
    'max_epoch':                      220,      # (!) best result but not consistent
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.9999,
    'learning_rate':                        0.01,
    'critic_learning_rate':                 0.001,
    'actor_lr_decay_rate':                  0.01,
    'critic_lr_decay_rate':                 0.01,
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (84, 84),
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'note':                           '(!) It work 1 time out of 3 time: BBOOT-Lunar-T-max_epoch=220 --> reached ~200 between epoch 73 and 110'
                                      'History: BBOOT-Lunar-N-batch_size_in_ts=2500 --> reached ~200 in 40/120 epoch'
                                      'History: BBOOT-Lunar-K-critic_learning_rate=(0.001) --> Reached avg return ~156 for 30/80 epoch'
                                      '(!) NOT CONSISTENT'
    }


test_hparam = {
    'paramameter_set_name':           'Batch-AAC',
    'rerun_tag':                      'TEST-RUN-H',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'TestSpec',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    # 'prefered_environment':           'CartPole-v0',
    # 'batch_size_in_ts':               300,
    'prefered_environment':           'LunarLander-v2',
    'batch_size_in_ts':               1000,
    'expected_reward_goal':           200,
    'max_epoch':                      10,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    # 'discout_factor':                 [0.999, 0.91],
    'learning_rate':                  3e-4,
    # 'learning_rate':                  [3e-4, 1e-3],
    'critic_learning_rate':           1e-3,
    'actor_lr_decay_rate':            1,                                              # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,                                              # set to 1 to swith OFF scheduler
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (4, 4),
    # 'theta_nn_h_layer_topo':          [(4, 4), (6, 6)],
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'isTestRun':                      True,
    'show_plot':                      False,
    }

parser = argparse.ArgumentParser(description=(
    "=============================================================================\n"
    ":: Command line option for the Actor-Critic Agent.\n\n"
    "To play:\n"
    "     python -m ActorCritic --play [--play_for] [--help] [--testRun]\n\n"
    "To train:\n"
    "     python -m ActorCritic --trainExperimentSpecification [--rerun] [--renderTraining] [--discounted] [--help] [--testRun]\n\n"
    "Choose --trainExperimentSpecification between the following:\n"
    "     - CartPole-v0 environment:\n"
    "          [--trainSplitMC]: Train a Batch Actor-Critic agent with Monte Carlo TD target\n"
    "          [--trainSplitBootstrap]: Train a Batch Actor-Critic agent with bootstrap estimate TD target\n"
    "          [--trainSharedBootstrap]: Train a Batch Actor-Critic agent with shared network\n"
    "          [--trainOnlineSplit]: Train a Online Actor-Critic agent with split network\n"
    "          [--trainOnlineSplit3layer]: Train a Online Actor-Critic agent with split network\n"
    "          [--trainOnlineShared3layer]: Train a Online Actor-Critic agent with Shared network\n"
    "          [--trainOnlineSplitTwoInputAdvantage]: Train a Online Actor-Critic agent with split Two input Advantage network\n"
    "     - LunarLander-v2 environment:\n"
    "          [--trainOnlineLunarLander]: Train on LunarLander a Online Actor-Critic agent with split Two input Advantage network\n"
    "          [--trainBatchLunarLander]: Train on LunarLander a Batch Actor-Critic agent\n"
    ),
    epilog="=============================================================================\n")

# parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--trainSplitMC', action='store_true', help='Train a Batch Actor-Critic agent with Monte Carlo TD target')
parser.add_argument('--trainSplitBootstrap', action='store_true', help='Train a Batch Actor-Critic agent with bootstrap estimate TD target')
parser.add_argument('--trainSharedBootstrap', action='store_true', help='Train a Batch Actor-Critic agent with shared network')

parser.add_argument('--trainOnlineSplit', action='store_true', help='Train a Online Actor-Critic agent with split network')
parser.add_argument('--trainOnlineSplit3layer', action='store_true', help='Train a Online Actor-Critic agent with split network')
parser.add_argument('--trainOnlineShared3layer', action='store_true', help='Train a Online Actor-Critic agent with Shared network')
parser.add_argument('--trainOnlineSplitTwoInputAdvantage', action='store_true', help='Train a Online Actor-Critic agent with split Two input Advantage network')

parser.add_argument('--trainOnlineLunarLander', action='store_true', help='Train on LunarLander a Online Actor-Critic agent with split Two input Advantage network')
parser.add_argument('--trainBatchLunarLander', action='store_true', help='Train on LunarLander a Batch Actor-Critic agent ')

parser.add_argument('-rer', '--rerun', type=int, default=1,
                    help='Rerun training experiment with the same spec r time (default=1)')

parser.add_argument('--renderTraining', action='store_true',
                    help='(Training option) Watch the agent execute trajectories while he is on traning duty')


parser.add_argument('-d', '--discounted', default=None, type=bool,
                    help='(Training option) Force training execution with discounted reward-to-go')


# (Ice-box) todo:implement --> select agent hparam to play by command line:
parser.add_argument('--playLunar',  action='store_true', help='Play on LunarLander-v2 a Batch Actor-Critic agent trained with Bootstrap target on a split network')
parser.add_argument('--playCartpole',  action='store_true', help='Play on CartPole-v0 a Batch Actor-Critic agent trained with Bootstrap target on a split network')

parser.add_argument('--play_for', type=int, default=10,
                    help='(Playing option) Max playing trajectory, default=20')

parser.add_argument('--record', action='store_true',
                    help='(Playing option) Record trained agent playing in a environment')

parser.add_argument('--testRun', action='store_true', help='Flag for automated continuous integration test')

args = parser.parse_args()

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * * *
# *                                                                                                                    *
# *                             Configure selected experiment specification & warmup agent                             *
# *                                                                                                                    *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * * *
consol_width = 90

if args.playCartpole:
    """ ---- Play Cartpole run ---- """

    # (Ice-box) todo:implement --> load hparam dict from the config.txt
    BMCSPL_B_G2_freezed_hparam = {
        'paramameter_set_name':           'Batch-AAC-Split-nn',
        'rerun_tag':                      'BMCSPL-B-G2',
        'algo_name':                      'Batch ActorCritic',
        'comment':                        'MonteCarlo-target',
        'AgentType':                      BatchActorCriticAgent,
        'Target':                         TargetType.MonteCarlo,
        'Network':                        NetworkType.Split,
        'prefered_environment':           'CartPole-v0',
        'expected_reward_goal':           200,
        'batch_size_in_ts':               4000,
        'max_epoch':                      30,
        'discounted_reward_to_go':        True,
        'discout_factor':                 0.99,
        'learning_rate':                  1e-2,
        'critic_learning_rate':           1e-2,
        'actor_lr_decay_rate':            1,  # 9e-1,
        'critic_lr_decay_rate':           1,  # 9e-1,
        'critique_loop_len':              80,
        'theta_nn_h_layer_topo':          (32, 32),
        'random_seed':                    0,
        'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
        'theta_output_layers_activation': None,
        'render_env_every_What_epoch':    100,
        'print_metric_every_what_epoch':  2,
        'isTestRun':                      False,
        'show_plot':                      False,
        'note':                           ''
        }

    run_dir = "Run-BMCSPL-B-G2-4-Batch-AAC-Split-nn-d9h15m37s39/checkpoint/Batch_ActorCritic_agent-200-29"
    play_agent(run_dir, BMCSPL_B_G2_freezed_hparam, args, record=args.record)

elif args.playLunar:
    """ ---- Play LunarLander run ---- """
    BATCH_AAC_LunarLander_freezed_hparam = {
        'rerun_tag':                      'BBOOT-Lunar-T',
        'paramameter_set_name':           'Batch-AAC-Split-nn',
        'comment':                        '',
        'AgentType':                      BatchActorCriticAgent,
        'Network':                        NetworkType.Split,
        'Target':                         TargetType.Bootstrap,
        'algo_name':                      'Batch ActorCritic',
        'prefered_environment':           'LunarLander-v2',
        'expected_reward_goal':           195,  # trigger model save on reach
        'batch_size_in_ts':               4000,
        'max_epoch':                      220,
        'discounted_reward_to_go':        True,
        'discout_factor':                 0.9999,
        'learning_rate':                  0.01,
        'critic_learning_rate':           0.001,
        'actor_lr_decay_rate':            0.01,
        'critic_lr_decay_rate':           0.01,
        'critique_loop_len':              80,
        'theta_nn_h_layer_topo':          (84, 84),
        'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
        'theta_output_layers_activation': None,
        'render_env_every_What_epoch':    5,
        'print_metric_every_what_epoch':  5,
        'note':                           '(!) It work 1 time out of 3 time: BBOOT-Lunar-T-max_epoch=220 --> reached ~200 between epoch 73 and 110'
                                          'History: BBOOT-Lunar-N-batch_size_in_ts=2500 --> reached ~200 in 40/120 epoch'
                                          'History: BBOOT-Lunar-K-critic_learning_rate=(0.001) --> Reached avg return ~156 for 30/80 epoch'
        }

    chekpoint_dir = "Run-BBOOT-Lunar-T-max_epoch=220-0-Batch-AAC-Split-nn()-d10h18m15s22/checkpoint/"
    # run_dir = chekpoint_dir + "Batch_ActorCritic_agent-211-116"
    run_dir = chekpoint_dir + "Batch_ActorCritic_agent-210-78"

    play_agent(run_dir, BATCH_AAC_LunarLander_freezed_hparam, args, record=args.record)

else:
    hparam = None
    key = None
    values_search_set = None

    # --- training ----------------------------------------------------------------------------------------------------
    experiment_start_message(consol_width, args.rerun)

    if args.trainSplitMC:
        """ ---- Batch Split network architecture with Monte Carlo TD target ---- """
        hparam, key, values_search_set = run_experiment(BATCH_AAC_MonteCarlo_SPLIT_net_hparam, args,
                                                        test_hparam, rerun_nb=args.rerun)

    elif args.trainSplitBootstrap:
        """ ---- Batch Split network architecture with Bootstrap estimate TD target run ---- """
        hparam, key, values_search_set = run_experiment(BATCH_AAC_bootstrap_SPLIT_net_hparam, args, test_hparam,
                                                        rerun_nb=args.rerun)

    elif args.trainSharedBootstrap:
        """ ---- Batch Shared network architecture with Bootstrap estimate TD target run ---- """
        hparam, key, values_search_set = run_experiment(BATCH_AAC_Bootstrap_SHARED_net_hparam, args,
                                                        test_hparam, rerun_nb=args.rerun)

    elif args.trainOnlineSplit:
        """ ---- ONLINE Split network architecture run ---- """
        hparam, key, values_search_set = run_experiment(ONLINE_AAC_Bootstrap_SPLIT_net_hparam, args,
                                                        test_hparam, rerun_nb=args.rerun)

    elif args.trainOnlineSplit3layer:
        """ ---- ONLINE Split network 3 hiden layer architecture  run ---- """
        hparam, key, values_search_set = run_experiment(ONLINE_AAC_Bootstrap_SPLIT_three_layer_hparam, args,
                                                        test_hparam, rerun_nb=args.rerun)

    elif args.trainOnlineShared3layer:
        """ ---- V2 ONLINE Shared network architecture with Bootstrap estimate TD target run ---- """
        hparam, key, values_search_set = run_experiment(ONLINE_AAC_Bootstrap_SHARED_three_layer_hparam, args,
                                                        test_hparam, rerun_nb=args.rerun)

    elif args.trainOnlineSplitTwoInputAdvantage:
        """ ---- ONLINE Split Two Input Advantage network 3 hiden layer architecture run ---- """
        hparam, key, values_search_set = run_experiment(
            ONLINE_AAC_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam, args, test_hparam, rerun_nb=args.rerun)

    elif args.trainOnlineLunarLander:
        """ ---- LunarLander ONLINE Split Two Input Advantage network 3 hiden layer architecture run ---- """
        hparam, key, values_search_set = run_experiment(
            ONLINE_AAC_LunarLander_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam, args, test_hparam, rerun_nb=args.rerun)

    elif args.trainBatchLunarLander:
        """ ---- LunarLander batch architecture run ---- """
        hparam, key, values_search_set = run_experiment(BATCH_AAC_LunarLander_hparam, args, test_hparam,
                                                        rerun_nb=args.rerun)

    else:
        raise NotImplementedError

    # --------------------------------------------------------------------------------------------------- training ---/

    experiment_closing_message(hparam, args.rerun, key, values_search_set, consol_width)

exit(0)
