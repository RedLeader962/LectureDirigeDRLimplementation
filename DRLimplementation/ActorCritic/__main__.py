# coding=utf-8
"""
Invoke Actor-Critic agent using

    python -m ActorCritic [--help] [--trainMC] [--renderTraining] [--discounted] [--play_for] [--testRun]

todo --> add command line new training spec:

Note on TensorBoard usage:
    Start TensorBoard in terminal:
        cd DRLimplementation   (!)
        tensorboard --logdir=ActorCritic/graph

    In browser, go to:
        http://0.0.0.0:6006/


"""
from typing import List, Tuple, Any, Iterable, Union, Type
import argparse
import tensorflow as tf

from blocAndTools.agent import Agent
from ActorCritic.BatchActorCriticAgent import BatchActorCriticAgent
from ActorCritic.OnlineActorCriticAgent import OnlineActorCriticAgent
from ActorCritic.OnlineTwoInputAdvantageActorCriticAgent import OnlineTwoInputAdvantageActorCriticAgent
from ActorCritic.reference_LilLog_BatchAAC import ReferenceActorCriticAgent
from blocAndTools.buildingbloc import ExperimentSpec
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
    'rerun_tag':                      'BMCSPL-A',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'MonteCarlo-target',
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

BATCH_AAC_MonteCarlo_SPLIT_net_hparam = BATCH_AAC_MonteCarlo_SPLIT_net_hparam.copy()
# BATCH_AAC_MonteCarlo_SPLIT_net_hparam['rerun_tag'] = 'BMCSPL-NoD-A'
# BATCH_AAC_MonteCarlo_SPLIT_net_hparam['comment'] = 'Undiscounted MonteCarlo-target'
# BATCH_AAC_MonteCarlo_SPLIT_net_hparam['discounted_reward_to_go'] = False

BATCH_AAC_MonteCarlo_SPLIT_net_hparam['rerun_tag'] = 'BMCSPL-Lunar-B'
BATCH_AAC_MonteCarlo_SPLIT_net_hparam['comment'] = 'LunarLander-v2'
BATCH_AAC_MonteCarlo_SPLIT_net_hparam['prefered_environment'] = 'LunarLander-v2'
BATCH_AAC_MonteCarlo_SPLIT_net_hparam['batch_size_in_ts'] = 8000
BATCH_AAC_MonteCarlo_SPLIT_net_hparam['max_epoch'] = 60
BATCH_AAC_MonteCarlo_SPLIT_net_hparam['theta_nn_h_layer_topo'] = (64, 64)


BATCH_AAC_bootstrap_SPLIT_net_hparam = {
    'paramameter_set_name':           'Batch-AAC-Split-nn',
    'rerun_tag':                      None,
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'Discounted-Bootstrap-target',
    'Target':                         TargetType.Bootstrap,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               4000,
    # 'max_epoch':                      30,
    'max_epoch':                      50,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.99,
    'learning_rate':                  1e-2,
    'critic_learning_rate':           1e-3,
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (32, 32),
    # 'theta_nn_h_layer_topo':          (62,),    # <--(!) not learning
    'random_seed':                    0,
    # 'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
    'theta_hidden_layers_activation': tf.nn.tanh,  # tf.nn.relu,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           "Both loss have a lot less variance. The algo take more time to converge."
    }

BATCH_AAC_bootstrap_SPLIT_net_hparam = BATCH_AAC_bootstrap_SPLIT_net_hparam.copy()
BATCH_AAC_bootstrap_SPLIT_net_hparam['learning_rate'] = 1e-2
BATCH_AAC_bootstrap_SPLIT_net_hparam['critic_learning_rate'] = 1e-2


# BATCH_AAC_bootstrap_SPLIT_net_hparam['comment'] = 'Discounted-Bootstrap-target Short-sighted'
# BATCH_AAC_bootstrap_SPLIT_net_hparam['discout_factor'] = 0.9

BATCH_AAC_bootstrap_SPLIT_net_hparam['rerun_tag'] = 'AA'
BATCH_AAC_bootstrap_SPLIT_net_hparam['comment'] = 'Discounted-Bootstrap-target Farsighted'
BATCH_AAC_bootstrap_SPLIT_net_hparam['discout_factor'] = 0.9999

BATCH_AAC_bootstrap_SPLIT_net_hparam['theta_hidden_layers_activation'] = tf.nn.relu
BATCH_AAC_bootstrap_SPLIT_net_hparam['note'] = "Both loss have a lot less variance. The algo take more time to converge. relu seams to work better"

BATCH_AAC_bootstrap_SPLIT_net_hparam['rerun_tag'] = 'AAA'
BATCH_AAC_bootstrap_SPLIT_net_hparam['random_seed'] = 33
BATCH_AAC_bootstrap_SPLIT_net_hparam['batch_size_in_ts'] = 2000
BATCH_AAC_bootstrap_SPLIT_net_hparam['critique_loop_len'] = 120


BATCH_AAC_bootstrap_SPLIT_net_hparam['rerun_tag'] = 'BBSPL-A'
BATCH_AAC_bootstrap_SPLIT_net_hparam['random_seed'] = 0
BATCH_AAC_bootstrap_SPLIT_net_hparam['batch_size_in_ts'] = 3000
BATCH_AAC_bootstrap_SPLIT_net_hparam['critique_loop_len'] = 120
BATCH_AAC_bootstrap_SPLIT_net_hparam['max_epoch'] = 50
BATCH_AAC_bootstrap_SPLIT_net_hparam['learning_rate'] = 1e-2
BATCH_AAC_bootstrap_SPLIT_net_hparam['critic_learning_rate'] = 1e-3
BATCH_AAC_bootstrap_SPLIT_net_hparam['theta_nn_h_layer_topo'] = (16, 32, 64)


BATCH_AAC_Bootstrap_SHARED_net_hparam = {
    'paramameter_set_name':           'Batch-AAC-Shared-nn',
    'rerun_tag':                      'BSHA-A',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'Bootstrap-Tiny-Batch-WORKING',
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
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               8,
    'stage_size_in_trj':              50,
    'max_epoch':                      45,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-4,
    'critic_learning_rate':           5e-4,
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
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               20,
    'stage_size_in_trj':              50,
    'max_epoch':                      45,
    'discout_factor':                 0.999,
    'learning_rate':                  5e-5,
    'critic_learning_rate':           5e-4,
    'critique_loop_len':              1,
    'theta_nn_h_layer_topo':          (16, 32, 256),
    'random_seed':                    0,
    # 'theta_hidden_layers_activation': tf.nn.leaky_relu,  # tf.nn.tanh, tf.nn.relu
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
    'Network':                        NetworkType.Shared,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               10,
    'stage_size_in_trj':              50,
    'max_epoch':                      45,
    'discout_factor':                 0.95,
    'learning_rate':                  3e-4,
    'critic_learning_rate':           3e-4,
    'critique_loop_len':              2,
    'theta_nn_h_layer_topo':          (32, 64, 256),
    'random_seed':                    0,
    # 'theta_hidden_layers_activation': tf.nn.leaky_relu,  # tf.nn.tanh, tf.nn.relu
    # 'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh, tf.nn.leaky_relu
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
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               5,
    'stage_size_in_trj':              50,
    'max_epoch':                      25,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-4,
    'critic_learning_rate':           5e-4,
    'critique_loop_len':              1,
    # 'theta_nn_h_layer_topo':          (16, 32, 80),
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

# BATCH_AAC_Bootstrap_SHARED_net_hparam = BATCH_AAC_bootstrap_SPLIT_net_hparam.copy()
# BATCH_AAC_Bootstrap_SHARED_net_hparam['comment'] = "Bootstrap SHARED network"
# BATCH_AAC_Bootstrap_SHARED_net_hparam['Target'] = TargetType.Bootstrap
# BATCH_AAC_Bootstrap_SHARED_net_hparam['Network'] = NetworkType.Shared
# BATCH_AAC_Bootstrap_SHARED_net_hparam['note'] = ""

lilLogBatch_AAC_hparam = {
    'paramameter_set_name':           'Batch-AAC',
    'rerun_tag':                      None,
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'Lil-Log reference',
    'Target':                         TargetType.MonteCarlo,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               4000,
    'max_epoch':                      30,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.99,
    'learning_rate':                  1e-2,
    'critic_learning_rate':           1e-2,
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (32, 32),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    'show_plot':                      False,
    }

test_hparam = {
    'paramameter_set_name':           'Batch-AAC',
    'rerun_tag':                      'A',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'TestSpec',
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               1000,
    'max_epoch':                      5,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    'learning_rate':                  3e-4,
    'critic_learning_rate':           1e-3,
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (8, 8),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      True,
    'show_plot':                      False,
    }

parser = argparse.ArgumentParser(description=(
    "=============================================================================\n"
    ":: Command line option for the Actor-Critic Agent.\n\n"
    "   The agent will play by default using previously trained computation graph.\n"
    "   You can execute training by using the argument: --train "),
    epilog="=============================================================================\n")

# parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--trainSplitMC', action='store_true', help='Train a Batch Actor-Critic agent with Monte Carlo TD target')
parser.add_argument('--trainSplitBootstrap', action='store_true', help='Train a Batch Actor-Critic agent with bootstrap estimate TD target')
parser.add_argument('--trainSharedBootstrap', action='store_true', help='Train a Batch Actor-Critic agent with shared network')
parser.add_argument('--trainOnlineSplit', action='store_true', help='Train a Online Actor-Critic agent with split network')

parser.add_argument('--trainOnlineSplit3layer', action='store_true', help='Train a Online Actor-Critic agent with split network')

parser.add_argument('--trainOnlineShared3layer', action='store_true', help='Train a Online Actor-Critic agent with Shared network')

parser.add_argument('--trainOnlineSplitTwoInputAdvantage', action='store_true', help='Train a Online Actor-Critic agent with split Two input Advantage network')


parser.add_argument('--reference', action='store_true', help='Execute training of reference Actor-Critic implementation by Lilian Weng')

parser.add_argument('-rer', '--rerun', type=int, default=1,
                    help='Rerun training experiment with the same spec r time (default=1)')

parser.add_argument('--renderTraining', action='store_true',
                    help='(Training option) Watch the agent execute trajectories while he is on traning duty')

parser.add_argument('-d', '--discounted', default=None, type=bool,
                    help='(Training option) Force training execution with discounted reward-to-go')

parser.add_argument('-p', '--play_for', type=int, default=20,
                    help='(Playing option) Max playing trajectory, default=20')

parser.add_argument('--testRun', action='store_true')

args = parser.parse_args()

exp_spec = ExperimentSpec()


def configure_exp_spec(hparam: dict, run_idx) -> ExperimentSpec:

    if args.testRun:
        exp_spec.set_experiment_spec(test_hparam)
    else:
        exp_spec.set_experiment_spec(hparam)

    exp_spec.rerun_idx = run_idx

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    return exp_spec

def warmup_agent_for_training(agent: Type[Agent], spec: ExperimentSpec):
    # global ac_agent
    ac_agent = agent(spec)
    ac_agent.train(render_env=args.renderTraining)

def warmup_agent_for_playing(agent: Type[Agent], spec: ExperimentSpec):
    raise NotImplementedError   # todo: implement select and PLAY agent
    ac_agent = agent(spec)
    ac_agent.play(run_name='todo --> CHANGE_TO_My_TrainedAgent', max_trajectories=args.play_for)


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * * *
# *                                                                                                                    *
# *                             Configure selected experiment specification & warmup agent                             *
# *                                                                                                                    *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * * *

consol_width = 90
print("\n")
for _ in range(3):
    print("\\" * consol_width)

print("\n:: The experiment will be rerun {} time".format(args.rerun))

for r_i in range(args.rerun):

    print(":: Starting rerun experiment no {}".format(r_i))

    if args.trainSplitMC:
        """ ---- Batch Split network architecture with Monte Carlo TD target ---- """
        exp_spec = configure_exp_spec(BATCH_AAC_MonteCarlo_SPLIT_net_hparam, r_i)
        warmup_agent_for_training(BatchActorCriticAgent, exp_spec)

    elif args.trainSplitBootstrap:
        """ ---- Batch Split network architecture with Bootstrap estimate TD target run ---- """
        exp_spec = configure_exp_spec(BATCH_AAC_bootstrap_SPLIT_net_hparam, r_i)
        warmup_agent_for_training(BatchActorCriticAgent, exp_spec)

    elif args.trainSharedBootstrap:
        """ ---- Batch Shared network architecture with Bootstrap estimate TD target run ---- """
        exp_spec = configure_exp_spec(BATCH_AAC_Bootstrap_SHARED_net_hparam, r_i)
        warmup_agent_for_training(BatchActorCriticAgent, exp_spec)

    elif args.trainOnlineSplit:
        """ ---- ONLINE Split network architecture run ---- """
        exp_spec = configure_exp_spec(ONLINE_AAC_Bootstrap_SPLIT_net_hparam, r_i)
        warmup_agent_for_training(OnlineActorCriticAgent, exp_spec)

    elif args.trainOnlineSplit3layer:
        """ ---- ONLINE Split network 3 hiden layer architecture  run ---- """
        exp_spec = configure_exp_spec(ONLINE_AAC_Bootstrap_SPLIT_three_layer_hparam, r_i)
        warmup_agent_for_training(OnlineActorCriticAgent, exp_spec)

    elif args.trainOnlineShared3layer:
        """ ---- V2 ONLINE Shared network architecture with Bootstrap estimate TD target run ---- """
        exp_spec = configure_exp_spec(ONLINE_AAC_Bootstrap_SHARED_three_layer_hparam, r_i)
        warmup_agent_for_training(OnlineActorCriticAgent, exp_spec)

    elif args.trainOnlineSplitTwoInputAdvantage:
        """ ---- ONLINE Split Two Input Advantage network 3 hiden layer architecture run ---- """
        exp_spec = configure_exp_spec(ONLINE_AAC_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam, r_i)
        warmup_agent_for_training(OnlineTwoInputAdvantageActorCriticAgent, exp_spec)

    elif args.reference:
        """ ---- Lil-Log reference run ---- """
        exp_spec = configure_exp_spec(lilLogBatch_AAC_hparam, r_i)
        warmup_agent_for_training(ReferenceActorCriticAgent, exp_spec)

    else:

        """ ---- Play run ---- """
        exp_spec = configure_exp_spec(BATCH_AAC_MonteCarlo_SPLIT_net_hparam, r_i)
        warmup_agent_for_playing(BatchActorCriticAgent, exp_spec)
        BatchActorCriticAgent(exp_spec)


name = exp_spec['paramameter_set_name']
name += " " + exp_spec['comment']

print("\n:: The experiment - {} - was rerun {} time.\n".format(name, args.rerun),
      exp_spec.__repr__(),
      "\n")

for _ in range(3):
    print("/" * consol_width)

exit(0)
