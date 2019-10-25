# coding=utf-8
"""
Invoke Actor-Critic agent using

    python -m ActorCritic [--help] [--train] [--render_training] [--discounted] [--play_for] [--test_run]


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
from ActorCritic.reference_LilLog_BatchAAC import ReferenceActorCriticAgent
from blocAndTools.buildingbloc import ExperimentSpec
from blocAndTools.rl_vocabulary import TargetType, NetworkType

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                                                                                                                   *
# *                                   Advantage Actor-Critic (batch architecture)                                     *
# *                                                                                                                   *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# Note: About Gamma value (aka the discout factor)
#       Big difference between 0.9 and 0.999.
#       Also you need to take into account the experiment average number of step per episode
#
#           Example with experiment average step of 100:
#              0.9^100 = 0.000026 vs 0.99^100 = 0.366003 vs 0.999^100 = 0.904792

batch_AAC_MonteCarlo_target_hparam = {
    'paramameter_set_name':           'Batch AAC MonteCarlo target',
    'algo_name':                      'ActorCritic',
    'comment':                        '',
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

batch_AAC_Bootstrap_target_hparam = {
    'paramameter_set_name':           'Batch AAC Element wise Bootstrap target',
    'algo_name':                      'ActorCritic',
    'comment':                        '',
    'Target':                         TargetType.Bootstrap,
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
    'note':                           "Both loss have a lot less variance. The algo take more time to converge"
    }

batch_AAC_Bootstrap_SHARED_net_hparam = {
    'paramameter_set_name':           'AAC shared network',
    'algo_name':                      'ActorCritic',
    'comment':                        'Bootstrap Tiny Batch WORKING',
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
    'random_seed':                    13,
    'theta_hidden_layers_activation': tf.nn.leaky_relu,  # tf.nn.tanh, tf.nn.leaky_relu
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  8,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           ("Does not learn on large batch! "
                                       "Work only on tiny batch (more or less 1 trajectory)"
                                       "Use small hlayer topo"
                                       "small learning rate"
                                       "Extremely sensible to hyper param tuning"
                                       "Can possibly not learn at all on different run with same hparam "
                                       "probably because of unlucky initialisation")
    }

# batch_AAC_Bootstrap_SHARED_net_hparam = batch_AAC_Bootstrap_target_hparam.copy()
# batch_AAC_Bootstrap_SHARED_net_hparam['comment'] = "Bootstrap SHARED network"
# batch_AAC_Bootstrap_SHARED_net_hparam['Target'] = TargetType.Bootstrap
# batch_AAC_Bootstrap_SHARED_net_hparam['Network'] = NetworkType.Shared
# batch_AAC_Bootstrap_SHARED_net_hparam['note'] = ""

lilLogBatch_AAC_hparam = {
    'paramameter_set_name':           'Batch AAC',
    'algo_name':                      'ActorCritic',
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
    'paramameter_set_name':           'Batch AAC',
    'algo_name':                      'ActorCritic',
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
    'random_seed':                    82,
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
parser.add_argument('--trainMC', action='store_true', help='Train a Actor-Critic agent with Monte Carlo TD target')
parser.add_argument('--trainBootstap', action='store_true', help='Train a Actor-Critic agent with bootstrap estimate TD target')
parser.add_argument('--trainShared', action='store_true', help='Train a Actor-Critic agent with shared network')
parser.add_argument('--reference', action='store_true', help='Execute training of reference Actor-Critic implementation by Lilian Weng')

parser.add_argument('-r', '--render_training', action='store_true',
                    help='(Training option) Watch the agent execute trajectories while he is on traning duty')

parser.add_argument('-d', '--discounted', default=None, type=bool,
                    help='(Training option) Force training execution with discounted reward-to-go')

parser.add_argument('-p', '--play_for', type=int, default=20,
                    help='(Playing option) Max playing trajectory, default=20')

parser.add_argument('--test_run', action='store_true')

args = parser.parse_args()

exp_spec = ExperimentSpec()

if args.trainMC:
    """ ---- Monte Carlo TD target ---- """
    # Configure experiment hyper-parameter
    if args.test_run:
        exp_spec.set_experiment_spec(test_hparam)
    else:
        exp_spec.set_experiment_spec(batch_AAC_MonteCarlo_target_hparam)

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    ac_agent = BatchActorCriticAgent(exp_spec)
    ac_agent.train(render_env=args.render_training)
elif args.trainBootstap:
    """ ---- Bootstrap estimate TD target run ---- """
    # Configure experiment hyper-parameter
    if args.test_run:
        exp_spec.set_experiment_spec(test_hparam)
    else:
        exp_spec.set_experiment_spec(batch_AAC_Bootstrap_target_hparam)

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    ac_agent = BatchActorCriticAgent(exp_spec)
    ac_agent.train(render_env=args.render_training)
elif args.trainShared:
    """ ---- Bootstrap estimate TD target run ---- """
    # Configure experiment hyper-parameter
    if args.test_run:
        exp_spec.set_experiment_spec(test_hparam)
    else:
        exp_spec.set_experiment_spec(batch_AAC_Bootstrap_SHARED_net_hparam)

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    ac_agent = BatchActorCriticAgent(exp_spec)
    ac_agent.train(render_env=args.render_training)
elif args.reference:
    """ ---- Lil-Log reference run ---- """
    # Configure experiment hyper-parameter
    if args.test_run:
        exp_spec.set_experiment_spec(test_hparam)
    else:
        exp_spec.set_experiment_spec(lilLogBatch_AAC_hparam)

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    ac_agent = ReferenceActorCriticAgent(exp_spec)
    ac_agent.train(render_env=args.render_training)
else:
    """ ---- Play run ---- """
    exp_spec.set_experiment_spec(batch_AAC_MonteCarlo_target_hparam)
    if args.test_run:
        exp_spec.set_experiment_spec({'isTestRun': True})

    ac_agent = BatchActorCriticAgent(exp_spec)
    raise NotImplementedError  # todo: implement train and select a agent
    ac_agent.play(run_name='todo --> CHANGE_TO_My_TrainedAgent', max_trajectories=args.play_for)

exit(0)
