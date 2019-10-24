# coding=utf-8
"""
Invoke Actor-Critic agent using

    python -m ActorCritic [--help] [--train] [--render_training] [--discounted] [--play_for] [--test_run]


Note on TensorBoard usage:
    Start TensorBoard in terminal:
        cd DRLimplementation    <-- (!)
        tensorboard --logdir=ActorCritic/graph/runs

    In browser, go to:
        http://0.0.0.0:6006/


"""
import argparse
import tensorflow as tf

from ActorCritic.integrationBatchAAC import IntegrationActorCriticAgent
from ActorCritic.reference_LilLog_BatchAAC import ReferenceActorCriticAgent
from ActorCritic.BatchActorCriticAgent import ActorCriticAgent
from blocAndTools.buildingbloc import ExperimentSpec

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

batch_AAC_hparam = {
    'paramameter_set_name':           'Batch AAC',
    'algo_name':                      'ActorCritic',
    'comment':                        None,
    'MonteCarloTarget':               True,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               3000,
    'max_epoch':                      100,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    'learning_rate':                  2e-3,
    'critic_learning_rate':           1e-3,
    'critique_loop_len':              40,
    'theta_nn_h_layer_topo':          (82,),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.tanh,  # tf.nn.relu,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    'show_plot':                      False,
    }

lilLogBatch_AAC_hparam = {
    'paramameter_set_name':           'Batch AAC',
    'algo_name':                      'ActorCritic',
    'comment':                        'Lil-Log reference',
    'MonteCarloTarget':               True,
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

integrationBatch_AAC_hparm = lilLogBatch_AAC_hparam.copy()
integrationBatch_AAC_hparm['paramameter_set_name'] = 'Integrate Batch AAC'
integrationBatch_AAC_hparm['comment'] = 'testImplementation'
integrationBatch_AAC_hparm['MonteCarloTarget'] = True

test_hparam = {
    'paramameter_set_name':           'Batch AAC',
    'algo_name':                      'ActorCritic',
    'comment':                        'TestSpec',
    'MonteCarloTarget':               True,
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
parser.add_argument('--train', action='store_true', help='Execute training of Actor-Critic agent')
parser.add_argument('--reference', action='store_true', help='Execute training of Reference Actor-Critic agent')
parser.add_argument('--integration', action='store_true', help='Execute training of Integration Actor-Critic agent')

parser.add_argument('-r', '--render_training', action='store_true',
                    help='(Training option) Watch the agent execute trajectories while he is on traning duty')
parser.add_argument('-d', '--discounted', default=None, type=bool,
                    help='(Training option) Force training execution with discounted reward-to-go')

parser.add_argument('-p', '--play_for', type=int, default=20,
                    help='(Playing option) Max playing trajectory, default=20')

parser.add_argument('--test_run', action='store_true')

args = parser.parse_args()

exp_spec = ExperimentSpec()

if args.train:
    # Configure experiment hyper-parameter
    if args.test_run:
        exp_spec.set_experiment_spec(test_hparam)
    else:
        exp_spec.set_experiment_spec(batch_AAC_hparam)

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    ac_agent = ActorCriticAgent(exp_spec)
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
elif args.integration:
    """ ---- Integration run ---- """
    # Configure experiment hyper-parameter
    if args.test_run:
        exp_spec.set_experiment_spec(test_hparam)
    else:
        exp_spec.set_experiment_spec(integrationBatch_AAC_hparm)

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    ac_agent = IntegrationActorCriticAgent(exp_spec)
    ac_agent.train(render_env=args.render_training)
else:
    exp_spec.set_experiment_spec(batch_AAC_hparam)
    if args.test_run:
        exp_spec.set_experiment_spec({'isTestRun': True})

    ac_agent = ActorCriticAgent(exp_spec)
    raise NotImplementedError  # todo: implement train and select a agent
    ac_agent.play(run_name='todo --> CHANGE_TO_My_TrainedAgent', max_trajectories=args.play_for)

exit(0)
