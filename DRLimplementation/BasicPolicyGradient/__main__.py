# coding=utf-8
"""
Invoke REINFORCE agent using

    python -m BasicPolicyGradient [--help] [--train] [--render_training] [--discounted] [--play_for] [--test_run]

Note on TensorBoard usage:
    Start TensorBoard in terminal:
        cd DRLimplementation
        tensorboard --logdir=BasicPolicyGradient/graph/runs

    In browser, go to:
        http://0.0.0.0:6006/

"""
import argparse
import tensorflow as tf

from BasicPolicyGradient.REINFORCEagent import REINFORCEagent
from blocAndTools.buildingbloc import ExperimentSpec

# Note: About Gamma value (aka the discout factor)
#       Big difference between 0.9 and 0.999.
#       Also you need to take into account the experiment average number of step per episode
#
#           Example with experiment average step of 100:
#              0.9^100 = 0.000026 vs 0.99^100 = 0.366003 vs 0.999^100 = 0.904792

cartpole_hparam = {
    'paramameter_set_name':           'Basic',
    'algo_name':                      'REINFORCE',
    'comment':                        None,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               5000,
    'max_epoch':                      40,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-2,
    'theta_nn_h_layer_topo':          (62,),
    'random_seed':                    82,
    'theta_hidden_layers_activation': tf.nn.tanh,  # tf.nn.relu,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    }

test_hparam = {
    'paramameter_set_name':           'Basic',
    'algo_name':                      'REINFORCE',
    'comment':                        'TestSpec',
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               1000,
    'max_epoch':                      5,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-2,
    'theta_nn_h_layer_topo':          (8, 8),
    'random_seed':                    82,
    'theta_hidden_layers_activation': tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      True,
    }

parser = argparse.ArgumentParser(description=(
    "=============================================================================\n"
    ":: Command line option for the REINFORCE Agent.\n\n"
    "   The agent will play by default using previously trained computation graph.\n"
    "   You can execute training by using the argument: --train "),
    epilog="=============================================================================\n")

# parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--train', action='store_true', help='Execute training of REINFORCE agent')

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
        exp_spec.set_experiment_spec(cartpole_hparam)

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    reinforce_agent = REINFORCEagent(exp_spec)
    reinforce_agent.train(render_env=args.render_training)
else:
    exp_spec.set_experiment_spec(cartpole_hparam)
    if args.test_run:
        exp_spec.set_experiment_spec({'isTestRun': True})

    reinforce_agent = REINFORCEagent(exp_spec)
    reinforce_agent.play(run_name='REINFORCE_agent-39', max_trajectories=args.play_for)

exit(0)
