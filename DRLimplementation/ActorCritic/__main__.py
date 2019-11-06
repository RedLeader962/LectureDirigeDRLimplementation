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
from typing import Type, List, Tuple, Any
import argparse
import tensorflow as tf

from blocAndTools.agent import Agent
from ActorCritic.BatchActorCriticAgent import BatchActorCriticAgent
from ActorCritic.OnlineActorCriticAgent import OnlineActorCriticAgent
from ActorCritic.OnlineTwoInputAdvantageActorCriticAgent import OnlineTwoInputAdvantageActorCriticAgent
from blocAndTools.buildingbloc import ExperimentSpec
from blocAndTools.rl_vocabulary import TargetType, NetworkType  # , ActorCriticAgentType



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

BATCH_AAC_Undiscounted_MonteCarlo_SPLIT_net_hparam = {
    'paramameter_set_name':           'Batch-AAC-Split-nn',
    'rerun_tag':                      'BMCSPL-NoD-A',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'Undiscounted MonteCarlo-target',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               4000,
    'max_epoch':                      30,
    'discounted_reward_to_go':        False,
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



BATCH_AAC_bootstrap_SPLIT_net_hparam = {
    'paramameter_set_name':           'Batch-AAC-Split-nn',
    'rerun_tag':                      'BBSPL-A',
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

BATCH_AAC_LunarLander_hparam = {
    'paramameter_set_name':           'Batch-AAC-Split-nn',
    'rerun_tag':                      'B-Lunar-A',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'MonteCarlo-target LunarLander',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'LunarLander-v2',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               8000,
    'max_epoch':                      60,
    'discounted_reward_to_go':        False,
    'discout_factor':                 0.99,
    'learning_rate':                  1e-2,
    'critic_learning_rate':           1e-2,
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (64, 64),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           ''
    }

test_hparam = {
    'paramameter_set_name':           'Batch-AAC',
    'rerun_tag':                      'TEST-RUN-G',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'TestSpec',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               300,
    'max_epoch':                      5,
    'discounted_reward_to_go':        True,
    # 'discout_factor':                 0.999,
    'discout_factor':                 [0.999, 0.9],
    'learning_rate':                  3e-4,
    'critic_learning_rate':           1e-3,
    'critique_loop_len':              80,
    'theta_nn_h_layer_topo':          (8, 8),
    # 'theta_nn_h_layer_topo':          [(4, 4), (6, 6)],
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

parser.add_argument('--trainOnlineLunarLander', action='store_true', help='Train on LunarLander a Online Actor-Critic agent with split Two input Advantage network')
parser.add_argument('--trainBatchLunarLander', action='store_true', help='Train on LunarLander a Batch Actor-Critic agent ')

parser.add_argument('-rer', '--rerun', type=int, default=1,
                    help='Rerun training experiment with the same spec r time (default=1)')

parser.add_argument('--renderTraining', action='store_true',
                    help='(Training option) Watch the agent execute trajectories while he is on traning duty')

parser.add_argument('-d', '--discounted', default=None, type=bool,
                    help='(Training option) Force training execution with discounted reward-to-go')

parser.add_argument('--play_for', type=int, default=20,
                    help='(Playing option) Max playing trajectory, default=20')

# (Priority) todo:implement --> select agent to play by command line:
parser.add_argument('-p', '--play', type=str,
                    help='(Playing option) Max playing trajectory, default=20')

parser.add_argument('--testRun', action='store_true')

args = parser.parse_args()

exp_spec = ExperimentSpec()


def run_experiment(hparam: dict, run_idx: int) -> Tuple[dict, str, list]:
    if args.testRun:
        hparam = test_hparam

    init_hparam = hparam.copy()

    hparam_search_list, key, values_search_set = configure_exp_spec_for_hparam_search(hparam)
    for hparam in hparam_search_list:
        exp_spec_ = configure_exp_spec_for_run(hparam, run_idx)
        warmup_agent_for_training(exp_spec_)

    return init_hparam, key, values_search_set

def hparam_search_list_to_str(akey, aValues_search_set):
    if akey is None:
        return ''
    else:
        values_str = ''
        for v in aValues_search_set:
            v_str = str(v)
            v_str = v_str.replace(' ', '')
            v_str = v_str.replace('(', '\\(')
            v_str = v_str.replace(')', '\\)')
            values_str += v_str + '|'

        values_str = values_str.rstrip('|')
        hparam_search_str = akey + '=(' + values_str + ')'
        return hparam_search_str

def configure_exp_spec_for_hparam_search(hparam: dict) -> Tuple[List[dict], Any, Any]:
    key, values = test_hparam_search_set(hparam)
    if key is None:
        return [hparam], None, None
    else:
        hparam_search_list = []
        for v in values:
            hp = hparam.copy()
            hp[key] = v
            tag = hp['rerun_tag']
            value_str = str(v)
            value_str = value_str.replace(' ', '')
            # value_str = value_str.replace(',', '-')
            hp['rerun_tag'] = tag + '-' + key + '=' + value_str
            hparam_search_list.append(hp)
        return hparam_search_list, key, values

def test_hparam_search_set(hparam: dict) -> Tuple[str, list] or Tuple[None, None]:
    """
    Test if a specifiation dictionary as a field containing multiple values to experiment
    :param hparam: the specification for an experiment
    :return: the field key and values_search_set to execute or None if all field are single value
    """
    for k, values in hparam.items():
        if isinstance(values, list):
            return k, values
    return None, None


def configure_exp_spec_for_run(hparam: dict, run_idx) -> ExperimentSpec:
    exp_spec.set_experiment_spec(hparam)

    exp_spec.rerun_idx = run_idx

    if args.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args.discounted})

    return exp_spec

def warmup_agent_for_training(spec: ExperimentSpec):
    agent = spec['AgentType']
    ac_agent = agent(spec)
    ac_agent.train(render_env=args.renderTraining)

def warmup_agent_for_playing(spec: ExperimentSpec):
    raise NotImplementedError   # todo: implement select and PLAY agent
    agent = spec['AgentType']
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


if args.play:
    # (Priority) todo:implement --> :
    """ ---- Play run ---- """
    if args.testRun:
        exp_spec.set_experiment_spec(test_hparam)
    else:
        exp_spec.set_experiment_spec(BATCH_AAC_MonteCarlo_SPLIT_net_hparam)
    warmup_agent_for_playing(exp_spec)
else:

    print("\n:: The experiment will be rerun {} time".format(args.rerun))
    initial_hparam = None
    key = None
    values_search_set = None

    for r_i in range(args.rerun):

        print(":: Starting rerun experiment no {}".format(r_i))

        if args.trainSplitMC:
            """ ---- Batch Split network architecture with Monte Carlo TD target ---- """
            # exp_spec = configure_exp_spec_for_run(BATCH_AAC_MonteCarlo_SPLIT_net_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(BATCH_AAC_MonteCarlo_SPLIT_net_hparam, r_i)

        elif args.trainSplitBootstrap:
            """ ---- Batch Split network architecture with Bootstrap estimate TD target run ---- """
            # exp_spec = configure_exp_spec_for_run(BATCH_AAC_bootstrap_SPLIT_net_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(BATCH_AAC_bootstrap_SPLIT_net_hparam, r_i)

        elif args.trainSharedBootstrap:
            """ ---- Batch Shared network architecture with Bootstrap estimate TD target run ---- """
            # exp_spec = configure_exp_spec_for_run(BATCH_AAC_Bootstrap_SHARED_net_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(BATCH_AAC_Bootstrap_SHARED_net_hparam, r_i)

        elif args.trainOnlineSplit:
            """ ---- ONLINE Split network architecture run ---- """
            # exp_spec = configure_exp_spec_for_run(ONLINE_AAC_Bootstrap_SPLIT_net_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(ONLINE_AAC_Bootstrap_SPLIT_net_hparam, r_i)

        elif args.trainOnlineSplit3layer:
            """ ---- ONLINE Split network 3 hiden layer architecture  run ---- """
            # exp_spec = configure_exp_spec_for_run(ONLINE_AAC_Bootstrap_SPLIT_three_layer_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(ONLINE_AAC_Bootstrap_SPLIT_three_layer_hparam, r_i)

        elif args.trainOnlineShared3layer:
            """ ---- V2 ONLINE Shared network architecture with Bootstrap estimate TD target run ---- """
            # exp_spec = configure_exp_spec_for_run(ONLINE_AAC_Bootstrap_SHARED_three_layer_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(ONLINE_AAC_Bootstrap_SHARED_three_layer_hparam, r_i)

        elif args.trainOnlineSplitTwoInputAdvantage:
            """ ---- ONLINE Split Two Input Advantage network 3 hiden layer architecture run ---- """
            # exp_spec = configure_exp_spec_for_run(ONLINE_AAC_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(ONLINE_AAC_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam, r_i)

        elif args.trainOnlineLunarLander:
            """ ---- LunarLander ONLINE Split Two Input Advantage network 3 hiden layer architecture run ---- """
            # exp_spec = configure_exp_spec_for_run(ONLINE_AAC_LunarLander_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(
                ONLINE_AAC_LunarLander_Bootstrap_TwoInputAdv_SPLIT_three_layer_hparam, r_i)

        elif args.trainBatchLunarLander:
            """ ---- LunarLander batch architecture run ---- """
            # exp_spec = configure_exp_spec_for_run(BATCH_AAC_LunarLander_hparam, r_i)
            # warmup_agent_for_training(exp_spec)
            initial_hparam, key, values_search_set = run_experiment(BATCH_AAC_LunarLander_hparam, r_i)

        else:
            raise NotImplementedError

    name = exp_spec['paramameter_set_name']
    name += " " + exp_spec['comment']

    exp_rerun_tag = initial_hparam['rerun_tag']
    if key is None:
        print("\n:: The experiment - {} - was rerun {} time".format(name, args.rerun),
              "\n:: TensorBoard rerun tag: {}".format(exp_rerun_tag),
              "\n")
    else:
        exp_hparam_search_str = hparam_search_list_to_str(key, values_search_set)
        exp_rerun_tag = exp_rerun_tag + '-' + exp_hparam_search_str
        exp_spec.set_experiment_spec(
            {
                'rerun_tag': exp_rerun_tag,
                key: values_search_set
                })

        print("\n:: Experiment - {}:".format(name),
              "\n\t\t- was run over hyperparameter['values']: {}{}".format(key, str(values_search_set)),
              "\n\t\t- each 'values' was rerun {} time".format(args.rerun),
              "\n:: TensorBoard rerun tag: {}\n".format(exp_rerun_tag),
              # "\n"
              )


for _ in range(3):
    print("/" * consol_width)

exit(0)
