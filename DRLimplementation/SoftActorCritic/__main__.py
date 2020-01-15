# coding=utf-8
"""
  
   .|'''.|            .'|.   .       |               .                   '   ..|'''.|          ||    .    ||
   ||..  '    ...   .||.   .||.     |||      ....  .||.    ...   ... ..    .|'     '  ... ..  ...  .||.  ...    ....
    ''|||.  .|  '|.  ||     ||     |  ||   .|   ''  ||   .|  '|.  ||' ''   ||          ||' ''  ||   ||    ||  .|   ''
  .     '|| ||   ||  ||     ||    .''''|.  ||       ||   ||   ||  ||       '|.      .  ||      ||   ||    ||  ||
  |'....|'   '|..|' .||.    '|.' .|.  .||.  '|...'  '|.'  '|..|' .||.       ''|....'  .||.    .||.  '|.' .||.  '|...'
  
                                                                                                         
                                                                                                        +--- kban style


Invoke Soft Actor-Critic agent (SAC) using

    To play:
        `python -m SoftActorCritic --play [--play_for] [--help] [--testRun]`

    To train:
        `python -m SoftActorCritic < trainExperimentSpecification >   [--rerun] [--renderTraining] [--discounted]
                                                                        [--help] [--testRun]`

        Choose < trainExperimentSpecification > between the following:
        - For 'MountainCarContinuous-v0' environment:
            [--trainMontainCar]: Train on Montain Car gym env a Soft Actor-Critic agent
        - For 'Pendulum-v0' environment:
            [--trainPendulum]: Train on Pendulum gym env a Soft Actor-Critic agent
        - For 'LunarLanderContinuous-v2' environment:
            [--trainLunarLander]: Train on LunarLander a Soft Actor-Critic agent

    Gym continuous environment ex:
        Classic control
        'MountainCarContinuous-v0': max_episode_steps=999, reward_threshold=90.0
        'Pendulum-v0': max_episode_steps=200
        
        Box2D
        'LunarLanderContinuous-v2': max_episode_steps=1000, reward_threshold=200
        'BipedalWalker-v2': max_episode_steps=1600, reward_threshold=300
        'BipedalWalkerHardcore-v2': max_episode_steps=2000, reward_threshold=300
        'CarRacing-v0': max_episode_steps=1000, reward_threshold=900
        'RocketLander-v0': max_episode_steps=1000, reward_threshold=8, https://github.com/EmbersArc/gym_rocketLander

Note on TensorBoard usage:
    Start TensorBoard in terminal:
        cd DRLimplementation   (!)
        tensorboard --logdir=SoftActorCritic/graph

    In browser, go to:
        http://0.0.0.0:6006/

    
"""
import argparse
import tensorflow as tf

from SoftActorCritic.SoftActorCriticAgent import SoftActorCriticAgent
from blocAndTools.experiment_runner import (
    run_experiment, _warmup_agent_for_playing, experiment_closing_message,
    experiment_start_message, play_agent,
    )

"""SAC specific hparam
        
    /--- General -------------------------------------------------------------------------------------------------------
        'timestep_per_epoch': int                   nb of step run before evaluating agent learned policy
                                                     and computing stats
        'gradient_step_interval': int               Train agent at every _ timestep
        'pool_capacity': int                        Nb of collected step to keep. Once reached, previously collected
                                                     step will start being overwriten by new one.
        'min_pool_size': int                        Nb of collected step before training can start. SAC paper=1000
        'reward_scaling': float                     The most important hymperparameter for SAC (1.0 <==> no scaling)
    /--- Target network update -----------------------------------------------------------------------------------------
        'target_update_interval': int               1000 for HARD TARGET update, 1 for EXPONENTIAL MOVING AVERAGE
        'target_smoothing_coefficient': float       (tau, polyak update coeficient)
                                                     Control over the EXPONENTIAL MOVING AVERAGE
                                                     the SAC paper recommand ~ 0.005
                                                     (SpiningUp = 0.995 with 'target_update_interval' = 1)
                                                    - Large tau can lead to instability, small cam make training slower
                                                    - tau=1 <--> HARD TARGET update
                                            
    /--- Policy related ------------------------------------------------------------------------------------------------
        'alpha': float                              (aka Temperature, Entropy regularization coefficient )
                                                    Control the trade-off between exploration-exploitation
                                                    We recover the standard maximum expected return objective,
                                                     aka the Q-fct, as alpha --> 0
                                                    SpinningUp=0.2, SAC paper=1.0
        
    /--- Policy evaluation ---------------------------------------------------------------------------------------------
        'max_eval_trj'                              nb of trajectory executed for agent evaluation using
                                                     a deterministic policy
        
    /--- Neural net ----------------------------------------------------------------------------------------------------
        Actor
            - Policy PI_phi network
                'phi_nn_h_layer_topo'
                'phi_hidden_layers_activation'
                'phi_output_layers_activation'
        
        Critic
            - (State value function) V_psi network
                'psi_nn_h_layer_topo'
                'psi_hidden_layers_activation'
                'psi_output_layers_activation'
            
            - (State-action value function) Q_theta network 1 and 2 (both have same architecture)
                'theta_nn_h_layer_topo'
                'theta_hidden_layers_activation'
                'theta_output_layers_activation'
        
    /--- learning_rate_scheduler()  ------------------------------------------------------------------------------------
        'max_gradient_step_expected'                Max number of training cycle expected during the experiment


    Note: About Gamma value (aka the discout factor)
      |    Big difference between 0.9 and 0.999.
      |    Also you need to take into account the experiment average number of step per episode
      |
      |        Example with experiment average step of 100:
      |           0.9^100 = 0.000026 vs 0.99^100 = 0.366003 vs 0.999^100 = 0.904792
      |
      |    Meaning a agent with Gamma=0.9 is short-sighted and one with Gamma=0.9999 is farsighted or clairvoyant


(!) Note: to trigger hyperparameter search, enclose search space values inside a list ex: [(16, 32), (64, 64), (84, 84)]

"""

SAC_base_hparam = {
    'rerun_tag':                      '',
    'paramameter_set_name':           'SAC',
    'comment':                        '',
    'algo_name':                      'Soft Actor Critic',
    'AgentType':                      SoftActorCriticAgent,
    'prefered_environment':           '',
    
    'expected_reward_goal':           90,  # Note: trigger model save on reach
    'max_epoch':                      50,
    'timestep_per_epoch':             5000,
    
    'reward_scaling':                 5.0,  # the only hparam requiring tuning
    
    'discout_factor':                 0.99,  # SAC paper: 0.99
    'learning_rate':                  0.003,  # SAC paper: 30e-4
    'critic_learning_rate':           0.003,  # SAC paper: 30e-4
    'max_gradient_step_expected':     250000,
    'actor_lr_decay_rate':            1.0,  # Note: set to 1.0 to swith OFF scheduler
    'critic_lr_decay_rate':           1.0,  # Note: set to 1.0 to swith OFF scheduler
    
    'target_smoothing_coefficient':   0.005,  # SAC paper: 0.005 (1 <==> HARD TARGET update), SpiningUp: 0.995,
    'target_update_interval':         1,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    'gradient_step_interval':         1,
    
    # HW5: recover a standard max expected return objective as alpha --> 0, SpinningUp alpha=0.2
    'alpha':                          0.95,
    'max_eval_trj':                   10,  #SpiningUp: 10
    
    'pool_capacity':                  int(1e6),  # SAC paper & SpinningUp: 1e6
    'min_pool_size':                  5000,  # SpinningUp: 10000
    'batch_size_in_ts':               100,  # SAC paper:256, SpinningUp:100
    
    'theta_nn_h_layer_topo':          (200, 200),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'theta_hidden_layers_activation': tf.nn.relu,
    'theta_output_layers_activation': None,
    'phi_nn_h_layer_topo':            (200, 200),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'phi_hidden_layers_activation':   tf.nn.relu,
    'phi_output_layers_activation':   None,
    'psi_nn_h_layer_topo':            (200, 200),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'psi_hidden_layers_activation':   tf.nn.relu,
    'psi_output_layers_activation':   None,
    
    'render_env_every_What_epoch':    5,
    'render_env_eval_interval':       5,
    'print_metric_every_what_epoch':  5,
    'log_metric_interval':            500,
    'note':                           ''
    }

# 'MountainCarContinuous-v0'
# - action_space:  Box(1,)
#    - high: [1.]
#    - low: [-1.]
# - observation_space:  Box(3,)
#    - high: [0.6,  0.07]
#    - low: [-1.2,  -0.07]
# - reward_range:  (-inf, inf)
# - spec:
#    - max_episode_steps: 999
#    - reward_threshold: 90.0       #  The reward threshold before the task is considered solved
SAC_MountainCar_hparam = {
    'rerun_tag':                      'MonCar',
    'paramameter_set_name':           'SAC',
    'comment':                        '',
    'algo_name':                      'Soft Actor Critic',
    'AgentType':                      SoftActorCriticAgent,
    'prefered_environment':           'MountainCarContinuous-v0',
    
    'expected_reward_goal':           90,  # Note: trigger model save on reach
    'max_epoch':                      50,
    'timestep_per_epoch':             10000,
    
    'reward_scaling':                 [3.0, 6.0, 12.0, 24.0, 48.0],
    
    'discout_factor':                 0.99,  # SAC paper: 0.99
    'learning_rate':                  0.003,  # SAC paper: 30e-4
    'critic_learning_rate':           0.003,  # SAC paper: 30e-4
    'max_gradient_step_expected':     5000000,
    'actor_lr_decay_rate':            1.0,  # Note: set to 1.0 to swith OFF scheduler
    'critic_lr_decay_rate':           1.0,  # Note: set to 1.0 to swith OFF scheduler
    
    'target_smoothing_coefficient':   1.0,  # SAC paper: EXPONENTIAL MOVING AVERAGE ~ 0.005, 1 <==> HARD TARGET update
    'target_update_interval':         1000,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    'gradient_step_interval':         1,
    
    'alpha':                          1,  # HW5: we recover a standard max expected return objective as alpha --> 0
    'max_eval_trj':                   10,  #SpiningUp: 10
    
    'pool_capacity':                  int(1e6),  # SAC paper & SpinningUp: 1e6
    'min_pool_size':                  8000,
    'batch_size_in_ts':               64,  # SAC paper:256, SpinningUp:100
    
    'theta_nn_h_layer_topo':          (200, 200),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'theta_hidden_layers_activation': tf.nn.relu,
    'theta_output_layers_activation': None,
    'phi_nn_h_layer_topo':            (200, 200),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'phi_hidden_layers_activation':   tf.nn.relu,
    'phi_output_layers_activation':   None,
    'psi_nn_h_layer_topo':            (200, 200),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'psi_hidden_layers_activation':   tf.nn.relu,
    'psi_output_layers_activation':   None,
    
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'note':                           ''
    }

# 'Pendulum-v0'
# - action_space:  Box(1,)
#    - high: [2.]
#    - low: [-2.]
# - observation_space:  Box(3,)
#    - high: [1. 1. 8.]
#    - low: [-1. -1. -8.]
# - reward_range:  (-inf, inf)
# - spec:
#    - max_episode_steps: 200
#    - reward_threshold: None       #  The reward threshold before the task is considered solved
#
# Gym: https://gym.openai.com/envs/MountainCarContinuous-v0/
SAC_Pendulum_hparam = dict(SAC_base_hparam)
SAC_Pendulum_hparam.update(
    {
        'rerun_tag':                     'RewardScaleTest',
        'comment':                       '',
        'prefered_environment':          'Pendulum-v0',
        'expected_reward_goal':          -160,
        'max_epoch':                     250,
        'timestep_per_epoch':            1000,
        'max_gradient_step_expected':    250000,
        'actor_lr_decay_rate':           1.0,  # Note: set to 1.0 to swith OFF scheduler
        'critic_lr_decay_rate':          1.0,  # Note: set to 1.0 to swith OFF scheduler
        'batch_size_in_ts':              100,  # SAC paper:256, SpinningUp:100
        'learning_rate':                 0.006,  # SAC paper: 30e-4
        'critic_learning_rate':          0.006,  # SAC paper: 30e-4
    
        # HW5: recover a standard max expected return objective as alpha --> 0, SpinningUp=0.2, SAC paper=1.0
        'alpha':                         0.2,
        'max_eval_trj':                  20,  #SpiningUp: 10
    
        'reward_scaling':                [2.0, 5.0, 10.0, 1.0, -1.0, -2.0, -5.0],
    
        'target_smoothing_coefficient':  0.995,  # SAC paper: 0.005 (1 <==> HARD TARGET update), SpiningUp: 0.995,
        'target_update_interval':        10,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    
        'pool_capacity':                 int(1e5),  # SAC paper & SpinningUp: 1e6
        'min_pool_size':                 10000,  # SpinningUp: 10000
    
        'render_env_every_What_epoch':   5,
        'render_env_eval_interval':      10,
        'print_metric_every_what_epoch': 10,
        'log_metric_interval':           50,
        'note':                          ''
        }
    )

# 'LunarLanderContinuous-v2'
# - action_space:  Box(2,) ⟶ [main engine, left-right engines]
#    - high: [1. , 1.]
#    - low: [-1. , -1.]
# - observation_space:  Box(8,)
#    - high: [inf inf inf inf inf inf inf inf]
#    - low: [-inf -inf -inf -inf -inf -inf -inf -inf]
# - reward_range:  (-inf, inf)
# - spec:
#    - max_episode_steps: 1000
#    - reward_threshold: 200       #  The reward threshold before the task is considered solved
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back.
# Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points.
# Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points.
# Landing outside landing pad is possible.
# Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
# Action is two real values vector from -1 to +1.
# First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power.
# Engine can't work with less than 50% power. S
# econd value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.
# source: https://gym.openai.com/envs/LunarLanderContinuous-v2/
SAC_LunarLander_hparam = dict(SAC_base_hparam)
SAC_LunarLander_hparam.update(
    {
        'rerun_tag':                     'LunarLander',
        'comment':                       '',
        'prefered_environment':          'LunarLanderContinuous-v2',
        'expected_reward_goal':          190,  # goal: 200
        'reward_scaling':                3.0,
        'render_env_every_What_epoch':   1,
        'render_env_eval_interval':      5,
        'print_metric_every_what_epoch': 5,
        'log_metric_interval':           500,
        'note':                          ''
        }
    )

# ... Test hparam ......................................................................................................
test_hparam = dict(SAC_base_hparam)
test_hparam.update(
    {
        'rerun_tag':             'TEST-RUN',
        'comment':               'TestSpec',
        'prefered_environment':  'Pendulum-v0',
    
        'max_epoch':             3,
        'timestep_per_epoch':    220,
    
        'expected_reward_goal':  -130,  # goal: 200
        'reward_scaling':        3.0,
    
        'pool_capacity':         int(1e2),  # SAC paper: 1e6
        'min_pool_size':         50,  # SpinningUp: 10000
        'batch_size_in_ts':      10,  # SAC paper:256, SpinningUp:100
    
        'theta_nn_h_layer_topo': (2, 2),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'phi_nn_h_layer_topo':   (2, 2),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'psi_nn_h_layer_topo':   (2, 2),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'isTestRun':             True,
        'show_plot':             False,
        }
    )

SAC_Pendulum_hparam_TEST_RERUN = dict(test_hparam)
SAC_Pendulum_hparam_TEST_RERUN.update(
    {
        'max_epoch':          2,
        'timestep_per_epoch': 220,
        'isTestRun':          False,
        }
    )
# .............................................................................................. Test hparam ...(end)...


parser = argparse.ArgumentParser(description=(
    "=============================================================================\n"
    ":: Command line option for the Soft Actor-Critic Agent.\n\n"
    "To play:\n"
    "     python -m SoftActorCritic --play [--play_for] [--help] [--testRun]\n\n"
    "To train:\n"
    "     python -m SoftActorCritic --trainExperimentSpecification   [--rerun] [--renderTraining] [--discounted] "
    "[--help] [--testRun]\n\n"
    "Choose --trainExperimentSpecification between the following:\n"
    "     - 'MountainCarContinuous-v0':\n"
    "          [--trainMontainCar]: Train on Montain Car gym env a Soft Actor-Critic agent\n"
    "     - 'Pendulum-v0':\n"
    "          [--trainPendulum]: Train on Pendulum gym env a Soft Actor-Critic agent\n"
    "     - 'LunarLanderContinuous-v2' environment:\n"
    "          [--trainLunarLander]: Train on LunarLander a Soft Actor-Critic agent\n"
),
    epilog="=============================================================================\n")

# parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--trainMontainCar', action='store_true',
                    help='Train on Montain Car gym env a Soft Actor-Critic agent')

parser.add_argument('--trainPendulum', action='store_true', help='Train on Pendulum a Soft Actor-Critic agent')

# todo:unit-test --> problem: rerun error
parser.add_argument('--trainPendulumTESTrerun', action='store_true', help='Train on Pendulum a Soft Actor-Critic agent')

parser.add_argument('--trainLunarLander', action='store_true', help='Train on LunarLander a Soft Actor-Critic agent')

parser.add_argument('-rer', '--rerun', type=int, default=1,
                    help='Rerun training experiment with the same spec r time (default=1)')

parser.add_argument('--renderTraining', action='store_true',
                    help='(Training option) Watch the agent execute trajectories while he is on traning duty')

parser.add_argument('-d', '--discounted', default=None, type=bool,
                    help='(Training option) Force training execution with discounted reward-to-go')

parser.add_argument('--playLunar', action='store_true', help='Play a trained Soft Actor-Critic agent on the '
                                                             'LunarLanderContinuous-v2 environment')
# (Ice-box) todo:implement --> select agent hparam to play by command line:

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

if args.playLunar:
    # (Ice-box) todo:implement --> load hparam dict from the config.txt
    
    raise NotImplementedError  # (CRITICAL) todo:implement --> play trainned agent from command line::

#     """ ---- Play LunarLander run ---- """
#     BATCH_AAC_LunarLander_freezed_hparam = {
#         'rerun_tag':                      'BBOOT-Lunar-T',
#         'paramameter_set_name':           'Batch-AAC-Split-nn',
#         'comment':                        '',
#         'AgentType':                      BatchActorCriticAgent,
#         'Network':                        NetworkType.Split,
#         'Target':                         TargetType.Bootstrap,
#         'algo_name':                      'Batch ActorCritic',
#         'prefered_environment':           'LunarLander-v2',
#         'expected_reward_goal':           195,  # trigger model save on reach
#         'batch_size_in_ts':               4000,
#         'max_epoch':                      220,
#         'discounted_reward_to_go':        True,
#         'discout_factor':                 0.9999,
#         'learning_rate':                  0.01,
#         'critic_learning_rate':           0.001,
#         'actor_lr_decay_rate':            0.01,
#         'critic_lr_decay_rate':           0.01,
#         'critique_loop_len':              80,
#         'theta_nn_h_layer_topo':          (84, 84),
#         'theta_hidden_layers_activation': tf.nn.relu,  # tf.nn.tanh,
#         'theta_output_layers_activation': None,
#         'render_env_every_What_epoch':    5,
#         'print_metric_every_what_epoch':  5,
#         'note':                           '(!) It work 1 time out of 3 time: BBOOT-Lunar-T --> reached ~200 '
#                                           'History: BBOOT-Lunar-N-batch_size_in_ts=2500 --> reached ~200 '
#                                           'History: BBOOT-Lunar-K-critic_learning_rate=(0.001) '
#                                           '                 --> Reached avg return ~156 for 30/80 epoch'
#         }
#
#     chekpoint_dir = "Run-BBOOT-Lunar-T-max_epoch=220-0-Batch-AAC-Split-nn()-d10h18m15s22/checkpoint/"
#     # run_dir = chekpoint_dir + "Batch_ActorCritic_agent-211-116"
#     run_dir = chekpoint_dir + "Batch_ActorCritic_agent-210-78"
#
#     play_agent(run_dir, BATCH_AAC_LunarLander_freezed_hparam, args, record=args.record)

else:
    hparam = None
    key = None
    values_search_set = None
    
    # --- training ----------------------------------------------------------------------------------------------------
    experiment_start_message(consol_width, args.rerun)

    if args.trainMontainCar:
        """ ---- Easy environment [--trainMontainCar] ---- """
        hparam, key, values_search_set = run_experiment(SAC_MountainCar_hparam, args,
                                                        test_hparam, rerun_nb=args.rerun)

    elif args.trainPendulum:
        """ ---- Harder environment [--trainPendulum] ---- """
        hparam, key, values_search_set = run_experiment(
            SAC_Pendulum_hparam, args, test_hparam, rerun_nb=args.rerun)

    elif args.trainPendulumTESTrerun:
        """ ---- Harder environment [--trainPendulumTESTrerun] ---- """
        hparam, key, values_search_set = run_experiment(
            SAC_Pendulum_hparam_TEST_RERUN, args, test_hparam, rerun_nb=args.rerun)

    elif args.trainLunarLander:
        """ ---- Harder environment [--trainLunarLander] ---- """
        hparam, key, values_search_set = run_experiment(
            SAC_LunarLander_hparam, args, test_hparam, rerun_nb=args.rerun)

    else:
        raise NotImplementedError
    
    # --------------------------------------------------------------------------------------------------- training ---/
    
    experiment_closing_message(hparam, args.rerun, key, values_search_set, consol_width)

exit(0)
