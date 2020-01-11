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
        `python -m SoftActorCritic --trainExperimentSpecification   [--rerun] [--renderTraining] [--discounted]
                                                                    [--help] [--testRun]`

        Choose `--trainExperimentSpecification` between the following:
        - 'MountainCarContinuous-v0' environment:
            [--trainMontainCar]: Train on Montain Car gym env a Soft Actor-Critic agent
        - 'LunarLanderContinuous-v2' environment:
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

    /--- Target network update -----------------------------------------------------------------------------------------
        'target_update_interval': int               1000 for HARD TARGET update, 1 for EXPONENTIAL MOVING AVERAGE
        'target_smoothing_coefficient': float       (tau) control of EXPONENTIAL MOVING AVERAGE,
                                                     the SAC paper recommand ~ 0.005
                                                    - Large tau can lead to instability, small cam make training slower
                                                    - tau=1 <--> HARD TARGET update
                                            
    /--- Policy related ------------------------------------------------------------------------------------------------
        'alpha': float                              (aka Temperature, Entropy regularization coefficient )
                                                    Control the trade-off between exploration-exploitation
                                                    We recover the standard maximum expected return objective,
                                                     aka the Q-fct, as alpha --> 0
        
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

# (CRITICAL) todo:implement --> train agent:
SAC_MountainCar_hparam = {
    'rerun_tag':                      'MonCar',
    'paramameter_set_name':           'SAC',
    'comment':                        '',
    'algo_name':                      'Soft Actor Critic',
    'AgentType':                      SoftActorCriticAgent,
    'prefered_environment':           'MountainCarContinuous-v0',
    
    'expected_reward_goal':           90,  # Note: trigger model save on reach
    'max_epoch':                      100,
    'timestep_per_epoch':             5000,
    
    'discout_factor':                 0.99,  # SAC paper: 0.99
    'learning_rate':                  0.003,  # SAC paper: 30e-4
    'critic_learning_rate':           0.003,  # SAC paper: 30e-4
    'max_gradient_step_expected':     500000,
    'actor_lr_decay_rate':            0.01,  # Note: set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           0.01,  # Note: set to 1 to swith OFF scheduler
    
    'target_smoothing_coefficient':   0.005,  # SAC paper: EXPONENTIAL MOVING AVERAGE ~ 0.005, 1 <==> HARD TARGET update
    'target_update_interval':         1,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    'gradient_step_interval':         1,
    
    'alpha':                          1,  # HW5: we recover a standard max expected return objective as alpha --> 0
    
    'max_eval_trj':                   10,  #SpiningUp: 10
    
    'pool_capacity':                  int(1e6),  # SAC paper: 1e6
    'min_pool_size':                  1000,
    'batch_size_in_ts':               256,  # SAC paper:256, SpinningUp:100
    
    'theta_nn_h_layer_topo':          (256, 256),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'theta_hidden_layers_activation': tf.nn.relu,
    'theta_output_layers_activation': None,
    'phi_nn_h_layer_topo':            (256, 256),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'phi_hidden_layers_activation':   tf.nn.relu,
    'phi_output_layers_activation':   None,
    'psi_nn_h_layer_topo':            (256, 256),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'psi_hidden_layers_activation':   tf.nn.relu,
    'psi_output_layers_activation':   None,
    
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'note':                           ''
    }

# todo --> training:
SAC_LunarLander_hparam = {
    'rerun_tag':            'MonCar',
    'paramameter_set_name': 'SAC',
    'comment':              '',
    'algo_name':            'Soft Actor Critic',
    'AgentType':            SoftActorCriticAgent,
    'prefered_environment': 'LunarLanderContinuous-v2',
    }

test_hparam = {
    'rerun_tag':                      'TEST-RUN',
    'paramameter_set_name':           'SAC',
    'comment':                        'TestSpec',  # Comment added to training folder name (can be empty)
    'algo_name':                      'Soft Actor Critic',
    'AgentType':                      SoftActorCriticAgent,
    'prefered_environment':           'MountainCarContinuous-v0',
    
    'expected_reward_goal':           90,  # Note: trigger model save on reach
    'max_epoch':                      10,
    'timestep_per_epoch':             500,
    
    'discout_factor':                 0.99,  # SAC paper: 0.99
    'learning_rate':                  0.003,  # SAC paper: 30e-4
    'critic_learning_rate':           0.003,  # SAC paper: 30e-4
    'max_gradient_step_expected':     500000,
    'actor_lr_decay_rate':            0.01,  # Note: set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           0.01,  # Note: set to 1 to swith OFF scheduler
    
    'target_smoothing_coefficient':   0.005,  # SAC paper: EXPONENTIAL MOVING AVERAGE ~ 0.005, 1 <==> HARD TARGET update
    'target_update_interval':         1,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    'gradient_step_interval':         1,
    
    'alpha':                          1,  # HW5: we recover a standard max expected return objective as alpha --> 0
    
    'max_eval_trj':                   10,  #SpiningUp: 10
    
    'pool_capacity':                  int(1e6),  # SAC paper: 1e6
    'min_pool_size':                  100,
    'batch_size_in_ts':               100,  # SAC paper:256, SpinningUp:100
    
    'theta_nn_h_layer_topo':          (4, 4),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'theta_hidden_layers_activation': tf.nn.relu,
    'theta_output_layers_activation': None,
    'phi_nn_h_layer_topo':            (4, 4),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'phi_hidden_layers_activation':   tf.nn.relu,
    'phi_output_layers_activation':   None,
    'psi_nn_h_layer_topo':            (4, 4),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'psi_hidden_layers_activation':   tf.nn.relu,
    'psi_output_layers_activation':   None,
    
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'random_seed':                    0,  # Note: 0 --> turned OFF (default)
    'isTestRun':                      True,
    'show_plot':                      False,
    'note':                           'My note ...'
    }

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
    "     - 'LunarLanderContinuous-v2' environment:\n"
    "          [--trainLunarLander]: Train on LunarLander a Soft Actor-Critic agent\n"
),
    epilog="=============================================================================\n")

# parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--trainMontainCar', action='store_true',
                    help='Train on Montain Car gym env a Soft Actor-Critic agent')

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
        """ ---- Easy environment ---- """
        hparam, key, values_search_set = run_experiment(SAC_MountainCar_hparam, args,
                                                        test_hparam, rerun_nb=args.rerun)
    
    elif args.trainLunarLander:
        raise NotImplementedError  # todo: implement
        """ ---- Harder environment ---- """
        hparam, key, values_search_set = run_experiment(
            SAC_LunarLander_hparam, args, test_hparam, rerun_nb=args.rerun)
    
    else:
        raise NotImplementedError
    
    # --------------------------------------------------------------------------------------------------- training ---/
    
    experiment_closing_message(hparam, args.rerun, key, values_search_set, consol_width)

exit(0)
