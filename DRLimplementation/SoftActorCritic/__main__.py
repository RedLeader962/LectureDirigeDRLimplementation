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
        `python -m SoftActorCritic [--playLunar, --playHardLunar, --playPendulum]  [--play_for] [--help]
            [--testRun] [--harderEnvCoeficient=1.6 default]`

    To train:
        `python -m SoftActorCritic < trainExperimentSpecification >   [--rerun] [--renderTraining] [--discounted]
                                                                        [--help] [--testRun]`

        Choose < trainExperimentSpecification > between the following:
        - For 'BipedalWalker-v2' environment:
            [--trainBipedalWalker]: Train on Bipedal Walker gym env a Soft Actor-Critic agent
        - For 'Pendulum-v0' environment:
            [--trainPendulum]: Train on Pendulum gym env a Soft Actor-Critic agent
        - For 'LunarLanderContinuous-v2' environment:
            [--trainLunarLander]: Train on LunarLander a Soft Actor-Critic agent
        - Experimentation utility:
            [--trainExperimentBuffer]: Run a batch of experiment spec

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
    run_experiment,
    experiment_closing_message,
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
        'reward_scaling': float                     The most important hyperparameter for SAC [0 < ...]
                                                        - The 'reward_scaling' coefficient must be tune based on
                                                            the avg reward magnitude
                                                          Avg mean reward (over 5000 samples) ex:
                                                            MountainCarContinuous-v0: -0.03120409531395761
                                                                    vs
                                                            LunarLanderContinuous-v2: -2.0004798743377448
                                                        - reward_scaling=0 <--> KILL the algorithm
                                                        - reward_scaling=1.0 <--> no scaling
                                                        - Note: It can land in [0 < rewS <= 1]
                                                        
    /--- Target network update -----------------------------------------------------------------------------------------
        'target_update_interval': int               1000 for HARD TARGET update, 1 for EXPONENTIAL MOVING AVERAGE
        'target_smoothing_coefficient': float       (aka: tau, polyak averaging coeficient)[0..1]
                                                     Control over the EXPONENTIAL MOVING AVERAGE
                                                        - SAC paper recommand ~ 0.005
                                                          (SpiningUp = 0.995 with 'target_update_interval' = 1)
                                                        - tau=Large --> can lead to instability,
                                                        - tau=Small --> can make training slower
                                                        - tau=1 <--> HARD TARGET update
                                            
    /--- Policy related ------------------------------------------------------------------------------------------------
        'alpha': float                              (aka Temperature, Entropy regularization coefficient )
                                                    Control the trade-off between exploration-exploitation
                                                    alpha~0 <--> SAC become a standard max expected return objective
                                                    (SpinningUp=0.2, SAC paper=1.0)
        
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
            
            - (State-action value function) Q_theta network {1, 2} (both have same architecture)
                'theta_nn_h_layer_topo'
                'theta_hidden_layers_activation'
                'theta_output_layers_activation'
        
    /--- learning_rate_scheduler()  ------------------------------------------------------------------------------------
        'max_gradient_step_expected'                Max number of training cycle expected during the experiment


    Note: About Gamma value (aka the discout factor)
      |    Big difference between 0.9 and 0.999.
      |    Also you need to take into account the experiment average number of step per episode
      |
      |        Example of 'discounted return' over 100 timestep:
      |           0.9^100 --> 0.000026 vs 0.99^100 --> 0.366003 vs 0.999^100 --> 0.904792
      |
      |    Meaning a agent with Gamma=0.9 is short-sighted and one with Gamma=0.9999 is farsighted or clairvoyant


(!) Note: to trigger hyperparameter search, enclose search space values inside a list ex: [(16, 32), (64, 64), (84, 84)]

"""
experiment_buffer = []

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
    'max_trj_steps':                  None,
    
    # (!) Note: The only hparam requiring carefull tuning.
    #       |       - The 'reward_scaling' coefficient must be tune based on the avg reward magnitude
    #       |       - It can land in [0 < rewS <= 1]
    'reward_scaling':                 1.0,  # SAC paper: 3, 10, 30  SpinningUp: 1.0 --> OFF
    
    'discout_factor':                 0.99,  # SAC paper: 0.99
    'learning_rate':                  0.003,  # SAC paper: 30e-4 SpinningUp: 0.001
    'critic_learning_rate':           0.003,  # SAC paper: 30e-4 SpinningUp: 0.001
    'max_gradient_step_expected':     250000,
    'actor_lr_decay_rate':            1.0,  # Note: set to 1.0 to swith OFF scheduler
    'critic_lr_decay_rate':           1.0,  # Note: set to 1.0 to swith OFF scheduler
    
    # Note: HARD TARGET update vs EXPONENTIAL MOVING AVERAGE (EMA)
    #  |                                        EMA         HARD TARGET
    #  |        target_smoothing_coefficient:   0.005       1.0
    #  |        target_update_interval:         1           1000
    #  |        gradient_step_interval:         1           4 (except for rlLab humanoid)
    'target_smoothing_coefficient':   0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
    'target_update_interval':         1,  # SAC paper: 1 or 1000 SpinningUp: all T.U. performed at trj end
    'gradient_step_interval':         1,  # SAC paper: 1 or 4 SpinningUp: all G. step performed at trj end
    
    # Note: alpha = 0 <==> SAC become a standard max expected return objective
    'alpha':                          1.0,  # SAC paper=1.0, SpinningUp=0.2
    'max_eval_trj':                   10,  #SpiningUp: 10
    
    # todo: pool_capacity test [1e4, 1e5, 1e6]
    'pool_capacity':                  int(1e6),  # SAC paper and SpinningUp: 1e6
    'min_pool_size':                  10000,  # SpinningUp: 10000
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
    'note':                           '',
    }

# ::: MontainCar experiment ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
# - Avg reward assement:   -0.03120409531395761

SAC_MountainCar_base_hparam = dict(SAC_base_hparam)
SAC_MountainCar_base_hparam.update(
    {
        'rerun_tag':                  'MonCarBase',
        'paramameter_set_name':       'SAC-MonCar',
        'comment':                    '',
        'prefered_environment':       'MountainCarContinuous-v0',
        'expected_reward_goal':       85,  # Goal: 90
        'max_epoch':                  50,
        'max_gradient_step_expected': 500000,
        'max_trj_steps':              999,
        'note':                       '',
        }
    )

# ... RewardScaling experiment .........................................................................................

SAC_MountainCar_rewardScaling_hparam = dict(SAC_base_hparam)
SAC_MountainCar_rewardScaling_hparam.update(
    {
        'rerun_tag':                    'MonCarBase',
        'comment':                      'rewardScaling',
        'batch_size_in_ts':             100,  # SAC paper:256, SpinningUp:100
        'reward_scaling':               0.5,  # SAC paper: [3, 10, 30]  SpinningUp: 1.0
        'target_smoothing_coefficient': 0.01,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'theta_nn_h_layer_topo':        (64, 64),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'phi_nn_h_layer_topo':          (64, 64),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'psi_nn_h_layer_topo':          (64, 64),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'note':                         '',
        }
    )
# Experiment >>>  Todo: MountainCarContinuous RewardScaling experiment
# experiment_buffer.append(SAC_MountainCar_rewardScaling_hparam)
# ................................................................................. RewardScaling experiment ...(end)...


# ::: Pendulum experiment ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
# - Avg reward assement:  -5.687553812910036
#
# Gym: https://gym.openai.com/envs/MountainCarContinuous-v0/
SAC_Pendulum_base_hparam = dict(SAC_base_hparam)
SAC_Pendulum_base_hparam.update(
    {
        'rerun_tag':                     'BasePendulum',
        'comment':                       '',
        'prefered_environment':          'Pendulum-v0',
        'expected_reward_goal':          (-561),
        'max_epoch':                     50,
        'timestep_per_epoch':            5000,
        'max_trj_steps':                 200,
        'max_gradient_step_expected':    250000,
        'actor_lr_decay_rate':           1.0,  # Note: set to 1.0 to swith OFF scheduler
        'critic_lr_decay_rate':          1.0,  # Note: set to 1.0 to swith OFF scheduler
        
        # (Ice-Boxed) todo: batch size test [64, 256, 100]
        'batch_size_in_ts':              100,  # SAC paper:256, SpinningUp:100
        
        'learning_rate':                 0.003,  # SAC paper: 30e-4 SpinningUp: 0.001
        'critic_learning_rate':          0.003,  # SAC paper: 30e-4 SpinningUp: 0.001
        
        'max_eval_trj':                  20,  #SpiningUp: 10
        
        # (!) Note: The only hparam requiring carefull tuning. Can be [0.0 < rewS <= 1.0]
        'reward_scaling':                0.5,  # SAC paper: [3, 10, 30]  SpinningUp: 1.0
        
        'pool_capacity':                 int(1e6),  # SAC paper & SpinningUp: 1e6
        'min_pool_size':                 10000,  # SpinningUp: 10000
        
        'theta_nn_h_layer_topo':         (16,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'phi_nn_h_layer_topo':           (16,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'psi_nn_h_layer_topo':           (16,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        
        'render_env_every_What_epoch':   5,
        'render_env_eval_interval':      3,
        'print_metric_every_what_epoch': 10,
        'log_metric_interval':           50,
        'note':                          '(Proof of life) Should reach Avg Return close to ~ -560',
        }
    )

# ... RewardScaling experiment .........................................................................................
SAC_Pendulum_RewardScaling_hparam = dict(SAC_Pendulum_base_hparam)
SAC_Pendulum_RewardScaling_hparam.update(
    {
        'rerun_tag':                    'RewardScaling',
        'comment':                      'EMA',
        'prefered_environment':         'Pendulum-v0',
        'expected_reward_goal':         (-150),
        
        # Note: alpha = 0 <==> SAC become a standard max expected return objective
        'alpha':                        1.0,  # SAC paper=1.0 SpinningUp=0.2
        'max_eval_trj':                 20,  # SpiningUp: 10
        
        # (!) Note: The only hparam requiring carefull tuning. Can be [0.0 < rewS <= 1.0]
        'reward_scaling':               [0.25, 0.5, 0.75, 1.0, 3.0, 10.0],  # SAC paper: 3, 10, 30  SpinningUp: 1.0
        
        'target_smoothing_coefficient': 0.05,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'note':                         '',
        }
    )
# Experiment >>> Done: RewardScaling experiment
# experiment_buffer.append(SAC_Pendulum_RewardScaling_hparam)
# ................................................................................. RewardScaling experiment ...(end)...


# ... Target Update experimentation ....................................................................................
SAC_Pendulum_EMA_hparam = dict(SAC_Pendulum_base_hparam)
SAC_Pendulum_EMA_hparam.update(
    {
        'rerun_tag':                    'TargetUpdate',
        'comment':                      'EMA',
        'expected_reward_goal':         (-150),
        'reward_scaling':               0.5,
        'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'target_update_interval':       1,  # SAC paper: 1 or 1000 SpinningUp: all T.U. performed at trj end
        'gradient_step_interval':       1,  # SAC paper: 1 or 4 SpinningUp: all G. step performed at trj end
        'note':                         'Similar result as in the SAC paper',
        }
    )

SAC_Pendulum_HardTarget_hparam = dict(SAC_Pendulum_base_hparam)
SAC_Pendulum_HardTarget_hparam.update(
    {
        'rerun_tag':                    'TargetUpdate',
        'comment':                      'HardTarget',
        'expected_reward_goal':         (-150),
        'reward_scaling':               0.5,
        'target_smoothing_coefficient': 1.0,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'target_update_interval':       1000,  # SAC paper: 1 or 1000 SpinningUp: all T.U. performed at trj end
        'gradient_step_interval':       4,  # SAC paper: 1 or 4 SpinningUp: all G. step performed at trj end
        'note':                         'Could not reproduce the result of the SAC paper. '
                                        'Agent failed to learn at all. '
                                        'Could the NN architecture be the cause?',
        }
    )

# Experiment >>> Done: TargetUpdate experiment  rerun 5 time
# experiment_buffer.append(SAC_Pendulum_EMA_hparam)
# experiment_buffer.append(SAC_Pendulum_HardTarget_hparam)
# ............................................................................ Target Update experimentation ...(end)...


# ... Alpha experiment .................................................................................................
SAC_Pendulum_Alpha_hparam = dict(SAC_Pendulum_base_hparam)
SAC_Pendulum_Alpha_hparam.update(
    {
        'rerun_tag':                    'Alpha',
        'comment':                      '',
        'expected_reward_goal':         (-150),
        'alpha':                        [1.0, 0.75, 0.5, 0.25, 0.0],  # SAC paper=1.0 SpinningUp=0.2
        'reward_scaling':               0.5,
        'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'target_update_interval':       1,  # SAC paper: 1 or 1000 SpinningUp: all T.U. performed at trj end
        'gradient_step_interval':       1,  # SAC paper: 1 or 4 SpinningUp: all G. step performed at trj end
        'note':                         '',
        }
    )
# Experiment >>> (Ice-Boxed) todo: Alpha experiment rerun 5 time
# experiment_buffer.append(SAC_Pendulum_Alpha_hparam)

# ......................................................................................... Alpha experiment ...(end)...

# ::: LunarLander experiment :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
# - Avg reward assement: -2.0004798743377448
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
SAC_LunarLander_base_hparam = dict(SAC_base_hparam)
SAC_LunarLander_base_hparam.update(
    {
        'rerun_tag':            'LunarLander-EMA',
        'comment':              'base',
        'prefered_environment': 'LunarLanderContinuous-v2',
        'max_trj_steps':        1000,
        'expected_reward_goal': 190,  # goal: 200
        'note':                 ''
        }
    )

SAC_LunarLander_baseHardtarget_hparam = dict(SAC_LunarLander_base_hparam)
SAC_LunarLander_baseHardtarget_hparam.update(
    {
        'rerun_tag':                    'LunarLander-HT',
        'target_smoothing_coefficient': 1.0,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'target_update_interval':       1000,  # SAC paper: 1 or 1000 SpinningUp: all T.U. performed at trj end
        'gradient_step_interval':       4,  # SAC paper: 1 or 4 SpinningUp: all G. step performed at trj end
        'note':                         'Unconclusive result, learning to do worst actualy'
                                        'Maybe the NN is to small (64X64) and the experiment lenght to short.'
        }
    )

SAC_LunarLander_base_Working_hparam = dict(SAC_LunarLander_base_hparam)
SAC_LunarLander_base_Working_hparam.update(
    {
        'rerun_tag':                    SAC_LunarLander_base_hparam['rerun_tag'] + '-MinPool',
        'comment':                      'working',
        'max_epoch':                    50,
        'max_gradient_step_expected':   250000,
        'batch_size_in_ts':             200,
        'pool_capacity':                int(5e4),  # <--(!)
        'min_pool_size':                20000,  #<--(!) with kernel init HE normal
        'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'theta_nn_h_layer_topo':        (200, 200),
        'phi_nn_h_layer_topo':          (200, 200),
        'psi_nn_h_layer_topo':          (200, 200),
        'reward_scaling':               40.0,
        'note':                         'min_pool_size is a extremetly important hparam'
                                        'min_pool_size=20000 make SAC fly consistently'
        }
    )

SAC_LunarLander_baseForHardEnv_hparam = dict(SAC_LunarLander_base_hparam)
SAC_LunarLander_baseForHardEnv_hparam.update(
    {
        'rerun_tag':                    SAC_LunarLander_base_hparam['rerun_tag'] + '-ForHardEnv',
        'comment':                      'LargePool',
        'max_epoch':                    500,
        'timestep_per_epoch':           1000,
        'max_gradient_step_expected':   500000,
        'batch_size_in_ts':             200,
        'pool_capacity':                int(2e5),  # <--(!)
        'min_pool_size':                80000,  #<--(!) with kernel init HE normal
        'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'theta_nn_h_layer_topo':        (200, 200),
        'phi_nn_h_layer_topo':          (200, 200),
        'psi_nn_h_layer_topo':          (200, 200),
        'reward_scaling':               40.0,
        'note':                         ''
        }
    )

SAC_LunarLander_rewScaleForHardEnv_hparam = dict(SAC_LunarLander_baseForHardEnv_hparam)
SAC_LunarLander_rewScaleForHardEnv_hparam.update(
    {
        'rerun_tag':                    SAC_LunarLander_baseForHardEnv_hparam['rerun_tag'] + '-RewS',
        'comment':                      'LargePool',
        'batch_size_in_ts':             200,
        'pool_capacity':                int(2e5),  # <--(!)
        'min_pool_size':                80000,  #<--(!) with kernel init HE normal
        'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'reward_scaling':               [75.0, 1.0, 100.0],
        'note':                         ''
        }
    )
# Experiment >>>   todo: [75.0, 1.0, 100.0] rerun 4 done: [40.0, 10.0,] rerun 1
# LunarLander-EMA-ForHardEnv-RewS-reward_scaling=(40.0|10.0|75.0)
# experiment_buffer.append(SAC_LunarLander_rewScaleForHardEnv_hparam)

# ... Lunar reward scale (second attempt) .................................................................
SAC_LunarLander_rewScaleLarge_hparam = dict(SAC_LunarLander_base_hparam)
new_rerun_tag = SAC_LunarLander_rewScaleLarge_hparam['rerun_tag'] + '-MedRewS-ModBuffer'
for nn in [(160, 160)]:
    nnstr = str(nn).strip('()')
    nnstr = nnstr.replace(', ', 'X')
    SAC_LunarLander_rewScaleLarge_hparam.update(
        {
            'rerun_tag':                    new_rerun_tag,
            'comment':                      'MedRewS NN{}'.format(nnstr),
            # 'max_epoch':                    25,
            # 'max_gradient_step_expected':   125000,
            'max_epoch':                    50,
            'max_gradient_step_expected':   250000,
            'batch_size_in_ts':             200,  # <--smaller batch. Previously was set to 100
            # 'pool_capacity':                int(3e4),  # <--larger pool. Previously was set to 1e4
            'pool_capacity':                int(5e4),  # <--larger pool. Previously was set to 1e4
            'min_pool_size':                10000,  # SpinningUp: 10000
            'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
            'theta_nn_h_layer_topo':        nn,
            'phi_nn_h_layer_topo':          nn,
            'psi_nn_h_layer_topo':          nn,
            
            # 'reward_scaling':               [95.0, 110.0, 80.0, ], # hypothèse 1
            'reward_scaling':               40.0,  #hypothèse 2 done: 30.0, 20.0, 40.0,
            # (!) Note: One success: reward_scaling=40.0 and NN 160X160 todo--> rerun 5
            #  |            Run-LunarLander-EMA-MedRewS-MedRewS-reward_scaling=40.0-1-SAC(
            #  |                    rewScaleMedium_smallPool_nnArchitecture160X160)-d22h4m11s7
            'note':                         'hypothesis 1: rew_thr_mujoco * rew_scale = rew_thr * ?'
                                            'should give rew_scale_Lunar=95 '
                                            'hypothesis 2: based on Mujoco/SAC paper experiment, '
                                            '   the mean-heated-rew ~ [19 < x < 30]'
                                            'Base on closer Box[3,] Hopper, our meanHeatedRewGoal ~ 19'
                                            'So we seek: rew_thr / maxStep * rewS ~ 19 '
                                            '1000*19/200/ = 95 = rewS_lunar'
            }
        )
    # Experiment >>>   todo: rerun 5 with reward_scaling=40.0 and NN 160X160
    #                   done: nn [(64,), (64, 64),(160,160),(200, 200)]
    # experiment_buffer.append(SAC_LunarLander_rewScaleLarge_hparam.copy())

# ......................................................... Lunar reward scale (second attempt) ...(end)...


SAC_LunarLander_MINpoolSize_hparam = dict(SAC_LunarLander_base_hparam)
new_rerun_tag_minPool = SAC_LunarLander_MINpoolSize_hparam['rerun_tag'] + '-MinPool-ModBuffer'
for nn in [(200, 200)]:
    nnstr = str(nn).strip('()')
    nnstr = nnstr.replace(', ', 'X')
    SAC_LunarLander_MINpoolSize_hparam.update(
        {
            'rerun_tag':                    new_rerun_tag_minPool,
            'comment':                      'NN{}'.format(nnstr),
            'max_epoch':                    50,
            'max_gradient_step_expected':   250000,
            'batch_size_in_ts':             200,
            'pool_capacity':                int(5e4),  # <--(!)
            'min_pool_size':                [20000, 300, 10000],  #<--(!) SpinningUp: 10000
            # 'min_pool_size':                20000,   #<--(!) with kernel init HE normal
            'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
            'theta_nn_h_layer_topo':        nn,
            'phi_nn_h_layer_topo':          nn,
            'psi_nn_h_layer_topo':          nn,
            'reward_scaling':               40.0,
            'note':                         'This is a extremetly important hparam'
                                            'min_pool_size=20000 make SAC fly consistently'
            }
        )
    # Experiment >>>    done: min_pool_size=[300, 20000, 10000] rerun 5
    # Rerun Tag: LunarLander-EMA-MinPool-ModBuffer-min_pool_size=(300|20000|10000)
    # (NICE TO HAVE) todo:investigate?? --> min_pool_size=200000 with  batch_size < 200 < batch_size:
    # (NICE TO HAVE) todo:investigate?? --> kernel initialization effect on agent performance:
    # experiment_buffer.append(SAC_LunarLander_MINpoolSize_hparam.copy())

SAC_LunarLander_NNsize_minPoolOk_hparam = dict(SAC_LunarLander_base_hparam)
for nn in [(16, 16), (64, 64), (100, 100)]:
    nnstr = str(nn).strip('()')
    nnstr = nnstr.replace(', ', 'X')
    SAC_LunarLander_NNsize_minPoolOk_hparam.update(
        {
            'rerun_tag':                    SAC_LunarLander_base_hparam['rerun_tag'] + '-MinPool-NNsize',
            'comment':                      'NN{}'.format(nnstr),
            'max_epoch':                    50,
            'max_gradient_step_expected':   250000,
            'batch_size_in_ts':             200,
            'pool_capacity':                int(5e4),  # <--(!)
            'min_pool_size':                20000,  # SpinningUp: 10000
            'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
            'theta_nn_h_layer_topo':        nn,
            'phi_nn_h_layer_topo':          nn,
            'psi_nn_h_layer_topo':          nn,
            'reward_scaling':               40.0,
            'note':                         ''
            }
        )
    # Experiment >>> done: LunarLander asses  NNsize rerun 1 todo: rerun 4
    # Rerun Tag: LunarLander-EMA-MinPool-NNsize-.*NN16X16|NN64X64|NN100X100
    # experiment_buffer.append(SAC_LunarLander_NNsize_minPoolOk_hparam.copy())

SAC_LunarLander_rewardScale_TWO_minPoolOk_hparam = dict(SAC_LunarLander_base_hparam)
lunar_nn = (200, 200)  # <--(!)
SAC_LunarLander_rewardScale_TWO_minPoolOk_hparam.update(
    {
        'rerun_tag':                    SAC_LunarLander_base_hparam['rerun_tag'] + '-MinPool-RewSFour',
        'comment':                      'rewardScaleOnMinPoolOK ',
        'max_epoch':                    50,
        'max_gradient_step_expected':   250000,
        'batch_size_in_ts':             200,  # <--(!)
        'pool_capacity':                int(5e4),  # SAC paper & SpinningUp: 1e6
        'min_pool_size':                20000,  # SpinningUp: 10000
        'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
        'theta_nn_h_layer_topo':        lunar_nn,
        'phi_nn_h_layer_topo':          lunar_nn,
        'psi_nn_h_layer_topo':          lunar_nn,
        'reward_scaling':               [1.0, 5.0, 20.0, 40.0, 65.0, 100.0],  # with rerun=3
        'note':                         ''
        }
    )
# Experiment >>> todo: LunarLander asses rewardScaleOnProperMinPool rerun 2 todo: rerun 4
# Rerun Tag: LunarLander-EMA-MinPool-RewS-reward_scaling=(1.0|20.0|40.0|65.0|100.0)
#  done: MinPool with lunar_nn=100X100 and batch_size_in_ts=100, Rerun Tag: LunarLander-reward_scaling=(1.0|20.0|40.0)
# experiment_buffer.append(SAC_LunarLander_rewardScale_TWO_minPoolOk_hparam)
# .................................................................... Proper minimum pool size ...(end)...

SAC_LunarLander_AlphaTest_hparam = dict(SAC_LunarLander_base_hparam)
new_rerun_tag = SAC_LunarLander_AlphaTest_hparam['rerun_tag'] + '-Alpha'
lunar_nn = (160, 160)
SAC_LunarLander_AlphaTest_hparam.update(
    {
        'rerun_tag':                  new_rerun_tag,
        'comment':                    'alphaTestTwo',  # Note: 'alphaTest' --> holder test without rew_scale at 1.0
        'max_epoch':                  50,
        'max_gradient_step_expected': 250000,
        # 'pool_capacity':              int(5e4),  # SAC paper & SpinningUp: 1e6
        'batch_size_in_ts':           200,
        'theta_nn_h_layer_topo':      lunar_nn,
        'phi_nn_h_layer_topo':        lunar_nn,
        'psi_nn_h_layer_topo':        lunar_nn,
        # note: (!) Since rew_scale is considered the inverse of the temperature alpha in the SAC paper
        #  ⬇
        'reward_scaling':             1.0,
        'alpha':                      [0.75, 0.5, 0.25],  # SAC paper=1.0, SpinningUp=0.2
        'note':                       'alphaTest first experiment --> Unconclusive'
                                      'alphaTestTwo experiment --> todo'
        }
    )
# Experiment >>>   todo: LunarLander alphaTestTwo
# experiment_buffer.append(SAC_LunarLander_AlphaTest_hparam)

SAC_LunarLander_noPiMinQ_hparam = dict(SAC_LunarLander_base_hparam)
SAC_LunarLander_noPiMinQ_hparam.update(
    {
        'rerun_tag':      'LunarLander',
        'comment':        'smallBuffer RS15e-1 noPiMinQ',
        'pool_capacity':  int(1e5),  # SAC paper & SpinningUp: 1e6
        'min_pool_size':  5000,  # SpinningUp: 10000
        'reward_scaling': 1.5,
        'note':           '',
        }
    )
# Experiment >>>   done: LunarLander alphaTest
# experiment_buffer.append(SAC_LunarLander_noPiMinQ_hparam)

SAC_LunarLander_smallBuffer_hparam = dict(SAC_LunarLander_base_hparam)
SAC_LunarLander_smallBuffer_hparam.update(
    {
        'rerun_tag':             'LunarLanderLargeNN',
        'comment':               'smallBuffer RS15e-1 ',
        'pool_capacity':         int(1e5),  # SAC paper & SpinningUp: 1e6
        'min_pool_size':         5000,  # SpinningUp: 10000
        'reward_scaling':        1.5,
        'theta_nn_h_layer_topo': (200, 200),
        'phi_nn_h_layer_topo':   (200, 200),
        'psi_nn_h_layer_topo':   (200, 200),
        'note':                  '',
        }
    )
# Experiment >>>   done: LunarLander alphaTest
# experiment_buffer.append(SAC_LunarLander_smallBuffer_hparam)

SAC_LunarLander_rewardScale_hparam = dict(SAC_LunarLander_base_hparam)
lunar_nn = (256, 256)
SAC_LunarLander_rewardScale_hparam.update(
    {
        'rerun_tag':                  'LunarLander',
        'comment':                    'rewardScaleFastStart',
        'max_epoch':                  15,
        'max_gradient_step_expected': 75000,
        'pool_capacity':              int(1e6),  # SAC paper & SpinningUp: 1e6
        'reward_scaling':             [0.5, 0.25, 0.8, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],  # with rerun=3
        'batch_size_in_ts':           100,
        'theta_nn_h_layer_topo':      lunar_nn,
        'phi_nn_h_layer_topo':        lunar_nn,
        'psi_nn_h_layer_topo':        lunar_nn,
        'note':                       'Evaluate for promissing start (over 15 epoch)'
                                      'Most promissing: ~3.0'
        }
    )
# Experiment >>> done: LunarLander asses the proper reward_scaling,
# with value [0.5, 0.25, 0.8, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,] and rerun=5
# experiment_buffer.append(SAC_LunarLander_rewardScale_hparam)


SAC_LunarLander_BatchSize_hparam = dict(SAC_LunarLander_base_hparam)
lunar_nn = (100, 100)
SAC_LunarLander_BatchSize_hparam.update(
    {
        'rerun_tag':                  'LunarLander',
        'comment':                    'batchSizeTest',
        'max_epoch':                  75,
        'max_gradient_step_expected': 375000,
        'pool_capacity':              int(1e5),  # SAC paper & SpinningUp: 1e6
        'min_pool_size':              40000,  # SpinningUp: 10000
        'batch_size_in_ts':           [64, 100, 256],
        'theta_nn_h_layer_topo':      lunar_nn,
        'phi_nn_h_layer_topo':        lunar_nn,
        'psi_nn_h_layer_topo':        lunar_nn,
        'note':                       'Only good result batch_size_in_ts=100, '
                                      'but it is unconclusive cause could not reproduce'
        }
    )
# Experiment >>>  todo: LunarLander asses the proper batch_size_in_ts,
# experiment_buffer.append(SAC_LunarLander_BatchSize_hparam)


SAC_LunarLander_nnArchitecture_hparam = dict(SAC_LunarLander_base_hparam)
for nn in [(64,), (64, 64), (82, 82), ]:
    nnstr = str(nn).strip('()')
    nnstr = nnstr.replace(', ', 'X')
    SAC_LunarLander_nnArchitecture_hparam.update(
        {
            # 'rerun_tag':                    'LunarLanderFastStart',
            #  done: [(64,), (200,), (400,), (64, 64),(100, 100)]   with SmallPool
            'rerun_tag':                    'LunarLanderBiggerPoolBigStep',
            'comment':                      'nnArchitecture{}'.format(nnstr),
            'max_epoch':                    75,
            'max_gradient_step_expected':   375000,
            'pool_capacity':                int(1e6),  # SAC paper & SpinningUp: 1e6
            'min_pool_size':                40000,  # SpinningUp: 10000
            'theta_nn_h_layer_topo':        nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'phi_nn_h_layer_topo':          nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'psi_nn_h_layer_topo':          nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'target_smoothing_coefficient': 0.005,  # SAC paper: 0.005 or 1.0  SpiningUp: 0.995,
            'target_update_interval':       1,  # SAC paper: 1 or 1000 SpinningUp: all T.U. performed at trj end
            'gradient_step_interval':       1,  # SAC paper: 1 or 4 SpinningUp: all G. step performed at trj end
            'note':                         'Evaluate for promissing start for nn archicture (over 15 epoch)'
                                            'Most pormissing 100X100',
            }
        )
    # Experiment >>> done: LunarLander BiggerPool BigStep, rerun=1 todo:  "rerun 2"
    # experiment_buffer.append(SAC_LunarLander_nnArchitecture_hparam.copy())

SAC_LunarLander_nnAndPoolAndGradientStep_hparam = dict(SAC_LunarLander_base_hparam)
for nn in [(64,), (200,), (400,), (64, 64)]:
    nnstr = str(nn).strip('()')
    nnstr = nnstr.replace(', ', 'X')
    SAC_LunarLander_nnAndPoolAndGradientStep_hparam.update(
        {
            'rerun_tag':                  'LunarLanderSmallPoolFastRenewal',
            'comment':                    'nnArchitecture{}'.format(nnstr),
            'max_epoch':                  50,
            'max_gradient_step_expected': 250000,
            'pool_capacity':              int(1e5),  # SAC paper & SpinningUp: 1e6
            'theta_nn_h_layer_topo':      nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'phi_nn_h_layer_topo':        nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'psi_nn_h_layer_topo':        nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'note':                       '',
            }
        )
    # Experiment >>> (STANDBY) todo: LunarLander asses Nn Architecture, "rerun 5" done: [ (300, 300),]
    # experiment_buffer.append(SAC_LunarLander_nnAndPoolAndGradientStep_hparam.copy())

# ::: BipedalWalker experiment :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 'BipedalWalker-v2'
# - action_space:  Box(4,)
#    - high: [1. 1. 1. 1. ]
#    - low: [-1. -1. -1. -1.]
# - observation_space:  Box(24,)
#    - high: [inf ... ]
#    - low: [-inf ... ]
# - reward_range:  (-inf, inf)
# - spec:
#    - max_episode_steps: 1600
#    - reward_threshold: 300       #  The reward threshold before the task is considered solved
# - Avg reward assement: -0.05886533021377718 & -0.07178085108315299 (for Hardcore version)
#
# Note: inspected in interactive console via a playground<envName>.env  Gym instance property
# Gym: https://gym.openai.com/envs/ ...
SAC_BiWalker_base_hparam = dict(SAC_base_hparam)
SAC_BiWalker_base_hparam.update(
    {
        'rerun_tag':            'BiWalker',
        'comment':              'base',
        'prefered_environment': 'BipedalWalker-v2',
        'expected_reward_goal': 290,  # goal: 300
        'max_trj_steps':        1600,
        'note':                 '',
        }
    )

SAC_BiWalker_rewardScale_hparam = dict(SAC_BiWalker_base_hparam)
SAC_BiWalker_rewardScale_hparam.update(
    {
        'rerun_tag':                  SAC_BiWalker_base_hparam['rerun_tag'] + 'RewS-Pool2e5',
        'comment':                    'rewardScale',
        'max_epoch':                  100,
        'timestep_per_epoch':         5000,
        'max_gradient_step_expected': 500000,
        'reward_scaling':             [25.0, 1.0],
        'pool_capacity':              int(2e5),
        'min_pool_size':              35000,  # SpinningUp: 10000
        'batch_size_in_ts':           200,  # SAC paper:256, SpinningUp:100
        'theta_nn_h_layer_topo':      (256, 256),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'phi_nn_h_layer_topo':        (256, 256),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'psi_nn_h_layer_topo':        (256, 256),  # SAC paper:(256, 256), SpinningUp:(400, 300)
        'note':                       '',
        }
    )
# Experiment >>>  inProgress: BiWalker asses the proper reward_scaling, rerun 1 done: [100.0, 75.0, 40.0,] rerun 1
# experiment_buffer.append(SAC_BiWalker_rewardScale_hparam)

SAC_BiWalker_NN_architecture_hparam = dict(SAC_BiWalker_base_hparam)
for nn in [(64,), (200,), (400,), (64, 64), (100, 100), (300, 300)]:
    nnstr = str(nn).strip('()')
    nnstr = nnstr.replace(', ', 'X')
    # print('nnArchitecture{}'.format(nnstr))
    SAC_BiWalker_NN_architecture_hparam.update(
        {
            'rerun_tag':             'BiWalker',
            'comment':               'nnArchitecture{}'.format(nnstr),
            'theta_nn_h_layer_topo': nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'phi_nn_h_layer_topo':   nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'psi_nn_h_layer_topo':   nn,  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'note':                  '',
            }
        )
#     # Experiment >>> todo: BiWalker asses the proper architecture, rerun 5
#     experiment_buffer.append(SAC_BiWalker_NN_architecture_hparam.copy())

# ... Test hparam ......................................................................................................
test_hparam = dict(SAC_base_hparam)
test_hparam.update(
    {
        'rerun_tag':             'TEST-RUN',
        'comment':               'TestSpec',
        'prefered_environment':  'Pendulum-v0',
        
        'max_epoch':             3,
        'timestep_per_epoch':    220,
        'max_trj_steps':         100,
        
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
        'max_trj_steps':      50,
        'isTestRun':          False,
        }
    )
# .............................................................................................. Test hparam ...(end)...


parser = argparse.ArgumentParser(description=(
    "=============================================================================\n"
    ":: Command line option for the Soft Actor-Critic Agent.\n\n"
    "To play:\n"
    "     python -m SoftActorCritic [--playLunar, --playHardLunar, --playPendulum] "
    "[--play_for] [--help] [--testRun] [--harderEnvCoeficient]\n\n"
    "To train:\n"
    "     python -m SoftActorCritic <trainExperimentSpecification>   [--rerun] [--renderTraining] [--discounted] "
    "[--help] [--testRun]\n\n"
    "Choose <trainExperimentSpecification> between the following:\n"
    "     - 'BipedalWalker-v2' :\n"
    "       [--trainBipedalWalker]: Train on Bipedal Walker gym env a Soft Actor-Critic agent\n"
    "     - 'Pendulum-v0':\n"
    "          [--trainPendulum]: Train on Pendulum gym env a Soft Actor-Critic agent\n"
    "     - 'LunarLanderContinuous-v2' environment:\n"
    "          [--trainLunarLander]: Train on LunarLander a Soft Actor-Critic agent\n"
    "     - Experimentation utility:\n"
    "          [--trainExperimentBuffer]: Run a batch of experiment spec\n"
),
    epilog="=============================================================================\n")

# parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--trainExperimentBuffer', action='store_true',
                    help='Utility: Train a Soft Actor-Critic agent on a batch of experiment spec')

parser.add_argument('--trainBipedalWalker', action='store_true',
                    help='Train on Bipedal Walker gym env a Soft Actor-Critic agent')

parser.add_argument('--trainPendulum', action='store_true', help='Train on Pendulum a Soft Actor-Critic agent')

# (Ice-box) todo:unit-test --> problem: rerun error
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

parser.add_argument('--playHardLunar', action='store_true', help='Play a trained Soft Actor-Critic agent on a HARD '
                                                                 'version of LunarLanderContinuous-v2 environment')

parser.add_argument('--harderEnvCoeficient', type=float, default=1.6,
                    help='Harder environment coeficient (if it can be applied)')

parser.add_argument('--playPendulum', action='store_true', help='Play a trained Soft Actor-Critic agent on the '
                                                                'Pendulum environment')
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

if args.playPendulum:
    # (Ice-box) todo:implement --> load hparam dict from the config.txt
    
    SAC_Pendulum_freezed_hparam = dict(SAC_base_hparam)
    SAC_Pendulum_freezed_hparam.update(
        {
            'rerun_tag':                     'RewardScaleTest',
            'comment':                       '',
            'prefered_environment':          'Pendulum-v0',
            'expected_reward_goal':          (-561),
            'max_epoch':                     50,
            'timestep_per_epoch':            5000,
            'max_gradient_step_expected':    250000,
            'actor_lr_decay_rate':           1.0,  # Note: set to 1.0 to swith OFF scheduler
            'critic_lr_decay_rate':          1.0,  # Note: set to 1.0 to swith OFF scheduler
            'batch_size_in_ts':              100,  # SAC paper:256, SpinningUp:100
            'learning_rate':                 0.003,  # SAC paper: 30e-4 SpinningUp: 0.001
            'critic_learning_rate':          0.003,  # SAC paper: 30e-4 SpinningUp: 0.001
    
            # Note: alpha = 0 <==> SAC become a standard max expected return objective
            'alpha':                         0.2,  # SpinningUp=0.2, SAC paper=1.0
            'max_eval_trj':                  20,  #SpiningUp: 10
    
            # (!) Note: The only hparam requiring carefull tuning. Can be [0.0 < rewS <= 1.0]
            # SAC paper: 3, 10, 30  SpinningUp: 1.0
            'reward_scaling':                0.3,
    
            # Note: HARD TARGET update vs EXPONENTIAL MOVING AVERAGE (EMA)
            #  |                                        EMA         HARD TARGET
            #  |        target_smoothing_coefficient:   1.0         0.005
            #  |        target_update_interval:         1           1000
            #  |        gradient_step_interval:         1           4 (except for rlLab humanoid)
            'target_smoothing_coefficient':  0.99,
            'target_update_interval':        1,  # SAC paper: 1 or 1000 SpinningUp: all T.U. performed at trj end
            'gradient_step_interval':        1,  # SAC paper: 1 or 4 SpinningUp: all G. step performed at trj end
    
            'pool_capacity':                 int(1e6),  # SAC paper & SpinningUp: 1e6
            'min_pool_size':                 10000,  # SpinningUp: 10000
    
            'theta_nn_h_layer_topo':         (16,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'phi_nn_h_layer_topo':           (16,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
            'psi_nn_h_layer_topo':           (16,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    
            'render_env_every_What_epoch':   5,
            'render_env_eval_interval':      3,
            'print_metric_every_what_epoch': 10,
            'log_metric_interval':           50,
            'note':                          '(Proof of life) Should reach Avg Return close to ~ -560'
            }
        )

    chekpoint_dir = "Run-RewardScaleTest-reward_scaling=0.3-0-SAC()-d15h16m36s40"

    run_dir = chekpoint_dir + "/checkpoint/" + "Soft_Actor_Critic_agent--100-49"
    play_agent(run_dir, SAC_Pendulum_freezed_hparam, args, record=args.record)

elif args.playLunar:
    chekpoint_dir = "Run-LunarLander-EMA-MinPool-ModBuffer-min_pool_size=20000-2-SAC(NN200X200)-d23h15m40s48"
    run_dir = chekpoint_dir + "/goal_reached/" + "Soft_Actor_Critic-goal-285-43"
    play_agent(run_dir, SAC_LunarLander_base_Working_hparam, args, record=args.record)

elif args.playHardLunar:
    SAC_LunarLander_baseForHardEnv_hparam.update({'harderEnvCoeficient': args.harderEnvCoeficient, })
    
    chekpoint_dir = "Run-LunarLander-EMA-ForHardEnv-RewS-reward_scaling=40.0-0-SAC(LargePool)-d28h22m35s1"
    run_dir = chekpoint_dir + "/goal_reached/" + "Soft_Actor_Critic-goal-284-496"
    play_agent(run_dir, SAC_LunarLander_baseForHardEnv_hparam, args, record=args.record)

else:
    hparam = None
    key = None
    values_search_set = None
    
    # --- training ----------------------------------------------------------------------------------------------------
    experiment_start_message(consol_width, args.rerun)
    
    if args.trainBipedalWalker:
        """ ---- Easy environment [--trainBipedalWalker] ---- """
        hparam, key, values_search_set = run_experiment(SAC_BiWalker_base_hparam, args,
                                                        test_hparam, rerun_nb=args.rerun)
    
    elif args.trainPendulum:
        """ ---- Harder environment [--trainPendulum] ---- """
        hparam, key, values_search_set = run_experiment(
            SAC_Pendulum_base_hparam, args, test_hparam, rerun_nb=args.rerun)
    
    elif args.trainPendulumTESTrerun:
        """ ---- Harder environment [--trainPendulumTESTrerun] ---- """
        hparam, key, values_search_set = run_experiment(
            SAC_Pendulum_hparam_TEST_RERUN, args, test_hparam, rerun_nb=args.rerun)
    
    elif args.trainLunarLander:
        """ ---- Harder environment [--trainLunarLander] ---- """
        hparam, key, values_search_set = run_experiment(
            SAC_LunarLander_base_Working_hparam, args, test_hparam, rerun_nb=args.rerun)
    
    elif args.trainExperimentBuffer:
        """ ---- Run batch of experiment ---- """
        for expSpec in experiment_buffer:
            hparam, key, values_search_set = run_experiment(expSpec, args, test_hparam, rerun_nb=args.rerun)
    
    else:
        raise NotImplementedError
    
    # --------------------------------------------------------------------------------------------------- training ---/
    
    experiment_closing_message(hparam, args.rerun, key, values_search_set, consol_width)

exit(0)
