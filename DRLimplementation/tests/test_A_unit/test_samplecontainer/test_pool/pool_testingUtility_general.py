# coding=utf-8
from typing import Type, Union

import tensorflow as tf

from SoftActorCritic import SoftActorCriticAgent
from blocAndTools import PoolManager, Fast_PoolManager, ExperimentSpec
from blocAndTools.buildingbloc import GymPlayground

# note: exp_spec key specific to SAC
#   |   'pool_capacity'

pool_BENCHMARK_hparam = {
    # required for pool testing
    'max_trj_steps':                  200,
    'min_pool_size':                  400,
    'batch_size_in_ts':               200,  # SAC paper:256, SpinningUp:100
    'timestep_per_epoch':             1000,
    
    # All the above are NOT required for pool testing
    'pool_capacity':                  600,  # SAC paper: 1e6
    'max_epoch':                      10,
    
    'rerun_tag':                      'TrjPool-Benchmark',
    'paramameter_set_name':           'SAC',
    'algo_name':                      'Soft Actor Critic',
    'AgentType':                      SoftActorCriticAgent,
    'prefered_environment':           'LunarLanderContinuous-v2',
    
    'expected_reward_goal':           90,  # Note: trigger model save on reach
    
    'reward_scaling':                 5.0,
    
    'discout_factor':                 0.99,  # SAC paper: 0.99
    'learning_rate':                  0.003,  # SAC paper: 30e-4
    'critic_learning_rate':           0.003,  # SAC paper: 30e-4
    'max_gradient_step_expected':     600,
    'actor_lr_decay_rate':            0.01,  # Note: set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           0.01,  # Note: set to 1 to swith OFF scheduler
    
    'target_smoothing_coefficient':   0.005,  # SAC paper: EXPONENTIAL MOVING AVERAGE ~ 0.005, 1 <==> HARD TARGET update
    'target_update_interval':         1,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    'gradient_step_interval':         1,
    
    'alpha':                          1,  # HW5: we recover a standard max expected return objective as alpha --> 0
    
    'max_eval_trj':                   10,  #SpiningUp: 10
    
    'theta_nn_h_layer_topo':          (2,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'theta_hidden_layers_activation': tf.nn.relu,
    'theta_output_layers_activation': None,
    'phi_nn_h_layer_topo':            (2,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'phi_hidden_layers_activation':   tf.nn.relu,
    'phi_output_layers_activation':   None,
    'psi_nn_h_layer_topo':            (2,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'psi_hidden_layers_activation':   tf.nn.relu,
    'psi_output_layers_activation':   None,
    
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'random_seed':                    0,  # Note: 0 --> turned OFF (default)
    'isTestRun':                      True,
    'show_plot':                      False,
    'note':                           'My note ...',
    }


def instantiate_top_component(manager: Type[Union[PoolManager, Fast_PoolManager]]) -> (ExperimentSpec, GymPlayground):
    exp_spec = ExperimentSpec()
    pool_fixture_hparam = dict(pool_BENCHMARK_hparam)
    pool_fixture_hparam.update({'comment': ">>> assess " + manager.__name__ + " class"})
    exp_spec.set_experiment_spec(pool_fixture_hparam)
    playground = GymPlayground(exp_spec.prefered_environment)
    return exp_spec, playground


def print_final_pool_sideEffect(exp_spec, poolmanager):
    print(":: Current pool size >>>", poolmanager.current_pool_size)
    taking_gardient_step = poolmanager.timestep_collected_so_far() - exp_spec['min_pool_size']
    print(":: Timestep collected so far >>>", poolmanager.timestep_collected_so_far(),
          "over", poolmanager.trj_collected_so_far(),
          "trajectories. Gardient step >>>", taking_gardient_step,
          )
