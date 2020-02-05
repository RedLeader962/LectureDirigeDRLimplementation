# coding=utf-8
import os

import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

from SoftActorCritic import SoftActorCriticBrain, SoftActorCriticAgent
from blocAndTools import buildingbloc as bloc
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()
#
# ROOT_DIRECTORY = "DRLimplementation"
# TARGET_WORKING_DIRECTORY = ROOT_DIRECTORY
#
#
# def set_up_cwd(initial_CWD):
#     print("\n:: START set_up_cwd, Initial was: ", initial_CWD)
#
#     path_basename = os.path.basename(initial_CWD)
#
#     if path_basename == TARGET_WORKING_DIRECTORY:
#         pass
#     elif path_basename == ROOT_DIRECTORY:
#         # then we must get one level down in directory tree
#         os.chdir(TARGET_WORKING_DIRECTORY)
#         print(":: change cwd to: ", os.getcwd())
#     else:
#         # we are to deep in directory tree
#         while os.path.basename(os.getcwd()) != TARGET_WORKING_DIRECTORY:
#             os.chdir("..")
#             print(":: CD to parent")
#     return None
#
#
# def return_to_initial_working_directory(initial_CWD):
#     print(":: return_to_initial_working_directory")
#     if os.path.basename(os.getcwd()) != os.path.basename(initial_CWD):
#         os.chdir(initial_CWD)
#         print(":: change cwd to: ", os.getcwd())
#     print(":: Teardown END\n")
#     return None
#
#
# def set_up_PWD_to_project_root():
#     initial_CWD = os.getcwd()
#     set_up_cwd(initial_CWD)
#
#     yield
#
#     return_to_initial_working_directory(initial_CWD)


unit_test_hparam = {
    'rerun_tag':                      'Profiling',
    'paramameter_set_name':           'SAC',
    'comment':                        '',
    'algo_name':                      'Soft Actor Critic',
    'AgentType':                      SoftActorCriticAgent,
    # 'prefered_environment':           'Pendulum-v0',
    
    'expected_reward_goal':           90,  # Note: trigger model save on reach
    'max_epoch':                      10,
    'timestep_per_epoch':             1000,
    'max_trj_steps':                  1000,
    
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
    
    'pool_capacity':                  5000,  # SAC paper: 1e6
    'min_pool_size':                  2000,
    'batch_size_in_ts':               200,  # SAC paper:256, SpinningUp:100
    
    'theta_nn_h_layer_topo':          (4,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'theta_hidden_layers_activation': tf.nn.relu,
    'theta_output_layers_activation': None,
    'phi_nn_h_layer_topo':            (4,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'phi_hidden_layers_activation':   tf.nn.relu,
    'phi_output_layers_activation':   None,
    'psi_nn_h_layer_topo':            (4,),  # SAC paper:(256, 256), SpinningUp:(400, 300)
    'psi_hidden_layers_activation':   tf.nn.relu,
    'psi_output_layers_activation':   None,
    
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'random_seed':                    0,  # Note: 0 --> turned OFF (default)
    'isTestRun':                      False,
    'show_plot':                      False,
    'note':                           'My note ...',
    }


# --- Soft Actor-Critic agent ------------------------------------------------------------------------------------------


def profile_SAC_training_on_Lunar() -> None:
    exp_spec = bloc.ExperimentSpec()
    unit_test_Lunar_hparam = dict(unit_test_hparam)
    unit_test_Lunar_hparam.update({
        'prefered_environment': 'LunarLanderContinuous-v2',
        'max_trj_steps':        1000,
        })
    exp_spec.set_experiment_spec(unit_test_Lunar_hparam)
    sac_agent_lunar = SoftActorCriticAgent(exp_spec)
    sac_agent_lunar.train()
    # tf_cv1.reset_default_graph()
    sac_agent_lunar.__del__()
    return None


profile_SAC_training_on_Lunar()
