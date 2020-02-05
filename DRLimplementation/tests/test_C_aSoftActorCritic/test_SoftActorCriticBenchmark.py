# coding=utf-8
import pytest
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

from SoftActorCritic import SoftActorCriticBrain, SoftActorCriticAgent
from blocAndTools import buildingbloc as bloc
from blocAndTools.rl_vocabulary import rl_name

tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()

sac_benchmark_hparam = {
    'rerun_tag':                      'SAC-Benchmark',
    'paramameter_set_name':           'SAC',
    'comment':                        '',
    'algo_name':                      'Soft Actor Critic',
    'AgentType':                      SoftActorCriticAgent,
    'prefered_environment':           'LunarLanderContinuous-v2',
    'max_trj_steps':                  1000,
    
    'expected_reward_goal':           90,  # Note: trigger model save on reach
    'max_epoch':                      10,
    'timestep_per_epoch':             1000,
    
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
    'min_pool_size':                  500,
    'batch_size_in_ts':               200,  # SAC paper:256, SpinningUp:100
    
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


def train_SAC_on_Lunar():
    exp_spec = bloc.ExperimentSpec()
    exp_spec.set_experiment_spec(sac_benchmark_hparam)
    sac_agent_lunar = SoftActorCriticAgent(exp_spec)
    sac_agent_lunar.train()
    tf_cv1.reset_default_graph()
    return None


# --- Soft Actor-Critic agent ------------------------------------------------------------------------------------------
# pytestmark = pytest.mark.skip("Skip SAC benchmark")

def test_benchmark_SAC(benchmark):
    benchmark(train_SAC_on_Lunar)
