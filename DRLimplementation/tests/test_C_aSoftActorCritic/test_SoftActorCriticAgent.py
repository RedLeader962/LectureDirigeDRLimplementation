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

# pytestmark = pytest.mark.skip("all tests still WIP")  # (Priority) todo:implement --> coverage: then remove line

unit_test_hparam = {
    'rerun_tag':                    'Unit-TEST',
    'paramameter_set_name':         'SAC',
    'comment':                      'UnitTestSpec',  # Comment added to training folder name (can be empty)
    'algo_name':                    'Soft Actor Critic',
    'AgentType':                    SoftActorCriticAgent,
    'prefered_environment':         'Pendulum-v0',
    
    'expected_reward_goal':         90,  # Note: trigger model save on reach
    'max_epoch':                    2,
    'timestep_per_epoch':           1000,
    'max_trj_steps':                200,
    
    'reward_scaling':               5.0,
    
    'discout_factor':               0.99,  # SAC paper: 0.99
    'learning_rate':                0.003,  # SAC paper: 30e-4
    'critic_learning_rate':         0.003,  # SAC paper: 30e-4
    'max_gradient_step_expected':   600,
    'actor_lr_decay_rate':          0.01,  # Note: set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':         0.01,  # Note: set to 1 to swith OFF scheduler
    
    'target_smoothing_coefficient': 0.005,  # SAC paper: EXPONENTIAL MOVING AVERAGE ~ 0.005, 1 <==> HARD TARGET update
    'target_update_interval':       1,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    'gradient_step_interval':       1,
    
    'alpha':                        1,  # HW5: we recover a standard max expected return objective as alpha --> 0
    
    'max_eval_trj':                 2,  #SpiningUp: 10
    
    'pool_capacity':                100,  # SAC paper: 1e6
    'min_pool_size':                  80,
    'batch_size_in_ts':               20,  # SAC paper:256, SpinningUp:100
    
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
    'note':                           'My note ...'
    }


@pytest.fixture
def gym_and_tf_continuous_montain_car_setup():
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_MontainCar_hparam = dict(unit_test_hparam)
    unit_test_MontainCar_hparam.update({
        'prefered_environment': 'MountainCarContinuous-v0',
        'max_trj_steps':        999,
        })
    exp_spec.set_experiment_spec(unit_test_MontainCar_hparam)
    
    yield exp_spec
    tf_cv1.reset_default_graph()


@pytest.fixture
def gym_and_tf_continuous_pendulum_setup():
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_Pendulum_hparam = dict(unit_test_hparam)
    unit_test_Pendulum_hparam.update({'prefered_environment': 'Pendulum-v0'})
    exp_spec.set_experiment_spec(unit_test_Pendulum_hparam)
    
    yield exp_spec
    tf_cv1.reset_default_graph()


@pytest.fixture
def gym_and_tf_continuous_LunarLander_setup():
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_Lunar_hparam = dict(unit_test_hparam)
    unit_test_Lunar_hparam.update({
        'prefered_environment': 'LunarLanderContinuous-v2',
        'max_trj_steps':        1000,
        })
    exp_spec.set_experiment_spec(unit_test_Lunar_hparam)

    yield exp_spec
    tf_cv1.reset_default_graph()


@pytest.fixture
def gym_and_tf_continuous_Bipedal_setup():
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_Bipedal_hparam = dict(unit_test_hparam)
    unit_test_Bipedal_hparam.update({
        'prefered_environment': 'BipedalWalker-v2',
        'max_trj_steps':        1600,
        })
    exp_spec.set_experiment_spec(unit_test_Bipedal_hparam)
    
    yield exp_spec
    tf_cv1.reset_default_graph()


# --- Soft Actor-Critic agent ------------------------------------------------------------------------------------------
# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_SoftActorCritic_agent_INSTANTIATE_AGENT_MontainCar_PASS(gym_and_tf_continuous_montain_car_setup):
    exp_spec = gym_and_tf_continuous_montain_car_setup
    sac_agent_montaincar = SoftActorCriticAgent(exp_spec)


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_SoftActorCritic_agent_INSTANTIATE_AGENT_Pendulum_PASS(gym_and_tf_continuous_pendulum_setup):
    exp_spec = gym_and_tf_continuous_pendulum_setup
    sac_agent_pendulum = SoftActorCriticAgent(exp_spec)


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_SoftActorCritic_agent_INSTANTIATE_AGENT_Lunar_PASS(gym_and_tf_continuous_LunarLander_setup):
    exp_spec = gym_and_tf_continuous_LunarLander_setup
    sac_agent_lunar = SoftActorCriticAgent(exp_spec)


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_SoftActorCritic_agent_TRAIN_AGENT_MontainCar_PASS(gym_and_tf_continuous_montain_car_setup):
    exp_spec = gym_and_tf_continuous_montain_car_setup
    sac_agent_montaincar = SoftActorCriticAgent(exp_spec)
    sac_agent_montaincar.train()


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_SoftActorCritic_agent_TRAIN_AGENT_Pendulum_PASS(gym_and_tf_continuous_pendulum_setup):
    exp_spec = gym_and_tf_continuous_pendulum_setup
    sac_agent_pendulum = SoftActorCriticAgent(exp_spec)
    sac_agent_pendulum.train()


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_SoftActorCritic_agent_TRAIN_AGENT_Lunar_PASS(gym_and_tf_continuous_LunarLander_setup):
    exp_spec = gym_and_tf_continuous_LunarLander_setup
    sac_agent_lunar = SoftActorCriticAgent(exp_spec)
    sac_agent_lunar.train()


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_SoftActorCritic_agent_TRAIN_AGENT_Bipedal_PASS(gym_and_tf_continuous_Bipedal_setup):
    exp_spec = gym_and_tf_continuous_Bipedal_setup
    sac_agent_bipedal = SoftActorCriticAgent(exp_spec)
    sac_agent_bipedal.train()
