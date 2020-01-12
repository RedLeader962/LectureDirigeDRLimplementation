# coding=utf-8
import pytest
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

from SoftActorCritic import SoftActorCriticBrain, SoftActorCriticAgent, critic_learning_rate_scheduler
from blocAndTools import buildingbloc as bloc
from blocAndTools.rl_vocabulary import rl_name

# noinspection DuplicatedCode
tf_cv1 = tf.compat.v1  # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()

unit_test_hparam = {
    'rerun_tag':                    'Unit-TEST',
    'paramameter_set_name':         'SAC',
    'comment':                      'UnitTestSpec',  # Comment added to training folder name (can be empty)
    'algo_name':                    'Soft Actor Critic',
    'AgentType':                    SoftActorCriticAgent,
    'prefered_environment':         'MountainCarContinuous-v0',
    
    'expected_reward_goal':         90,  # Note: trigger model save on reach
    'max_epoch':                    10,
    'timestep_per_epoch':           500,
    
    'reward_scaling':               5.0,
    
    'discout_factor':               0.99,  # SAC paper: 0.99
    'learning_rate':                0.003,  # SAC paper: 30e-4
    'critic_learning_rate':         0.003,  # SAC paper: 30e-4
    'max_gradient_step_expected':   500000,
    'actor_lr_decay_rate':          0.01,  # Note: set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':         0.01,  # Note: set to 1 to swith OFF scheduler
    
    'target_smoothing_coefficient': 0.005,  # SAC paper: EXPONENTIAL MOVING AVERAGE ~ 0.005, 1 <==> HARD TARGET update
    'target_update_interval':       1,  # SAC paper: 1 for EXPONENTIAL MOVING AVERAGE, 1000 for HARD TARGET update
    'gradient_step_interval':       1,
    
    'alpha':                        1,  # HW5: we recover a standard max expected return objective as alpha --> 0
    
    'max_eval_trj':                 10,  #SpiningUp: 10
    
    'pool_capacity':                int(1e6),  # SAC paper: 1e6
    'min_pool_size':                100,
    'batch_size_in_ts':             100,  # SAC paper:256, SpinningUp:100
    
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


@pytest.fixture
def gym_and_tf_continuous_setup():
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    exp_spec.set_experiment_spec(unit_test_hparam)

    playground = bloc.GymPlayground('LunarLanderContinuous-v2')
    obs_t_ph, act_ph, _ = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    obs_t_prime_ph = bloc.continuous_space_placeholder(space=playground.OBSERVATION_SPACE,
                                                       name=vocab.obs_tPrime_ph)
    reward_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.rew_ph)
    trj_done_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.trj_done_ph)

    yield obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    tf_cv1.reset_default_graph()


@pytest.fixture
def gym_and_KERAS_DEV_continuous_setup():
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    SoftActorCriticBrain.USE_KERAS_LAYER = False
    
    exp_spec = bloc.ExperimentSpec()
    exp_spec.set_experiment_spec(unit_test_hparam)
    
    playground = bloc.GymPlayground('LunarLanderContinuous-v2')
    obs_t_ph, act_ph, _ = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    obs_t_prime_ph = bloc.continuous_space_placeholder(space=playground.OBSERVATION_SPACE,
                                                       name=vocab.obs_tPrime_ph)
    reward_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.rew_ph)
    trj_done_t_ph = tf_cv1.placeholder(dtype=tf.float32, shape=(None,), name=vocab.trj_done_ph)
    
    yield obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    tf_cv1.reset_default_graph()
    SoftActorCriticBrain.USE_KERAS_LAYER = True


# --- ActorCritic_agent -------------------------------------------------------------------------------------------

def test_SoftActorCritic_brain_tensor_entity_call_warning_investigation_PASS(gym_and_KERAS_DEV_continuous_setup):
    obs_t_ph, _, _, _, _, exp_spec, playground = gym_and_KERAS_DEV_continuous_setup
    exp_spec.set_experiment_spec({'phi_nn_h_layer_topo': (2, 2)})
    
    pi, pi_log_p, policy_mu = SoftActorCriticBrain.build_gaussian_policy_graph(obs_t_ph, exp_spec,
                                                                               playground)


# @pytest.mark.skip(reason="Temp: Mute for now")
def test_SoftActorCritic_brain_Actor_Pi_BUILD_PASS(gym_and_tf_continuous_setup):
    obs_t_ph, _, _, _, _, exp_spec, playground = gym_and_tf_continuous_setup
    pi, pi_log_p, policy_mu = SoftActorCriticBrain.build_gaussian_policy_graph(obs_t_ph, exp_spec, playground)


# @pytest.mark.skip(reason="Temp: Mute for now")
def test_SoftActorCritic_brain_Critic_V_BUILD_PASS(gym_and_tf_continuous_setup):
    obs_t_ph, _, obs_t_prime_ph, _, _, exp_spec, _ = gym_and_tf_continuous_setup
    V_psi, V_psi_frozen = SoftActorCriticBrain.build_critic_graph_v_psi(obs_t_ph, obs_t_prime_ph, exp_spec)


# @pytest.mark.skip(reason="Temp: Mute for now")
def test_SoftActorCritic_brain_Critic_Q_BUILD_PASS(gym_and_tf_continuous_setup):
    obs_t_ph, act_ph, _, _, _, exp_spec, _ = gym_and_tf_continuous_setup
    Q_theta_1, Q_theta_2 = SoftActorCriticBrain.build_critic_graph_q_theta(obs_t_ph, act_ph, exp_spec)


# @pytest.mark.skip(reason="Temp: Mute for now")
def test_SoftActorCritic_brain_Critic_V_TRAIN_PASS(gym_and_tf_continuous_setup):
    continuous_setup = gym_and_tf_continuous_setup
    obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground = continuous_setup
    
    critic_lr_schedule, critic_global_grad_step = critic_learning_rate_scheduler(exp_spec)
    
    pi, pi_log_p, policy_mu = SoftActorCriticBrain.build_gaussian_policy_graph(obs_t_ph, exp_spec, playground)
    V_psi, V_psi_frozen = SoftActorCriticBrain.build_critic_graph_v_psi(obs_t_ph, obs_t_prime_ph, exp_spec)
    Q_theta_1, Q_theta_2 = SoftActorCriticBrain.build_critic_graph_q_theta(obs_t_ph, act_ph, exp_spec)
    
    V_psi_loss, V_psi_optimizer, V_psi_frozen_update_ops = SoftActorCriticBrain.critic_v_psi_train(V_psi, V_psi_frozen,
                                                                                                   Q_theta_1, Q_theta_2,
                                                                                                   pi_log_p, exp_spec,
                                                                                                   critic_lr_schedule,
                                                                                                   critic_global_grad_step)


# @pytest.mark.skip(reason="Temp: Mute for now")
def test_SoftActorCritic_brain_Critic_Q_TRAIN_PASS(gym_and_tf_continuous_setup):
    continuous_setup = gym_and_tf_continuous_setup
    obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground = continuous_setup
    
    pi, pi_log_p, policy_mu = SoftActorCriticBrain.build_gaussian_policy_graph(obs_t_ph, exp_spec, playground)
    V_psi, V_psi_frozen = SoftActorCriticBrain.build_critic_graph_v_psi(obs_t_ph, obs_t_prime_ph, exp_spec)
    Q_theta_1, Q_theta_2 = SoftActorCriticBrain.build_critic_graph_q_theta(obs_t_ph, act_ph, exp_spec)
    
    critic_lr_schedule, critic_global_grad_step = critic_learning_rate_scheduler(exp_spec)
    
    q_theta_train_ops = SoftActorCriticBrain.critic_q_theta_train(V_psi_frozen, Q_theta_1, Q_theta_2, reward_t_ph,
                                                                  trj_done_t_ph, exp_spec, critic_lr_schedule,
                                                                  critic_global_grad_step)


# @pytest.mark.skip(reason="Temp: Mute for now")
def test_SoftActorCritic_brain_Actor_Pi_TRAIN_PASS(gym_and_tf_continuous_setup):
    continuous_setup = gym_and_tf_continuous_setup
    obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground = continuous_setup
    
    pi, pi_log_p, policy_mu = SoftActorCriticBrain.build_gaussian_policy_graph(obs_t_ph, exp_spec, playground)
    V_psi, V_psi_frozen = SoftActorCriticBrain.build_critic_graph_v_psi(obs_t_ph, obs_t_prime_ph, exp_spec)
    Q_theta_1, Q_theta_2 = SoftActorCriticBrain.build_critic_graph_q_theta(obs_t_ph, act_ph, exp_spec)
    
    actor_kl_loss, actor_policy_optimizer_op = SoftActorCriticBrain.actor_train(pi_log_p,
                                                                                Q_theta_1, Q_theta_2, exp_spec)
