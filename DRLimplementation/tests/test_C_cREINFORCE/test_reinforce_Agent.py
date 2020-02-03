# coding=utf-8
import pytest
import tensorflow as tf

from BasicPolicyGradient import REINFORCEagent
from blocAndTools.rl_vocabulary import TargetType, NetworkType
from blocAndTools import buildingbloc as bloc

tf_cv1 = tf.compat.v1  # shortcut

test_reinforce_hparam = {
    'paramameter_set_name':           'Basic',
    'algo_name':                      'REINFORCE',
    'comment':                        'TestSpec',
    # 'prefered_environment':           'CartPole-v0',
    # 'prefered_environment':           'LunarLanderContinuous-v2',
    # 'prefered_environment':           'LunarLander-v2',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               300,
    'max_epoch':                      2,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-2,
    'theta_nn_h_layer_topo':          (2,),
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      True,
    'show_plot':                      False,
    }


@pytest.fixture(params=['CartPole-v0',
                        'LunarLander-v2',
                        pytest.param('LunarLanderContinuous-v2',
                                     marks=pytest.mark.skip(reason="Continuous policy not implemented")),
                        ])
def reinforce_setup(request):
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_Discrete_cartpole_hparam = dict(test_reinforce_hparam)
    unit_test_Discrete_cartpole_hparam.update({
        'prefered_environment': request.param,
        })
    exp_spec.set_experiment_spec(unit_test_Discrete_cartpole_hparam)
    
    yield exp_spec
    tf_cv1.reset_default_graph()


#
# cartpole_hparam = {
#         'paramameter_set_name':           'Basic',
#         'algo_name':                      'REINFORCE',
#         'comment':                        None,
#         'prefered_environment':           'CartPole-v0',
#         'expected_reward_goal':           200,
#         'batch_size_in_ts':               5000,
#         'max_epoch':                      40,
#         'discounted_reward_to_go':        True,
#         'discout_factor':                 0.999,
#         'learning_rate':                  1e-2,
#         'theta_nn_h_layer_topo':          (62,),
#         'random_seed':                    82,
#         'theta_hidden_layers_activation': tf.nn.tanh,  # tf.nn.relu,
#         'theta_output_layers_activation': None,
#         'render_env_every_What_epoch':    100,
#         'print_metric_every_what_epoch':  2,
#         'isTestRun':                      True,
#         'show_plot':                      False,
#         }
#
# @pytest.fixture
# def reinforce_PLAY_setup():
#     """
#     :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
#     """
#     exp_spec = bloc.ExperimentSpec()
#     # unit_test_Discrete_cartpole_hparam = dict(cartpole_hparam)
#     # unit_test_Discrete_cartpole_hparam.update({
#     #     })
#
#     exp_spec.set_experiment_spec(cartpole_hparam)
#     yield exp_spec
#     tf_cv1.reset_default_graph()


# --- reinforce agent -------------------------------------------------------------------------------------------

# ... intantiate brain .................................................................................................
# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_reinforce_INSTANTIATE_AGENT_PASS(reinforce_setup):
    exp_spec = reinforce_setup
    REINFORCEagent(exp_spec)


# ......................................................................................... intantiate brain ...(end)...

# ... train brain ......................................................................................................
# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_reinforce_TRAIN_AGENT_PASS(reinforce_setup):
    exp_spec = reinforce_setup
    agent = REINFORCEagent(exp_spec)
    agent.train()

# .............................................................................................. train brain ...(end)...
