# coding=utf-8
import pytest
import tensorflow as tf

tf_cv1 = tf.compat.v1  # shortcut

from BasicPolicyGradient import REINFORCEagent
from blocAndTools import buildingbloc as bloc

cartpole_REINFORCE_hparam = {
    'paramameter_set_name':           'Basic',
    'algo_name':                      'REINFORCE',
    'comment':                        None,
    'prefered_environment':           'CartPole-v0',
    'expected_reward_goal':           200,
    'batch_size_in_ts':               5000,
    'max_epoch':                      40,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    'learning_rate':                  1e-2,
    'theta_nn_h_layer_topo':          (62,),
    'random_seed':                    82,
    'theta_hidden_layers_activation': tf.nn.tanh,  # tf.nn.relu,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    100,
    'print_metric_every_what_epoch':  2,
    'isTestRun':                      True,
    'show_plot':                      False,
    }


@pytest.fixture
def reinforce_PLAY_setup():
    exp_spec = bloc.ExperimentSpec()
    exp_spec.set_experiment_spec(cartpole_REINFORCE_hparam)
    yield exp_spec
    tf_cv1.reset_default_graph()


def test_PLAY_AGENT_PASS(reinforce_PLAY_setup):
    exp_spec = reinforce_PLAY_setup
    
    # import os
    # print(os.getcwd())
    
    agent = REINFORCEagent(exp_spec,
                           agent_root_dir="/Users/redleader/PycharmProjects/LectureDirigeDRLimplementation"
                                          "/DRLimplementation/tests/test_A_unit")
    agent.play(run_name='REINFORCE_agent-39', max_trajectories=2)

# def test__instantiate_data_collector():
#     assert False
#
#
# def test__render_trajectory_on_condition():
#     assert False
#
#
# def test__save_learned_model():
#     assert False
#
#
# def test__save_checkpoint():
#     assert False
#
# def test_load_selected_trained_agent():
#     assert False
