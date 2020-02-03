# coding=utf-8
import pytest
import tensorflow as tf

from ActorCritic import ActorCriticBrainSplitNetwork
from ActorCritic.BatchActorCriticAgent import BatchActorCriticAgent
from ActorCritic.OnlineActorCriticAgent import OnlineActorCriticAgent
from ActorCritic.OnlineTwoInputAdvantageActorCriticAgent import OnlineTwoInputAdvantageActorCriticAgent
from blocAndTools.rl_vocabulary import TargetType, NetworkType

from blocAndTools import buildingbloc as bloc

tf_cv1 = tf.compat.v1  # shortcut

test_AC_discrete_hparam = {
    'paramameter_set_name':           'Batch-AAC',
    'rerun_tag':                      'TEST-RUN-H',
    'algo_name':                      'Batch ActorCritic',
    'comment':                        'TestSpec',
    'AgentType':                      BatchActorCriticAgent,
    'Target':                         TargetType.MonteCarlo,
    'Network':                        NetworkType.Split,
    # 'prefered_environment':           'CartPole-v0',
    # 'batch_size_in_ts':               300,
    'prefered_environment':           'LunarLander-v2',
    # 'batch_size_in_ts':               1000,
    'batch_size_in_ts':               300,
    'expected_reward_goal':           200,  # for LunarLander-v2
    'max_epoch':                      4,
    'discounted_reward_to_go':        True,
    'discout_factor':                 0.999,
    # 'discout_factor':                 [0.999, 0.91],
    'learning_rate':                  3e-4,
    # 'learning_rate':                  [3e-4, 1e-3],
    'critic_learning_rate':           1e-3,
    'actor_lr_decay_rate':            1,  # set to 1 to swith OFF scheduler
    'critic_lr_decay_rate':           1,  # set to 1 to swith OFF scheduler
    'critique_loop_len':              5,
    'theta_nn_h_layer_topo':          (2,),
    # 'theta_nn_h_layer_topo':          [(4, 4), (6, 6)],
    'random_seed':                    0,
    'theta_hidden_layers_activation': tf.nn.tanh,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch':    5,
    'print_metric_every_what_epoch':  5,
    'isTestRun':                      True,
    'show_plot':                      False,
    }


@pytest.fixture(params=[NetworkType.Shared, NetworkType.Split])
def batchAC_MonteCarlo_discrete_LunarLander_setup(request):
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_Discrete_Lunar_hparam = dict(test_AC_discrete_hparam)
    unit_test_Discrete_Lunar_hparam.update({
        'prefered_environment': 'LunarLander-v2',
        'AgentType':            BatchActorCriticAgent,
        'Target':               TargetType.MonteCarlo,
        'Network':              request.param,
        })
    exp_spec.set_experiment_spec(unit_test_Discrete_Lunar_hparam)
    
    yield exp_spec
    tf_cv1.reset_default_graph()


@pytest.fixture(params=[NetworkType.Shared, NetworkType.Split])
def batchAC_Bootstrap_discrete_LunarLander_setup(request):
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_Discrete_Lunar_hparam = dict(test_AC_discrete_hparam)
    unit_test_Discrete_Lunar_hparam.update({
        'prefered_environment': 'LunarLander-v2',
        'AgentType':            BatchActorCriticAgent,
        'Target':               TargetType.Bootstrap,
        'Network':              request.param,
        })
    exp_spec.set_experiment_spec(unit_test_Discrete_Lunar_hparam)
    
    yield exp_spec
    tf_cv1.reset_default_graph()


@pytest.fixture(params=[NetworkType.Shared, NetworkType.Split])
def onlineAC_discrete_LunarLander_setup(request):
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_Discrete_Lunar_hparam = dict(test_AC_discrete_hparam)
    unit_test_Discrete_Lunar_hparam.update({
        'prefered_environment': 'LunarLander-v2',
        'AgentType':            OnlineActorCriticAgent,
        'Network':              request.param,
        'stage_size_in_trj':    2,
        })
    exp_spec.set_experiment_spec(unit_test_Discrete_Lunar_hparam)
    
    yield exp_spec
    tf_cv1.reset_default_graph()


@pytest.fixture(params=[pytest.param(NetworkType.Shared, marks=pytest.mark.skip(reason="Not implemented")),
                        NetworkType.Split])
def onlineTwoInputAdvantageAC_discrete_LunarLander_setup(request):
    """
    :return: obs_t_ph, act_ph, obs_t_prime_ph, reward_t_ph, trj_done_t_ph, exp_spec, playground
    """
    exp_spec = bloc.ExperimentSpec()
    unit_test_Discrete_Lunar_hparam = dict(test_AC_discrete_hparam)
    unit_test_Discrete_Lunar_hparam.update({
        'prefered_environment': 'LunarLander-v2',
        'AgentType':            OnlineTwoInputAdvantageActorCriticAgent,
        'Network':              request.param,
        'stage_size_in_trj':    2,
        })
    exp_spec.set_experiment_spec(unit_test_Discrete_Lunar_hparam)
    
    yield exp_spec
    tf_cv1.reset_default_graph()


# @pytest.fixture
# def gym_and_tf_discrete_setup():
#     """
#     :return: (obs_p, act_p, exp_spec, playground)
#     :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
#     """
#     exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
#     exp_spec.set_experiment_spec({'critic_learning_rate': 1e-3})
#     playground = bloc.GymPlayground('LunarLander-v2')
#     obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
#                                                                                 action_shape_constraint=(1,))
#     yield obs_p, act_p, Q_values_ph, exp_spec, playground
#     tf_cv1.reset_default_graph()
#
# @pytest.fixture
# def gym_and_tf_continuous_setup():
#     """
#     :return: (obs_p, act_p, exp_spec, playground)
#     :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
#     """
#     exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
#     exp_spec.set_experiment_spec({'critic_learning_rate': 1e-3})
#
#     playground = bloc.GymPlayground('LunarLanderContinuous-v2')
#     obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
#                                                                                 action_shape_constraint=(1,))
#     yield obs_p, act_p, exp_spec, playground
#     tf_cv1.reset_default_graph()


# --- ActorCritic_agent -------------------------------------------------------------------------------------------

# ... intantiate brain .................................................................................................
# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_ActorCritic_INSTANTIATE_Batch_MONTECARLO_AGENT_Lunar_PASS(batchAC_MonteCarlo_discrete_LunarLander_setup):
    exp_spec = batchAC_MonteCarlo_discrete_LunarLander_setup
    BatchActorCriticAgent(exp_spec)


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_ActorCritic_INSTANTIATE_Batch_BOOTSTRAP_AGENT_Lunar_PASS(batchAC_Bootstrap_discrete_LunarLander_setup):
    exp_spec = batchAC_Bootstrap_discrete_LunarLander_setup
    BatchActorCriticAgent(exp_spec)


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_ActorCritic_INSTANTIATE_Online_AGENT_Lunar_PASS(onlineAC_discrete_LunarLander_setup):
    exp_spec = onlineAC_discrete_LunarLander_setup
    OnlineActorCriticAgent(exp_spec)


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_ActorCritic_INSTANTIATE_OnlineTwoInputA_AGENT_Lunar_PASS(onlineTwoInputAdvantageAC_discrete_LunarLander_setup):
    exp_spec = onlineTwoInputAdvantageAC_discrete_LunarLander_setup
    OnlineTwoInputAdvantageActorCriticAgent(exp_spec)


# ......................................................................................... intantiate brain ...(end)...

# ... train brain ......................................................................................................
# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_ActorCritic_TRAIN_Batch_MONTECARLO_AGENT_Lunar_PASS(batchAC_MonteCarlo_discrete_LunarLander_setup):
    exp_spec = batchAC_MonteCarlo_discrete_LunarLander_setup
    agent = BatchActorCriticAgent(exp_spec)
    agent.train()


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_ActorCritic_TRAIN_Batch_BOOTSTRAP_AGENT_Lunar_PASS(batchAC_Bootstrap_discrete_LunarLander_setup):
    exp_spec = batchAC_Bootstrap_discrete_LunarLander_setup
    agent = BatchActorCriticAgent(exp_spec)
    agent.train()


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_ActorCritic_TRAIN_Online_AGENT_Lunar_PASS(onlineAC_discrete_LunarLander_setup):
    exp_spec = onlineAC_discrete_LunarLander_setup
    agent = OnlineActorCriticAgent(exp_spec)
    agent.train()


# @pytest.mark.skip(reason="Work fine. Mute for now")
def test_ActorCritic_TRAIN_OnlineTwoInputA_AGENT_Lunar_PASS(onlineTwoInputAdvantageAC_discrete_LunarLander_setup):
    exp_spec = onlineTwoInputAdvantageAC_discrete_LunarLander_setup
    agent = OnlineTwoInputAdvantageActorCriticAgent(exp_spec)
    agent.train()
# .............................................................................................. train brain ...(end)...
