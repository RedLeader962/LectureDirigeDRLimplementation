# coding=utf-8
import sys

import gym
import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

import blocAndTools.tensorflowbloc

tf_cv1 = tf.compat.v1  # shortcut

from blocAndTools import buildingbloc as bloc, rewardtogo as rtg, visualisationtools
from blocAndTools.container.samplecontainer import TrajectoryCollector, UniformBatchCollector
from blocAndTools.rl_vocabulary import rl_name

vocab = rl_name()


#  todo:implement --> fixture with yield (aka: setup & teardown) fct for TF graph.
#                         Execute:
#                             sess = tf.InteractiveSession()
#                             sess.run(tf.global_variables_initializer())
#                             tf.reset_default_graph()
#                             sess.close()

# ---- setup & teardown -------------------------------------------------------------------------------------------

@pytest.fixture
def tf_setup():
    """
    :return: (in_p, out_p, nn_shape)
    :rtype: (tf.Tensor, tf.Tensor, list)
    """
    in_p = tf_cv1.placeholder(tf.float32, shape=(None, 8))
    out_p = tf_cv1.placeholder(tf.float32, shape=(None, 2))
    nn_shape = (2, 2)
    return in_p, out_p, nn_shape

@pytest.fixture
def gym_continuous_setup():
    """
    :return: (exp_spec, playground)
    :rtype: (ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    playground = bloc.GymPlayground('LunarLanderContinuous-v2')
    yield exp_spec, playground
    tf_cv1.reset_default_graph()

@pytest.fixture
def gym_discrete_setup():
    """
    :return: (exp_spec, playground)
    :rtype: (ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    playground = bloc.GymPlayground('LunarLander-v2')
    yield exp_spec, playground
    tf_cv1.reset_default_graph()

@pytest.fixture
def gym_and_tf_continuous_setup():
    """
    :return: (obs_p, act_p, exp_spec, playground)
    :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    playground = bloc.GymPlayground('LunarLanderContinuous-v2')
    obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                action_shape_constraint=(1,))
    yield obs_p, act_p, exp_spec, playground
    tf_cv1.reset_default_graph()

@pytest.fixture
def gym_and_tf_discrete_setup():
    """
    :return: (obs_p, act_p, exp_spec, playground)
    :rtype: (tf.Tensor, tf.Tensor, ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(batch_size_in_ts=1000, max_epoch=2, theta_nn_hidden_layer_topology=(2, 2))
    playground = bloc.GymPlayground('LunarLander-v2')
    obs_p, act_p, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                action_shape_constraint=(1,))
    yield obs_p, act_p, exp_spec, playground
    tf_cv1.reset_default_graph()



# --- ExperimentSpec ---------------------------------------------------------------------------------------------

def test_ExperimentSpec_init_ENV_TOPOLOGY_FAIL():
    with pytest.raises(Exception):
        bloc.ExperimentSpec(theta_nn_hidden_layer_topology=[1, ])


def test_set_experiment_spec_PASS(gym_discrete_setup):
    (exp_spec, playground) = gym_discrete_setup

    parma_dict = {
        'batch_size_in_ts':10,
        'max_epoch': 10,
        'discout_factor': 0.5,
        'learning_rate': 10,
        'theta_nn_h_layer_topo': (10, 10),
        'random_seed': 10,
        'theta_hidden_layers_activation': tf.tanh,
        'theta_output_layers_activation': tf.tanh
    }

    exp_spec.set_experiment_spec(parma_dict)



# ---- playground ------------------------------------------------------------------------------------------------

def test_Playground_init_ENV_FAIL():
    with pytest.raises(Exception):
        bloc.GymPlayground('UnExistingEnvironment!!!')


def test_Playground_continuous():
    play = bloc.GymPlayground('LunarLanderContinuous-v2')
    assert play.ACTION_SPACE.shape == (2,)
    assert play.ACTION_CHOICES == 2
    assert play.OBSERVATION_SPACE.shape == (8,)
    assert play.OBSERVATION_DIM == 8


def test_Playground_continuous_Hard_Lunar():
    play = bloc.GymPlayground('LunarLanderContinuous-v2', harderEnvCoeficient=1.5)
    assert play.ACTION_SPACE.shape == (2,)
    assert play.ACTION_CHOICES == 2
    assert play.OBSERVATION_SPACE.shape == (8,)
    assert play.OBSERVATION_DIM == 8


def test_Playground_continuous_Hard_no_env_FAIL():
    with pytest.raises(Exception):
        play = bloc.GymPlayground('Pendulum-v0', harderEnvCoeficient=1.5)


def test_Playground_discreet():
    play = bloc.GymPlayground('LunarLander-v2')
    assert play.ACTION_CHOICES == 4
    assert play.OBSERVATION_DIM == 8


# ---- gym_playground_to_tensorflow_graph_adapter --------------------------------------------------------------------

def test_gym_env_to_tf_graph_adapter_WRONG_IMPORT_TYPE():
    with pytest.raises(AssertionError):
        bloc.gym_playground_to_tensorflow_graph_adapter(gym, (1,))

def test_gym_env_to_tf_graph_adapter_DISCRETE_PASS(gym_discrete_setup):
    _, playground = gym_discrete_setup
    input_placeholder, output_placeholder, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                                         action_shape_constraint=(
                                                                                                         1,))
    assert input_placeholder.shape[-1] == playground.OBSERVATION_SPACE.shape[0]
    print(output_placeholder.shape)
    assert output_placeholder.shape.rank == 1

def test_gym_env_to_tf_graph_adapter_CONTINUOUS_PASS(gym_continuous_setup):
    _, playground = gym_continuous_setup
    input_placeholder, output_placeholder, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                                         action_shape_constraint=(
                                                                                                         1,))
    assert input_placeholder.shape[-1] == playground.OBSERVATION_SPACE.shape[0]
    assert output_placeholder.shape.rank == 2


# --- build_MLP_computation_graph ------------------------------------------------------------------------------------

def test_build_MLP_computation_graph_io(tf_setup, gym_discrete_setup):
    _, out_p, nn_shape = tf_setup
    exp_spec, playground = gym_discrete_setup
    keras_input = keras.Input(shape=(12,))

    mlp_hidden_ops = bloc.build_MLP_computation_graph(keras_input, playground.ACTION_CHOICES, nn_shape)
    print("\n\n>>> {}\n\n".format(mlp_hidden_ops))
    # model = keras.Model(inputs=keras_input, outputs=mlp_hidden_ops)
    # print(model.to_yaml())


def test_build_MLP_computation_graph_with_DISCRETE_adapter(gym_discrete_setup):
    _, playground = gym_discrete_setup
    input_placeholder, out_placeholder, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                                      action_shape_constraint=(
                                                                                                          1,))
    bloc.build_MLP_computation_graph(input_placeholder, playground.ACTION_CHOICES, hidden_layer_topology=(2, 2))


def test_build_MLP_computation_graph_with_CONTINUOUS_adapter(gym_continuous_setup):
    _, playground = gym_continuous_setup
    input_placeholder, out_placeholder, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                                      action_shape_constraint=(
                                                                                                          1,))
    bloc.build_MLP_computation_graph(input_placeholder, playground.ACTION_CHOICES, hidden_layer_topology=(2, 2))


def test_build_KERAS_MLP_computation_graph_io(tf_setup, gym_discrete_setup):
    _, out_p, nn_shape = tf_setup
    exp_spec, playground = gym_discrete_setup
    keras_input = keras.Input(shape=(12,))
    
    mlp_hidden_ops = bloc.build_KERAS_MLP_computation_graph(keras_input, playground.ACTION_CHOICES, nn_shape)
    print("\n\n>>> {}\n\n".format(mlp_hidden_ops))
    # model = keras.Model(inputs=keras_input, outputs=mlp_hidden_ops)
    # print(model.to_yaml())


def test_build_KERAS_MLP_computation_graph_with_DISCRETE_adapter(gym_discrete_setup):
    _, playground = gym_discrete_setup
    input_placeholder, out_placeholder, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                                      action_shape_constraint=(
                                                                                                          1,))
    bloc.build_KERAS_MLP_computation_graph(input_placeholder, playground.ACTION_CHOICES, hidden_layer_topology=(2, 2))


def test_build_KERAS_MLP_computation_graph_with_CONTINUOUS_adapter(gym_continuous_setup):
    _, playground = gym_continuous_setup
    input_placeholder, out_placeholder, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                                      action_shape_constraint=(
                                                                                                          1,))
    bloc.build_KERAS_MLP_computation_graph(input_placeholder, playground.ACTION_CHOICES, hidden_layer_topology=(2, 2))


def test_integration_Playground_to_adapter_to_build_graph(gym_continuous_setup):
    exp_spec, playground = gym_continuous_setup
    
    # (!) fake input data
    input_data = np.ones((20, *playground.OBSERVATION_SPACE.shape))
    
    input_placeholder, out_placeholder, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                                      action_shape_constraint=(
                                                                                                      1,))

    """Build a Multi Layer Perceptron (MLP) as the policy parameter theta using a computation graph"""
    theta = bloc.build_MLP_computation_graph(input_placeholder, playground.ACTION_CHOICES,
                                             exp_spec.theta_nn_h_layer_topo)

    writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    with tf_cv1.Session() as sess:
        # initialize random variable in the computation graph
        sess.run(tf_cv1.global_variables_initializer())

        # execute mlp computation graph with input data
        a = sess.run(theta, feed_dict={input_placeholder: input_data})

        # print("\n\n>>>run theta:\n{}\n\n".format(a))
    writer.close()



# --- TrajectoriesBatchContainer & epoch_buffer ---------------------------------------------------------------------------

def test_SamplingContainer_CONTINUOUS_BASIC(gym_continuous_setup):
    (exp_spec, playground) = gym_continuous_setup
    env = playground.env

    # todo: finish implementing test case

    """ STEP-1: Container instantiation"""
    the_TRAJECTORY_COLLECTOR = TrajectoryCollector(exp_spec, playground)
    the_UNI_BATCH_COLLECTOR = UniformBatchCollector(capacity=exp_spec.batch_size_in_ts)


    """--Simulator code: trajectorie -------------------------------------------------------------------------------"""
    while the_UNI_BATCH_COLLECTOR.is_not_full():
        observation = env.reset()

        step = 0
        """--Simulator code: time-step------------------------------------------------------------------------------"""
        while True:
            step += 1
            # env.render()  # (!) keep render() turn OFF during unit test

            print(observation)

            action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
            observation, reward, done, info = env.step(action)

            print("\ninfo: {}\n".format(info))

            """ STEP-2: append sample to container"""
            the_TRAJECTORY_COLLECTOR.collect_OAR(observation, action, reward)

            if done:
                """ STEP-3: acces container"""
                _ = the_TRAJECTORY_COLLECTOR.trajectory_ended()

                the_TRAJECTORY_COLLECTOR.compute_Qvalues_as_rewardToGo()
                trj_container = the_TRAJECTORY_COLLECTOR.pop_trajectory_and_reset()
                collected_timestep = len(trj_container)
                assert step == collected_timestep, "Trajectory lenght do not match nb collected_timestep"


                the_UNI_BATCH_COLLECTOR.collect(trj_container)
                np_array_obs, np_array_act, np_array_rew, Q_values, trajectory_return, trajectory_lenght = trj_container.unpack()


                print(
                    "\n\n----------------------------------------------------------------------------------------"
                    "\n Trajectorie finished after {} timesteps".format(step + 1))
                print("observation: {}".format(np_array_obs))
                print("reward: {}".format(np_array_rew))
                break



def test_SamplingContainer_DISCRETE_BASIC(gym_discrete_setup):
    (exp_spec, playground) = gym_discrete_setup
    env = playground.env

    # todo: finish implementing test case

    """ STEP-1: Container instantiation"""
    the_TRAJECTORY_COLLECTOR = TrajectoryCollector(exp_spec, playground)
    the_UNI_BATCH_COLLECTOR = UniformBatchCollector(capacity=exp_spec.batch_size_in_ts)

    """--Simulator code: trajectorie -------------------------------------------------------------------------------"""
    while the_UNI_BATCH_COLLECTOR.is_not_full():
        observation = env.reset()

        step = 0
        """--Simulator code: time-step------------------------------------------------------------------------------"""
        while True:
            step += 1
            # env.render()  # (!) keep render() turn OFF during unit test

            print(observation)

            action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
            observation, reward, done, info = env.step(action)

            print("\ninfo: {}\n".format(info))

            """ STEP-2: append sample to container"""
            the_TRAJECTORY_COLLECTOR.collect_OAR(observation, action, reward)

            if done:
                """ STEP-3: acces container"""
                _ = the_TRAJECTORY_COLLECTOR.trajectory_ended()

                the_TRAJECTORY_COLLECTOR.compute_Qvalues_as_rewardToGo()
                trj_container = the_TRAJECTORY_COLLECTOR.pop_trajectory_and_reset()
                collected_timestep = len(trj_container)
                assert step == collected_timestep, "Trajectory lenght do not match nb collected_timestep"

                the_UNI_BATCH_COLLECTOR.collect(trj_container)
                np_array_obs, np_array_act, np_array_rew, Q_values, trajectory_return, trajectory_lenght = trj_container.unpack()


                print(
                    "\n\n----------------------------------------------------------------------------------------"
                    "\n Trajectorie finished after {} timesteps".format(step + 1))
                print("observation: {}".format(np_array_obs))
                print("reward: {}".format(np_array_rew))
                break



# --- policy_theta ----------------------------------------------------------------------------------------------
def test_policy_theta_discrete_space_PARAM_FAIL(gym_and_tf_discrete_setup):

    _ , act_p, exp_spec, playground = gym_and_tf_discrete_setup
    obs_p_wrong_shape = tf_cv1.placeholder(tf.float32, shape=(None, 3))
    theta_mlp = bloc.build_MLP_computation_graph(obs_p_wrong_shape, playground.ACTION_CHOICES,
                                                 exp_spec.theta_nn_h_layer_topo)

    with pytest.raises(AssertionError):
        bloc.policy_theta_discrete_space(obs_p_wrong_shape, playground)

def test_policy_theta_discrete_space_ENV_NOT_DISCRETE(gym_and_tf_continuous_setup):
    obs_p, act_p, exp_spec, continuous_playground = gym_and_tf_continuous_setup

    out_p_wrong_shape = tf_cv1.placeholder(tf.float32, shape=(None, 43))
    theta_mlp = bloc.build_MLP_computation_graph(obs_p, continuous_playground.ACTION_CHOICES,
                                                 exp_spec.theta_nn_h_layer_topo)

    with pytest.raises(AssertionError):
        bloc.policy_theta_discrete_space(theta_mlp, continuous_playground)


def test_policy_theta_discrete_space_PASS(gym_and_tf_discrete_setup):

    obs_p, act_p, exp_spec, playground = gym_and_tf_discrete_setup
    theta_mlp = bloc.build_MLP_computation_graph(obs_p, playground.ACTION_CHOICES, exp_spec.theta_nn_h_layer_topo)
    bloc.policy_theta_discrete_space(theta_mlp, playground)
    # todo: implement test case

def test_policy_theta_continuous_space_PARAM_FAIL(gym_and_tf_continuous_setup):
    _, act_p, exp_spec, playground = gym_and_tf_continuous_setup

    obs_p_wrong_shape = tf_cv1.placeholder(tf.float32, shape=(None, 43))

    with pytest.raises(AssertionError):
        bloc.policy_theta_continuous_space(obs_p_wrong_shape, playground)

# def test_policy_theta_continuous_space_BUILDGRAPH_PASS(gym_and_tf_continuous_setup):
#     obs_p, act_p, exp_spec, playground = gym_and_tf_continuous_setup
#
#     theta_mlp = bloc.build_MLP_computation_graph(obs_p, playground, exp_spec.theta_nn_h_layer_topo)
#
#     bloc.policy_theta_continuous_space(theta_mlp, playground)
#     # todo: finish test case

def test_policy_theta_continuous_space_ENV_NOT_DISCRETE(gym_and_tf_discrete_setup):

    obs_p, act_p, exp_spec, discrete_playground = gym_and_tf_discrete_setup
    obs_p_wrong_shape = tf_cv1.placeholder(tf.float32, shape=(None, 43))
    theta_mlp = bloc.build_MLP_computation_graph(obs_p, discrete_playground.ACTION_CHOICES,
                                                 exp_spec.theta_nn_h_layer_topo)

    with pytest.raises(AssertionError):
        bloc.policy_theta_continuous_space(theta_mlp, discrete_playground)



# --- feed_dictionary --------------------------------------------------------------------------------------------
def test_build_feed_dictionary_PASS(gym_and_tf_discrete_setup):
    obs_p, act_p, exp_spec, playground = gym_and_tf_discrete_setup

    l_ph = [obs_p, act_p]
    l_ar = [np.array(obs_p._shape_as_list()), np.array(act_p._shape_as_list())]

    blocAndTools.tensorflowbloc.build_feed_dictionary(l_ph, l_ar)


def test_build_feed_dictionary_FAIL(gym_and_tf_discrete_setup):
    obs_p, act_p, exp_spec, playground = gym_and_tf_discrete_setup

    l_ph = [obs_p, act_p]
    l_ar = [np.array(obs_p._shape_as_list())]
    with pytest.raises(AssertionError):
        blocAndTools.tensorflowbloc.build_feed_dictionary(l_ph, l_ar)


# --- Return function --------------------------------------------------------------------------------------------

def test_reward_to_go_PASS(gym_and_tf_continuous_setup):
    N = 20
    rewards = [x for x in range(N)]
    reward_to_go = rtg.reward_to_go(rewards)

    print(reward_to_go)
    assert reward_to_go[N-1] == N-1, "shape:{} - {}".format(reward_to_go.shape, reward_to_go)
    assert isinstance(reward_to_go, list)


def test_reward_to_go_DUMMY_PASS(gym_and_tf_continuous_setup):
    N = 10
    rewards = [1 for x in range(N)]
    expected_reward_to_go = [x for x in range(N)]
    expected_reward_to_go.reverse()

    reward_to_go = rtg.reward_to_go(rewards)

    print(reward_to_go)
    assert np.equal(rewards, expected_reward_to_go).any()
    assert isinstance(reward_to_go, list)


def test_dicounted_reward_to_go_PASS(gym_and_tf_continuous_setup):
    _, _, exp_spec, _ = gym_and_tf_continuous_setup
    exp_spec.discout_factor = 0.98  # (!)the last assert depend on this value
    N = 20
    rewards = [x for x in range(N)]
    reward_to_go = rtg.reward_to_go(rewards)

    discounted_reward_to_go = rtg.discounted_reward_to_go(rewards, exp_spec)
    print("discout_factor: {}\n".format(exp_spec.discout_factor))
    print("\t{} reward_to_go".format(reward_to_go))
    print("\t{} discounted_reward_to_go".format(discounted_reward_to_go))

    assert isinstance(discounted_reward_to_go, list)
    assert discounted_reward_to_go[N-1] == N-1, "shape:{} - {}".format(discounted_reward_to_go.shape, discounted_reward_to_go)
    assert discounted_reward_to_go[0] == 136

def test_dicounted_reward_to_go_FAIL(gym_and_tf_continuous_setup):
    _, _, exp_spec, _ = gym_and_tf_continuous_setup
    N = 20
    rewards = [x for x in range(N)]

    exp_spec.discout_factor = 2
    with pytest.raises(AssertionError):
        rtg.discounted_reward_to_go(rewards, exp_spec)

    exp_spec.discout_factor = -0.1
    with pytest.raises(AssertionError):
        rtg.discounted_reward_to_go(rewards, exp_spec)


def test_reward_to_go_NP_PASS(gym_and_tf_continuous_setup):
    N = 20
    rewards = [x for x in range(N)]
    np_rewards = np.array(rewards)
    reward_to_go = rtg.reward_to_go_np(np_rewards)

    print(reward_to_go)
    assert reward_to_go[N-1] == N-1, "shape:{} - {}".format(reward_to_go.shape, reward_to_go)
    assert isinstance(reward_to_go, np.ndarray)


def test_dicounted_reward_to_go_np_PASS(gym_and_tf_continuous_setup):
    _, _, exp_spec, _ = gym_and_tf_continuous_setup
    exp_spec.discout_factor = 0.98  # (!)the last assert depend on this value
    N = 20
    rewards = [x for x in range(N)]
    np_rewards = np.array(rewards)
    reward_to_go = rtg.reward_to_go_np(np_rewards)

    discounted_reward_to_go = rtg.discounted_reward_to_go_np(np_rewards, exp_spec)
    print("discout_factor: {}\n".format(exp_spec.discout_factor))
    print("\t{} reward_to_go".format(reward_to_go))
    print("\t{} discounted_reward_to_go".format(discounted_reward_to_go))

    assert isinstance(discounted_reward_to_go, np.ndarray)
    assert discounted_reward_to_go[N-1] == N-1, "shape:{} - {}".format(discounted_reward_to_go.shape, discounted_reward_to_go)
    assert discounted_reward_to_go[0] == 136

def test_dicounted_reward_to_go_np_FAIL(gym_and_tf_continuous_setup):
    _, _, exp_spec, _ = gym_and_tf_continuous_setup
    N = 20
    rewards = [x for x in range(N)]
    np_rewards = np.array(rewards)

    exp_spec.discout_factor = 2
    with pytest.raises(AssertionError):
        rtg.discounted_reward_to_go_np(np_rewards, exp_spec)

    exp_spec.discout_factor = -0.1
    with pytest.raises(AssertionError):
        rtg.discounted_reward_to_go_np(np_rewards, exp_spec)

    exp_spec.discout_factor = -0.5
    ones_2D = np.ones((4, 2))
    with pytest.raises(AssertionError):
        rtg.discounted_reward_to_go_np(ones_2D, exp_spec)





# ---- tensor experiment -----------------------------------------------------------------------------------------

def test_create_tensor():
    ops_a = tf.add(3, 5)
    # print(">>> ops_a: {}".format(ops_a))

    with tf_cv1.Session() as sess:
        # HINT: could be --> output_stream = sys.stderr
        print_ops = tf.print("ops_a:", ops_a, output_stream=sys.stdout)

        run_result = sess.run([ops_a, print_ops])

        print(">>> run result: {}\n\n".format(run_result))


def test_vocab_PASS():
    print(vocab)

def test_CycleIndexer_PASS():
    cycle_indexer = visualisationtools.CycleIndexer(20)

    for _ in range(40):
        i, j = cycle_indexer.__next__()
        print(i, j)


