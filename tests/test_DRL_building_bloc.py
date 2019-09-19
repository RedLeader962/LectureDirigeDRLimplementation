# coding=utf-8
import sys

import gym
import pytest
import DRL_building_bloc as bloc
import numpy as np

import tensorflow as tf
from tensorflow import keras
tf_cv1 = tf.compat.v1   # shortcut


# ---- setup & teardown -------------------------------------------------------------------------------------------

@pytest.fixture
def tf_setup():
    """
    :return: (in_p, out_p, nn_shape)
    :rtype: (tf.Tensor, tf.Tensor, list)
    """
    in_p = tf_cv1.placeholder(tf.float32, shape=(None, 8))
    out_p = tf_cv1.placeholder(tf.float32, shape=(None, 2))
    nn_shape = [2, 2]
    return in_p, out_p, nn_shape

@pytest.fixture
def gym_continuous_setup():
    """
    :return: (exp_spec, playground)
    :rtype: (ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(timestep_max_per_trajectorie=10, trajectories_batch_size=2, max_epoch=2, neural_net_hidden_layer_topology=[2, 2])
    playground = bloc.GymPlayground('LunarLanderContinuous-v2')
    return exp_spec, playground

@pytest.fixture
def gym_discrete_setup():
    """
    :return: (exp_spec, playground)
    :rtype: (ExperimentSpec, GymPlayground)
    """
    exp_spec = bloc.ExperimentSpec(timestep_max_per_trajectorie=10, trajectories_batch_size=2, max_epoch=2, neural_net_hidden_layer_topology=[2, 2])
    playground = bloc.GymPlayground('LunarLander-v2')
    return exp_spec, playground


# --- ExperimentSpec ---------------------------------------------------------------------------------------------

def test_ExperimentSpec_init_ENV_TOPOLOGY_FAIL():
    with pytest.raises(Exception):
        bloc.ExperimentSpec(neural_net_hidden_layer_topology=(1,))


# ---- playground ------------------------------------------------------------------------------------------------

def test_Playground_init_ENV_FAIL():
    with pytest.raises(Exception):
        bloc.GymPlayground('UnExistingEnvironment!!!')


def test_Playground_continuous():
    play = bloc.GymPlayground('LunarLanderContinuous-v2')
    assert play.ACTION_SPACE_SHAPE == (2,)
    assert play.OBSERVATION_SPACE_SHAPE == (8,)

def test_Playground_discreet():
    play = bloc.GymPlayground('LunarLander-v2')
    assert play.ACTION_SPACE_SHAPE == 4


# ---- gym_playground_to_tensorflow_graph_adapter --------------------------------------------------------------------

def test_gym_env_to_tf_graph_adapter_WRONG_IMPORT_TYPE():
    with pytest.raises(AssertionError):
        bloc.gym_playground_to_tensorflow_graph_adapter(gym)

def test_gym_env_to_tf_graph_adapter_DISCRETE_PASS(gym_discrete_setup):
    _, playground = gym_discrete_setup
    input_placeholder, output_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    assert input_placeholder.shape[-1] == playground.OBSERVATION_SPACE_SHAPE[0]
    print(output_placeholder.shape)
    assert output_placeholder.shape[-1] == playground.ACTION_SPACE_SHAPE

def test_gym_env_to_tf_graph_adapter_CONTINUOUS_PASS(gym_continuous_setup):
    _, playground = gym_continuous_setup
    input_placeholder, output_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    assert input_placeholder.shape[-1] == playground.OBSERVATION_SPACE_SHAPE[0]
    assert output_placeholder.shape[-1] == playground.ACTION_SPACE_SHAPE[0]


# --- build_MLP_computation_graph ------------------------------------------------------------------------------------

def test_build_MLP_computation_graph_io(tf_setup):
    _, out_p, nn_shape = tf_setup
    keras_input = keras.Input(shape=(12,))

    mlp_hidden_ops = bloc.build_MLP_computation_graph(keras_input, out_p, nn_shape)
    print("\n\n>>> {}\n\n".format(mlp_hidden_ops))
    model = keras.Model(inputs=keras_input, outputs=mlp_hidden_ops)
    # print(model.to_yaml())


def test_build_MLP_computation_graph_with_DISCRETE_adapter(gym_discrete_setup):
    _, playground = gym_discrete_setup
    input_placeholder, out_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    bloc.build_MLP_computation_graph(input_placeholder, out_placeholder, hidden_layer_topology=[2,2])

def test_build_MLP_computation_graph_with_CONTINUOUS_adapter(gym_continuous_setup):
    _, playground = gym_continuous_setup
    input_placeholder, out_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    bloc.build_MLP_computation_graph(input_placeholder, out_placeholder, hidden_layer_topology=[2,2])


def test_integration_Playground_to_adapter_to_build_graph(gym_continuous_setup):
    exp_spec, playground = gym_continuous_setup

    # (!) fake input data
    input_data = np.ones((exp_spec.trajectories_batch_size, *playground.OBSERVATION_SPACE_SHAPE))

    input_placeholder, out_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(playground)

    """Build a Multi Layer Perceptron (MLP) as the policy parameter theta using a computation graph"""
    theta = bloc.build_MLP_computation_graph(input_placeholder, out_placeholder, exp_spec.nn_h_layer_topo)

    writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    with tf_cv1.Session() as sess:
        # initialize random variable in the computation graph
        sess.run(tf_cv1.global_variables_initializer())

        # execute mlp computation graph with input data
        a = sess.run(theta, feed_dict={input_placeholder: input_data})

        # print("\n\n>>>run theta:\n{}\n\n".format(a))
    writer.close()



# --- TrajectoriesBuffer & epoch_buffer ---------------------------------------------------------------------------

def test_SamplingContainer_CONTINUOUS_BASIC(gym_continuous_setup):
    (exp_spec, playground) = gym_continuous_setup
    env = playground.env

    # todo: finish implementing test case

    """ STEP-1: Container instantiation"""
    timestep_collector = bloc.TimestepCollector(exp_spec, playground)


    """--Simulator code: trajectorie -------------------------------------------------------------------------------"""
    for traj in range(exp_spec.trajectories_batch_size):
        observation = env.reset()

        """--Simulator code: time-step------------------------------------------------------------------------------"""
        for step in range(exp_spec.timestep_max_per_trajectorie):
            # env.render()  # (!) keep render() turn OFF during unit test

            print(observation)

            action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
            observation, reward, done, info = env.step(action)

            print("\ninfo: {}\n".format(info))

            """ STEP-2: append sample to container"""
            timestep_collector.append(observation, action, reward)



            """ STEP-3: acces container"""
            if done or (step == exp_spec.timestep_max_per_trajectorie - 1):
                trajectorie_container = timestep_collector.get_collected_trajectorie_and_reset()
                np_array_obs, np_array_act, np_array_rew = trajectorie_container.unpack()

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
    timestep_collector = bloc.TimestepCollector(exp_spec, playground)

    """--Simulator code: trajectorie -------------------------------------------------------------------------------"""
    for traj in range(exp_spec.trajectories_batch_size):
        observation = env.reset()

        """--Simulator code: time-step------------------------------------------------------------------------------"""
        for step in range(exp_spec.timestep_max_per_trajectorie):
            # env.render()  # (!) keep render() turn OFF during unit test

            print(observation)

            action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
            observation, reward, done, info = env.step(action)

            print("\ninfo: {}\n".format(info))

            """ STEP-2: append sample to container"""
            timestep_collector.append(observation, action, reward)

            """ STEP-3: acces container"""
            if done or (step == exp_spec.timestep_max_per_trajectorie - 1):
                trajectorie_container = timestep_collector.get_collected_trajectorie_and_reset()
                np_array_obs, np_array_act, np_array_rew = trajectorie_container.unpack()

                print(
                    "\n\n----------------------------------------------------------------------------------------"
                    "\n Trajectorie finished after {} timesteps".format(step + 1))
                print("observation: {}".format(np_array_obs))
                print("reward: {}".format(np_array_rew))
                break


# def test_TrajectoriesBuffer_ICEBOX(gym_continuous_setup):
#     exp_spec, playground = gym_continuous_setup
#     bloc.TrajectoriesBuffer(exp_spec.timestep_max_per_trajectorie, playground)
#     # todo: implement function & test case


def test_epoch_buffer_PASS():
    bloc.epoch_buffer()
    # todo: implement test case


def test_SamplingContainer_NORMALIZE_CONTAINER_SIZE(gym_continuous_setup):
    (exp_spec, playground) = gym_continuous_setup
    env = playground.env

    """ STEP-1: Container instantiation"""
    timestep_collector = bloc.TimestepCollector(exp_spec, playground)
    raise NotImplementedError  # todo: test case


def test_sampling_and_storing_by_nparray_iteration_BENCHMARK(gym_continuous_setup):

    raise NotImplementedError  # todo: (!) setup benchmark test

    (exp_spec, playground) = gym_continuous_setup
    env = playground.env

    """--Simulator code: trajectorie--------------------------------------------------------------------------------"""
    for traj in range(exp_spec.trajectories_batch_size):
        observation = env.reset()

        """ STEP-1: Container instantiation"""
        # QuickFix:  *playground.OBSERVATION_SPACE_SHAPE to unpack shape --> wont work with Discrete space dimension
        observations = np.zeros((*playground.OBSERVATION_SPACE_SHAPE, exp_spec.timestep_max_per_trajectorie))
        actions = np.zeros((*playground.ACTION_SPACE_SHAPE, exp_spec.timestep_max_per_trajectorie))
        rewards = np.zeros(exp_spec.timestep_max_per_trajectorie)

        """--Simulator code: time-step------------------------------------------------------------------------------"""
        for step in range(exp_spec.timestep_max_per_trajectorie):
            # env.render()  # (!) keep render() turn OFF during unit test

            print(observation)

            action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
            observation, reward, done, info = env.step(action)

            print("\ninfo: {}\n".format(info))

            """ STEP-2: append sample to container"""
            actions[:, step] = action
            observations[:, step] = observation
            rewards[step] = reward

            """ STEP-3: acces container"""
            if done or (step == exp_spec.timestep_max_per_trajectorie - 1):
                print(
                    "\n\n----------------------------------------------------------------------------------------"
                    "\n Episode finished after {} timesteps".format(step + 1))
                print("observation: {}".format(observations))
                print("reward: {}".format(rewards))
                break

def test_SamplingContainer_BENCHMARK(gym_continuous_setup):

    raise NotImplementedError  # todo: (!) setup benchmark test

    (exp_spec, playground) = gym_continuous_setup
    env = playground.env


    """ STEP-1: Container instantiation"""
    timestep_collector = bloc.TimestepCollector(exp_spec, playground)

    """--Simulator code: trajectorie -------------------------------------------------------------------------------"""
    for traj in range(exp_spec.trajectories_batch_size):
        observation = env.reset()

        """--Simulator code: time-step------------------------------------------------------------------------------"""
        for step in range(exp_spec.timestep_max_per_trajectorie):
            # env.render()  # (!) keep render() turn OFF during unit test

            print(observation)

            action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
            observation, reward, done, info = env.step(action)

            print("\ninfo: {}\n".format(info))

            """ STEP-2: append sample to container"""
            timestep_collector.append(observation, action, reward)

            """ STEP-3: acces container"""
            if done or (step == exp_spec.timestep_max_per_trajectorie - 1):
                trajectorie_container = timestep_collector.get_collected_trajectorie_and_reset()
                np_array_obs, np_array_act, np_array_rew = trajectorie_container.unpack()

                print(
                    "\n\n----------------------------------------------------------------------------------------"
                    "\n Trajectorie finished after {} timesteps".format(step + 1))
                print("observation: {}".format(np_array_obs))
                print("reward: {}".format(np_array_rew))
                break


# --- build_feed_dictionary -------------------------------------------------------------------------------------

def test_build_feed_dictionary_PASS():
    bloc.build_feed_dictionary()
    raise NotImplementedError   # todo: implement test case

# --- policy_theta ----------------------------------------------------------------------------------------------
def test_policy_theta_discrete_space_PASS():
    bloc.policy_theta_discrete_space()
    # todo: implement test case

def test_policy_theta_continuous_space_PASS():
    bloc.policy_theta_continuous_space()
    # todo: implement test case


# ---- tensor experiment ------------------------------------------------------------------------------------------

def test_create_tensor():
    ops_a = tf.add(3, 5)
    # print(">>> ops_a: {}".format(ops_a))

    with tf_cv1.Session() as sess:
        # HINT: could be --> output_stream = sys.stderr
        print_ops = tf.print("ops_a:", ops_a, output_stream=sys.stdout)

        run_result = sess.run([ops_a, print_ops])

        print(">>> run result: {}\n\n".format(run_result))