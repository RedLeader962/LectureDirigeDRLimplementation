# coding=utf-8
# from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import gym
import numpy as np

import tensorflow as tf
from tensorflow import keras
tf_cv1 = tf.compat.v1   # shortcut

import DRL_building_bloc as bloc
from vocabulary import rl_name as vocab
import pretty_printing
import tensorflow_weak_warning_supressor as no_cpu_compile_warn
no_cpu_compile_warn.execute()

"""
Start TensorBoard in terminal:
    tensorboard --logdir=DRL-TP1-Policy-Gradient/graph/

In browser, go to:
    http://0.0.0.0:6006/ 
"""

def VanilaPolicyGradientAgentSingleTrajectorieBatch(render_env=False):

    exp_spec = bloc.ExperimentSpec(trajectories_batch_size=1)
    playground = bloc.GymPlayground(environment_name='LunarLander-v2')

    env = playground.env

    # /---- Build computation graph -----
    # Build a Multi Layer Perceptron (MLP) as the policy parameter theta using a computation graph

    obs_ph, act_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    # theta_mlp = bloc.build_MLP_computation_graph(obs_ph, act_ph, exp_spec.nn_h_layer_topo)
    discrete_policy_theta = bloc.policy_theta_discrete_space(obs_ph, act_ph, exp_spec)



    # /---- Container instantiation -----
    timestep_collector = bloc.TimestepCollector(exp_spec, playground)

    # /---- Start computation graph -----
    writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    with tf_cv1.Session() as sess:
        sess.run(tf_cv1.global_variables_initializer())     # initialize random variable in the computation graph

        # /---- Simulator: Epoch -----
        for epoch in range(exp_spec.max_epoch):

            # /---- Simulator: trajectorie -----
            for trajectorie in range(exp_spec.trajectories_batch_size):
                observation = env.reset()   # fetch initial observation

                # /---- Simulator: time-step -----
                for step in range(exp_spec.timestep_max_per_trajectorie):

                    if render_env:
                        # keep environment rendering turned OFF during unit test
                        env.render()

                    # /---- act in the environment -----
                    # note: Single trajectorie batch size hack for computation graph observation placeholder
                    #   |       observation.shape = (8,)
                    #   |   vs
                    #   |       np.expand_dims(observation, axis=0).shape = (1, 8)
                    batch_size_one_observation = np.expand_dims(observation, axis=0)
                    action = sess.run(discrete_policy_theta, feed_dict={obs_ph: batch_size_one_observation})

                    print(action)
                    observation, reward, done, info = env.step(action)

                    print(observation)
                    print("\ninfo: {}\n".format(info))

                    # /---- Collect current timestep events -----
                    timestep_collector.append(observation, action, reward)


                    # /---- end trajectorie -----
                    if done or (step == exp_spec.timestep_max_per_trajectorie - 1):
                        trajectorie_container = timestep_collector.get_collected_trajectorie_and_reset()
                        np_array_obs, np_array_act, np_array_rew = trajectorie_container.unpack()

                        print(
                            "\n\n----------------------------------------------------------------------------------------"
                            "\n Trajectorie finished after {} timesteps".format(step + 1))
                        print("observation: {}".format(np_array_obs))
                        print("reward: {}".format(np_array_rew))

                        trajectorie_container = timestep_collector.get_collected_trajectorie_and_reset()
                        break

    writer.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Command line arg for agent traning")
    parser.add_argument('--render_env', type=bool, default=False)
    args = parser.parse_args()

    VanilaPolicyGradientAgentSingleTrajectorieBatch(render_env=args.render_env)


