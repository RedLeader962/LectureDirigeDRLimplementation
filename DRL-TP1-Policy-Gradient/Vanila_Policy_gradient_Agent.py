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

from vocabulary import rl_name
vocab = rl_name()



"""
Start TensorBoard in terminal:
    tensorboard --logdir=DRL-TP1-Policy-Gradient/graph/

In browser, go to:
    http://0.0.0.0:6006/ 
"""

def vanila_policy_gradient_agent_discrete(render_env=False):

    exp_spec = bloc.ExperimentSpec(trajectories_batch_size=1)
    playground = bloc.GymPlayground(environment_name='LunarLander-v2')

    env = playground.env

    # /---- Build computation graph -----
    # Build a Multi Layer Perceptron (MLP) as the policy parameter theta using a computation graph
    # Build the Policy_theta computation graph

    observation_placeholder, action_placeholder = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    theta_mlp = bloc.build_MLP_computation_graph(observation_placeholder, action_placeholder.shape, exp_spec.nn_h_layer_topo)
    discrete_policy_theta = bloc.policy_theta_discrete_space(theta_mlp, action_placeholder.shape, playground)

    # /---- Container instantiation -----
    timestep_collector = bloc.TimestepCollector(exp_spec, playground)
    # todo --> TrajectoriesBatchContainer

    # # /---- Pseudo loss -----
    # # todo --> pseudo loss function
    # loss_op = None
    # raise NotImplementedError   # todo: implement loss_op
    #
    # # /---- Optimizer -----
    # optimizer_op = tf.train.AdamOptimizer(learning_rate=exp_spec.learning_rate).minimize(loss_op)


    # /---- Start computation graph -----
    writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    with tf_cv1.Session() as sess:
        sess.run(tf_cv1.global_variables_initializer())     # initialize random variable in the computation graph

        # /---- Simulator: Epoch -----
        for epoch in range(exp_spec.max_epoch):

            # /---- Simulator: trajectorie -----
            for trajectorie in range(exp_spec.trajectories_batch_size):
                print("/{}\n:: Trajectorie {} started\n\n".format("-" * 90, trajectorie + 1))
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
                    action = sess.run(discrete_policy_theta, feed_dict={observation_placeholder: batch_size_one_observation})

                    action = np.squeeze(action)
                    observation, reward, done, info = env.step(action)

                    print("E:{} T:{} TS:{} | action[{}] --> reward={}".format(epoch + 1, trajectorie + 1,
                                                                              step + 1, action, reward))

                    if len(info) is not 0:
                        print("\ninfo: {}\n".format(info))

                    # /---- Collect current timestep events -----
                    timestep_collector.append(observation, action, reward)


                    # /---- end trajectorie -----
                    if done or (step == exp_spec.timestep_max_per_trajectorie - 1):
                        trajectorie_container = timestep_collector.get_collected_trajectorie_and_reset()
                        observations, actions, rewards = trajectorie_container.unpack()

                        print("\n:: Trajectorie {} finished after {} timesteps".format(trajectorie + 1, step + 1))
                        print("\ntrajectorie_container size: {}".format(len(trajectorie_container)))
                        print("\nobservation: {}".format(observations))
                        print("\nAction: {}".format(actions))
                        print("\nreward: {}\n\n".format(rewards))
                        print("{}/\n\n".format("-" * 90))

                        trajectorie_container = timestep_collector.get_collected_trajectorie_and_reset()
                        break

                # /---- Collect trajectorie_container into trajectories_batch_container -----


            # /---- Compute gradient & update policy -----
            observations, actions, rewards = trajectorie_container.unpack()

            # todo -->
            # feed_dictionary = bloc.build_feed_dictionary([observation_placeholder, action_placeholder],
            #                                              [observations, actions])
            # sess.run([loss_op, optimizer_op], feed_dict=feed_dictionary)

    writer.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Command line arg for agent traning")
    parser.add_argument('--render_env', type=bool, default=False)
    args = parser.parse_args()

    vanila_policy_gradient_agent_discrete(render_env=args.render_env)


