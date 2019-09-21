# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...

import tensorflow as tf


tf_cv1 = tf.compat.v1   # shortcut

import DRL_building_bloc as bloc

import tensorflow_weak_warning_supressor as no_cpu_compile_warn
no_cpu_compile_warn.execute()

from vocabulary import rl_name
vocab = rl_name()
# endregion



"""
Start TensorBoard in terminal:
    tensorboard --logdir=DRL-TP1-Policy-Gradient/graph/

In browser, go to:
    http://0.0.0.0:6006/ 
"""

def train_REINFORCE_agent_discrete(render_env=False, discounted_reward_to_go=True):

    exp_spec = bloc.ExperimentSpec(trajectories_batch_size=1)
    playground = bloc.GymPlayground(environment_name='LunarLander-v2')

    env = playground.env

    """ --- Build computation graph ---
        Build a Multi Layer Perceptron (MLP) as the policy parameter theta using a computation graph
        Build the Policy_theta computation graph
    """
    observation_p, action_p = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    Q_values_p = tf_cv1.placeholder(tf.float32, shape=(None,), name='q_values_placeholder')

    sampled_action_op, theta_mlp_op, pseudo_loss_op = bloc.REINFORCE_policy(observation_p, action_p, Q_values_p,
                                                                   playground, exp_spec)

    """ ---- Container instantiation ---- """
    timestep_collector = bloc.TimestepCollector(exp_spec, playground)

    """ ---- Optimizer ---- """
    policy_optimizer_op = bloc.policy_optimizer(pseudo_loss_op, exp_spec)


    """ ---- Warm-up the computation graph and start learning! ---- """
    writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    with tf_cv1.Session() as sess:
        sess.run(tf_cv1.global_variables_initializer())     # initialize random variable in the computation graph


        """ ---- Simulator: Epoch ---- """
        for epoch in range(exp_spec.max_epoch):

            """ ---- Simulator: trajectory ---- """
            for trajectory in range(exp_spec.trajectories_batch_size):
                print("/{}\n:: Trajectorie {} started\n\n".format("-" * 90, trajectory + 1))
                observation = env.reset()   # fetch initial observation

                """ ---- Simulator: time-step ---- """
                for step in range(exp_spec.timestep_max_per_trajectorie):

                    if render_env:
                        # keep environment rendering turned OFF during unit test
                        env.render()

                    """ ---- act in the environment ---- """
                    step_observation = bloc.format_single_step_observation(observation)

                    action_array = sess.run(sampled_action_op, feed_dict={observation_p: (
                        step_observation)})

                    action = bloc.format_single_step_action(action_array)
                    observation, reward, done, info = env.step(action)
                    print("E:{} T:{} TS:{} | action[{}] --> reward={}".format(epoch + 1, trajectory + 1,
                                                                              step + 1, action, reward))

                    if len(info) is not 0:
                        print("\ninfo: {}\n".format(info))

                    """ ---- Collect current timestep events ---- """
                    timestep_collector.append(observation, action, reward)

                    """ ---- end trajectory ---- """
                    if done or (step == exp_spec.timestep_max_per_trajectorie - 1):

                        trajectory_container = timestep_collector.get_collected_trajectory_and_reset_collector(
                            discounted_q_values=discounted_reward_to_go)

                        observations, actions, rewards, Q_values = trajectory_container.unpack()

                        # discounted_reward_to_go = bloc.discounted_reward_to_go(list(rewards), exp_spec)

                        print("\n:: Trajectory {} finished after {} timesteps".format(trajectory + 1, step + 1))
                        print("\ntrajectory_container size: {}".format(len(trajectory_container)))
                        print("\nobservation: {}".format(observations))
                        print("\nAction: {}".format(actions))
                        print("\nreward: {}\n\n".format(rewards))
                        print("\ndiscounted_reward_to_go: {}\n\n".format(discounted_reward_to_go))
                        print("{}/\n\n".format("-" * 90))

                        break

                """ ---- Collect trajectory_container into trajectories_batch_container ---- """


            """ ---- Compute gradient & update policy ---- """
            # observations, actions, rewards, Q_values = trajectory_container.unpack()

            # todo -->
            # feed_dictionary = bloc.build_feed_dictionary([observation_p, action_p],
            #                                              [observations, actions])
            # sess.run([loss_op, policy_optimizer_op], feed_dict=feed_dictionary)

    writer.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Command line arg for agent traning")
    parser.add_argument('--render_env', type=bool, default=False)
    parser.add_argument('--discounted_reward_to_go', type=bool, default=True)
    args = parser.parse_args()

    train_REINFORCE_agent_discrete(render_env=args.render_env)


