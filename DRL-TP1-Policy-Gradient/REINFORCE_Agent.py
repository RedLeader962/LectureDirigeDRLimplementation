# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...

import tensorflow as tf
tf_cv1 = tf.compat.v1   # shortcut
import numpy as np
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

    exp_spec.max_epoch = 10
    exp_spec.timestep_max_per_trajectorie = 200
    exp_spec.trajectories_batch_size=1

    env = playground.env

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    # *                                                                                                               *
    # *                                  Build computation graph & data collector                                     *
    # *                                                                                                               *
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


    """ ---- Build the Policy_theta computation graph with theta as multi-layer perceptron ---- """
    observation_ph, action_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground)
    Q_values_ph = tf_cv1.placeholder(tf.float32, shape=(None,), name=vocab.Qvalues_placeholder)
    reinforce_policy = bloc.REINFORCE_policy(observation_ph, action_ph, Q_values_ph, exp_spec, playground)
    (sampled_action, theta_mlp, pseudo_loss) = reinforce_policy
    # Todo --> Refactor Q_values_ph: push inside gym_playground_to_tensorflow_graph_adapter


    """ ---- Collector instantiation ---- """
    timestep_collector = bloc.TimestepCollector(exp_spec, playground)
    trajectories_collector = bloc.TrajectoriesCollector(exp_spec)


    """ ---- Optimizer ---- """
    policy_optimizer_op = bloc.policy_optimizer(pseudo_loss, exp_spec)


    """ ---- Warm-up the computation graph and start learning! ---- """
    writer = tf_cv1.summary.FileWriter('./graph', tf_cv1.get_default_graph())
    tf_cv1.set_random_seed(exp_spec.random_seed)
    np.random.seed(exp_spec.random_seed)
    with tf_cv1.Session() as sess:
        sess.run(tf_cv1.global_variables_initializer())     # initialize random variable in the computation graph

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # *                                                                                                           *
        # *                                               Training loop                                               *
        # *                                                                                                           *
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        """ ---- Simulator: Epochs ---- """
        for epoch in range(exp_spec.max_epoch):

            """ ---- Simulator: trajectories ---- """
            for trj in range(exp_spec.trajectories_batch_size):
                print("\n:: Epoch: {} {}\n\t ↳:: Trajectorie {} started\n".format( epoch+1, "-" * 67, trj + 1))
                observation = env.reset()   # fetch initial observation

                """ ---- Simulator: time-steps ---- """
                for step in range(exp_spec.timestep_max_per_trajectorie):

                    if render_env:
                        env.render()    # (!) keep environment rendering turned OFF during unit test

                    """ ---- Agent: act in the environment ---- """
                    step_observation = bloc.format_single_step_observation(observation)
                    action_array = sess.run(sampled_action, feed_dict={observation_ph: step_observation})

                    action = bloc.format_single_step_action(action_array)
                    observation, reward, done, info = env.step(action)

                    """ ---- Agent: Collect current timestep events ---- """
                    timestep_collector.collect(observation, action, reward)

                    if len(info) is not 0:
                        print("\ninfo: {}\n".format(info))
                    print("\t\t E:{} Tr:{} TS:{}\t|\taction[{}]\t--> \treward = {}".format(epoch + 1, trj + 1,
                                                                              step + 1, action, reward))

                    """ ---- Simulator: trajectory as ended ---- """
                    if done or (step == exp_spec.timestep_max_per_trajectorie - 1):

                        """ ---- Agent: Collect the sampled trajectory  ---- """
                        trajectory_container = timestep_collector.get_collected_timestep_and_reset_collector(
                                                                        discounted_q_values=discounted_reward_to_go)

                        trajectories_collector.collect(trajectory_container)

                        # print(trajectory_container)
                        print("\n\n\t ↳::Trajectory {} finished after {} timesteps\n".format(trj + 1, step + 1))
                        print("{} E:{} Tr:{} END ::\n\n".format("-" * 63, epoch + 1, trj + 1))
                        break


            """ ---- Simulator: epoch as ended ---- 
            
            trajectories_dict keys:   
                
                    'trjs_obss', 'trjs_acts', 
                    'trjs_Qvalues', 'trjs_returns', 
                    'trjs_len', 'epoch_average_return', 
                    'epoch_average_lenghts'
            """
            trajectories_dict = trajectories_collector.get_collected_trajectories_and_reset_collector()

            ax = 0
            observations = np.squeeze(trajectories_dict['trjs_obss'], axis=ax)
            actions = np.squeeze(trajectories_dict['trjs_acts'], axis=ax)
            Q_values = np.squeeze(trajectories_dict['trjs_Qvalues'], axis=ax)

            assert observation_ph.shape.is_compatible_with(observations)
            assert action_ph.shape.is_compatible_with(actions)
            assert Q_values_ph.shape.is_compatible_with(Q_values)

            """ ---- Agent: Compute gradient & update policy ---- """
            feed_dictionary = bloc.build_feed_dictionary([observation_ph, action_ph, Q_values_ph],
                                                         [observations, actions, Q_values])

            trajectory_loss, _ = sess.run([pseudo_loss, policy_optimizer_op], feed_dict=feed_dictionary)



    writer.close()




if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Command line arg for agent traning")
    parser.add_argument('--render_env', type=bool, default=False)
    parser.add_argument('--discounted_reward_to_go', type=bool, default=True)
    args = parser.parse_args()

    train_REINFORCE_agent_discrete(render_env=args.render_env, discounted_reward_to_go=args.discounted_reward_to_go)


