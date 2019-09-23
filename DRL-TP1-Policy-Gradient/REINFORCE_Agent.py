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


RENDER_ENV = True   # (!) Environment rendering manual selection. Comment this line


"""
Start TensorBoard in terminal:
    tensorboard --logdir=DRL-TP1-Policy-Gradient/graph/

In browser, go to:
    http://0.0.0.0:6006/ 
"""

def train_REINFORCE_agent_discrete(render_env=None, discounted_reward_to_go=None):

    exp_spec = bloc.ExperimentSpec()
    parma_dict = {
        'paramameter_set_name': 'Training spec',
        'timestep_max_per_trajectorie': 600,
        'trajectories_batch_size': 40,
        'max_epoch': 1000,
        'discounted_reward_to_go': True,
        'discout_factor': 0.999,
        'learning_rate': 1e-2,
        'nn_h_layer_topo': (8, 32, 32),
        'random_seed': 42,
        'hidden_layers_activation': tf.tanh,
        'output_layers_activation': tf.tanh,
        'render_env_every_What_epoch': 100
    }

    test_parma_dict = {
        'paramameter_set_name': 'Test spec',
        'timestep_max_per_trajectorie': 20,
        'trajectories_batch_size': 3,
        'max_epoch': 20,
        'discounted_reward_to_go': True,
        'discout_factor': 0.999,
        'learning_rate': 1e-2,
        'nn_h_layer_topo': (8, 8),
        'random_seed': 42,
        'hidden_layers_activation': tf.tanh,
        'output_layers_activation': tf.tanh,
        'render_env_every_What_epoch': 5
    }

    exp_spec.set_experiment_spec(parma_dict)
    # exp_spec.set_experiment_spec(test_parma_dict)

    if discounted_reward_to_go is not None:
        exp_spec.set_experiment_spec(
            {
                'discounted_reward_to_go': discounted_reward_to_go
            }
        )

    if RENDER_ENV is not None:
        render_env = RENDER_ENV

    print("\n\n>>> Environment rendering: {}".format(render_env))

    playground = bloc.GymPlayground(environment_name='LunarLander-v2')


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
    trajectories_collector = bloc.TrajectoriesCollector()


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
            print("\n\n:: Epoch: {:^3} {}".format( epoch+1, "-" * 75))

            """ ---- Simulator: trajectories ---- """
            for trj in range(exp_spec.trajectories_batch_size):
                observation = playground.env.reset()   # fetch initial observation


                """ ---- Simulator: time-steps ---- """
                for step in range(exp_spec.timestep_max_per_trajectorie):

                    if render_env and (epoch % exp_spec.render_env_every_What_epoch == 0) and trj==0:
                        playground.env.render()    # (!) keep environment rendering turned OFF during unit test

                    """ ---- Agent: act in the environment ---- """
                    step_observation = bloc.format_single_step_observation(observation)
                    action_array = sess.run(sampled_action, feed_dict={observation_ph: step_observation})

                    action = bloc.format_single_step_action(action_array)
                    observation, reward, done, info = playground.env.step(action)

                    """ ---- Agent: Collect current timestep events ---- """
                    timestep_collector.collect(observation, action, reward)

                    if len(info) is not 0:
                        print("\ninfo: {}\n".format(info))
                    # print("\t\t E:{} Tr:{} TS:{}\t\t|\taction[{}]\t--> \treward = {}".format(
                    #     epoch + 1, trj + 1, step + 1, action, reward))

                    """ ---- Simulator: trajectory as ended ---- """
                    if done or (step == exp_spec.timestep_max_per_trajectorie - 1):

                        """ ---- Agent: Collect the sampled trajectory  ---- """
                        trajectory_container = timestep_collector.get_collected_timestep_and_reset_collector(
                                                                        discounted_q_values=exp_spec.discounted_reward_to_go)

                        trajectories_collector.collect(trajectory_container)

                        # print(trajectory_container)
                        print("\t  ↳ ::Trajectory {:>4}     --->     got reward {:>8.2f}   after  {:>4}  timesteps".format(
                            trj + 1, trajectory_container.the_trajectory_return, step + 1))
                        break

            """ ---- Simulator: epoch as ended, it's time to learn! ---- """
            number_of_trj_collected = trajectories_collector.get_number_of_trajectories_collected()
            total_timestep_collected = trajectories_collector.get_total_timestep_collected()

            print("\n:: Collected {:>3} trajectories for a total of {:>5} timestep.".format(number_of_trj_collected, total_timestep_collected))

            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *
            # *                                                                                                      *
            # *                                      Update policy_theta                                             *
            # *                                                                                                      *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *

            """ ---- Prepare data for backpropagation in the neural net ---- """
            epoch_container = trajectories_collector.get_collected_trajectories_and_reset_collector()

            observations = epoch_container.trjs_observations
            actions = epoch_container.trjs_actions
            Q_values = epoch_container.trjs_Qvalues

            """ ---- Tensor/ndarray shape compatibility assessment ---- """
            assert observation_ph.shape.is_compatible_with(np.array(observations).shape), \
                "Obs: {} != {}".format(observation_ph.shape, np.array(observations).shape)
            assert action_ph.shape.is_compatible_with(np.array(actions).shape), \
                "Act: {} != {}".format(action_ph.shape, np.array(actions).shape)
            assert Q_values_ph.shape.is_compatible_with(np.array(Q_values).shape), \
                "Qval: {} != {}".format(Q_values_ph.shape, np.array(Q_values).shape)

            """ ---- Agent: Compute gradient & update policy ---- """
            feed_dictionary = bloc.build_feed_dictionary([observation_ph, action_ph, Q_values_ph],
                                                    [observations, actions, Q_values])
            epoch_loss, _ = sess.run([pseudo_loss, policy_optimizer_op],
                                     feed_dict=feed_dictionary)


            epoch_average_trjs_return, epoch_average_trjs_lenght = epoch_container.compute_metric()

            print("\n:: Epoch {:>3} metric:\n\t  ↳ | pseudo loss: {:>6.2f} "
                  "| average trj return: {:>6.2f} | average trj lenght: {:>6.2f}\n".format(
                epoch, epoch_loss, epoch_average_trjs_return, epoch_average_trjs_lenght))

            print("{} EPOCH:{:>3} END ::\n\n".format("-" * 72, epoch + 1, trj + 1))




    writer.close()
    playground.env.close()




if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Command line arg for agent traning")
    parser.add_argument('--render_env', type=bool, default=False)
    parser.add_argument('--discounted_reward_to_go', type=bool, default=None)
    args = parser.parse_args()

    train_REINFORCE_agent_discrete(render_env=args.render_env, discounted_reward_to_go=args.discounted_reward_to_go)


