# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals


# region ::Import statement ...
import tensorflow as tf
tf_cv1 = tf.compat.v1   # shortcut
import numpy as np
import matplotlib.pyplot as plt

import DRL_building_bloc as bloc
from DRL_building_bloc import ExperimentSpec, GymPlayground, REINFORCE_policy
from visualisation_tool import CycleIndexer, ConsolPrintLearningStats
from sample_container import TrajectoryContainer, TrajectoryCollector, BatchContainer, UniformBatchCollector

import tensorflow_weak_warning_supressor as no_cpu_compile_warn
no_cpu_compile_warn.execute()

from vocabulary import rl_name
vocab = rl_name()
# endregion

# (!) Environment rendering manual selection.
# RENDER_ENV = True
RENDER_ENV = False


"""
Start TensorBoard in terminal:
    tensorboard --logdir=DRL-TP1-Policy-Gradient/graph/

In browser, go to:
    http://0.0.0.0:6006/ 
"""

def train_REINFORCE_agent_discrete(render_env=None, discounted_reward_to_go=None, print_metric_every_what_epoch=5):

    cartpole_parma_dict = {
        'prefered_environment': 'CartPole-v1',
        'paramameter_set_name': 'CartPole-v1 - Training spec',
        'timestep_max_per_trajectorie': 500,           # check the max_episode_steps specification of your chosen env
        'batch_size_in_ts': 40,
        'max_epoch': 5000,
        'discounted_reward_to_go': True,
        'discout_factor': 0.999,
        'learning_rate': 1e-3,
        'nn_h_layer_topo': (62, ),
        'random_seed': 42,
        'hidden_layers_activation': tf.nn.tanh,
        'output_layers_activation': tf.nn.tanh,
        'render_env_every_What_epoch': 100,
        'print_metric_every_what_epoch': 10,
    }

    cartpole_parma_dict_2 = {
        'prefered_environment': 'CartPole-v1',
        'paramameter_set_name': 'CartPole-v1',
        'timestep_max_per_trajectorie': 500,           # check the max_episode_steps specification of your chosen env
        'batch_size_in_ts': 5000,
        'max_epoch': 50,
        'discounted_reward_to_go': False,
        'discout_factor': 0.999,
        'learning_rate': 1e-2,
        'nn_h_layer_topo': (62, ),
        'random_seed': 42,
        'hidden_layers_activation': tf.nn.tanh,        # tf.nn.relu,
        'output_layers_activation': None,
        # 'output_layers_activation': tf.nn.sigmoid,
        'render_env_every_What_epoch': 100,
        'print_metric_every_what_epoch': 2,
    }

    test_parma_dict = {
        'prefered_environment': 'CartPole-v0',
        'paramameter_set_name': 'Test spec',
        'timestep_max_per_trajectorie': 200,
        'batch_size_in_ts': 10,
        'max_epoch': 20,
        'discounted_reward_to_go': True,
        'discout_factor': 0.999,
        'learning_rate': 1e-2,
        'nn_h_layer_topo': (8, 8),
        'random_seed': 42,
        'hidden_layers_activation': tf.nn.tanh,
        'output_layers_activation': tf.nn.tanh,
        'render_env_every_What_epoch': 5,
        'print_metric_every_what_epoch': 5,
    }

    # Note: Gamma value is critical.
    #       Big difference between 0.9 and 0.999.
    #       Also you need to take into account the experiment average number of step per episode
    #
    #           Example with experiment average step of 100:
    #              0.9^100 = 0.000026 vs 0.99^100 = 0.366003 vs 0.999^100 = 0.904792

    """
    For OpenAi Gym registered environment, go to:

        * Bird eye view: https://gym.openai.com/envs
        * Specification: https://github.com/openai/gym/blob/master/gym/envs/__init__.py 

                eg:
                    register(
                        id='CartPole-v1',
                        entry_point='gym.envs.classic_control:CartPoleEnv',
                        max_episode_steps=500,
                        reward_threshold=475.0,
                    )
                    
        'MountainCar-v0', 'MountainCarContinuous-v0', 'CartPole-v1', 'Pendulum-v0', 'LunarLander-v2', 'LunarLanderContinuous-v2', ...

    """
    # (nice to have) todo:refactor --> automate timestep_max_per_trajectorie field default: fetch the value from the selected env
    exp_spec = ExperimentSpec(print_metric_every_what_epoch)

    # exp_spec.set_experiment_spec(test_parma_dict)
    exp_spec.set_experiment_spec(cartpole_parma_dict_2)
    # exp_spec.set_experiment_spec(cartpole_parma_dict)

    playground = GymPlayground(environment_name=exp_spec.prefered_environment)

    if discounted_reward_to_go is not None:
        exp_spec.set_experiment_spec(
            {
                'discounted_reward_to_go': discounted_reward_to_go
            }
        )

    if RENDER_ENV is not None:
        render_env = RENDER_ENV

    print("\n\n>>> Environment rendering: {}".format(render_env))



    consol_print_learning_stats = ConsolPrintLearningStats(exp_spec, exp_spec.print_metric_every_what_epoch)


    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    # *                                                                                                               *
    # *                                  Build computation graph & data collector                                     *
    # *                                                                                                               *
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


    """ ---- Build the Policy_theta computation graph with theta as multi-layer perceptron ---- """
    # Placeholder
    # (CRITICAL) todo:assessment --> uneven placeholder prevent the optimiser from minimizing the loss
    obs_shape_constraint = tuple([exp_spec.max_epoch * exp_spec.timestep_max_per_trajectorie])  # <-- (!)
    # act_pl_shape_constraint = (1, )
    # observation_ph, action_ph, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
    #     playground, act_pl_shape_constraint, obs_shape_constraint=obs_shape_constraint)
    observation_ph, action_ph, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
        playground, None, obs_shape_constraint=None)



    # The policy & is neural net theta
    reinforce_policy = REINFORCE_policy(observation_ph, action_ph, Q_values_ph, exp_spec, playground)
    (policy_action_sampler, theta_mlp, pseudo_loss) = reinforce_policy
    # Todo --> Refactor Q_values_ph: push inside gym_playground_to_tensorflow_graph_adapter


    """ ---- Collector instantiation ---- """
    the_TRAJECTORY_COLLECTOR = TrajectoryCollector(exp_spec, playground)
    the_UNI_BATCH_COLLECTOR = UniformBatchCollector(exp_spec)


    """ ---- Optimizer ---- """
    policy_optimizer_op = bloc.policy_optimizer(pseudo_loss, exp_spec.learning_rate)


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

        consol_print_learning_stats.start_the_crazy_experiment()

        """ ---- Simulator: Epochs ---- """
        for epoch in range(exp_spec.max_epoch):

            consol_print_learning_stats.next_glorious_epoch()

            """ ---- Simulator: trajectories ---- """
            while not the_UNI_BATCH_COLLECTOR.full():      # (Priority) todo:fixme!! --> must be handle by the batch_collector:
                observation = playground.env.reset()   # fetch initial observation

                consol_print_learning_stats.next_glorious_trajectory()

                """ ---- Simulator: time-steps ---- """
                for step in range(exp_spec.timestep_max_per_trajectorie):

                    if (render_env and (epoch % exp_spec.render_env_every_What_epoch == 0)
                            and the_UNI_BATCH_COLLECTOR.trj_collected_so_far() == 0):
                        playground.env.render()  # (!) keep environment rendering turned OFF during unit test

                    """ ---- Agent: act in the environment ---- """
                    step_observation = bloc.format_single_step_observation(observation)
                    action_array = sess.run(policy_action_sampler, feed_dict={observation_ph: step_observation})

                    action = bloc.format_single_step_action(action_array)
                    observation, reward, done, info = playground.env.step(action)

                    """ ---- Agent: Collect current timestep events ---- """
                    the_TRAJECTORY_COLLECTOR.collect(observation, action, reward)

                    # if len(info) is not 0:
                    #     print("\ninfo: {}\n".format(info))

                    # Timestep consol print
                    # print("\t\t E:{} Tr:{} TS:{}\t\t|\taction[{}]\t--> \treward = {}".format(
                    #     epoch + 1, trj + 1, step + 1, action, reward))

                    """ ---- Simulator: trajectory as ended ---- """
                    if done:
                        trj_return = the_TRAJECTORY_COLLECTOR.trajectory_ended()

                        consol_print_learning_stats.trajectory_training_stat(
                            the_trajectory_return=trj_return, timestep=step)

                        """ ---- Agent: Collect the sampled trajectory  ---- """
                        trj_container = the_TRAJECTORY_COLLECTOR.pop_trajectory_and_reset()
                        the_UNI_BATCH_COLLECTOR.collect(trj_container)
                        break





            """ ---- Simulator: epoch as ended, it's time to learn! ---- """
            number_of_trj_collected = the_UNI_BATCH_COLLECTOR.trj_collected_so_far()
            total_timestep_collected = the_UNI_BATCH_COLLECTOR.timestep_collected_so_far()


            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *
            # *                                                                                                      *
            # *                                      Update policy_theta                                             *
            # *                                                                                                      *
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ** * * * *

            """ ---- Prepare data for backpropagation in the neural net ---- """
            batch_container = the_UNI_BATCH_COLLECTOR.pop_batch_and_reset_collector()

            # (Priority) todo:fixme!! --> must be compute by the batch_collector:
            epoch_average_trjs_return, epoch_average_trjs_lenght = batch_container.compute_metric()

            observations = batch_container.trjs_observations
            actions = batch_container.trjs_actions
            Q_values = batch_container.trjs_Qvalues

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



            consol_print_learning_stats.epoch_training_stat(
                epoch_loss=epoch_loss, epoch_average_trjs_return=epoch_average_trjs_return,
                epoch_average_trjs_lenght=epoch_average_trjs_lenght,
                number_of_trj_collected=number_of_trj_collected,
                total_timestep_collected=total_timestep_collected
            )



    consol_print_learning_stats.print_experiment_stats()
    writer.close()
    tf_cv1.reset_default_graph()
    sess.close()
    playground.env.close()

    plt.close()




if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Command line arg for agent traning")
    parser.add_argument('--render_env', type=bool, default=False)
    parser.add_argument('--discounted_reward_to_go', type=bool, default=None)
    args = parser.parse_args()

    train_REINFORCE_agent_discrete(render_env=args.render_env, discounted_reward_to_go=args.discounted_reward_to_go)


