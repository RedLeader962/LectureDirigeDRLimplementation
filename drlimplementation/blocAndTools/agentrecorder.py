# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...
from datetime import datetime

import tensorflow as tf
tf_cv1 = tf.compat.v1   # shortcut

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from blocAndTools import buildingbloc as bloc
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from DRLTP1PolicyGradient.REINFORCEbrain import REINFORCE_policy
from drlimplementation.blocAndTools.rl_vocabulary import rl_name
vocab = rl_name()
# endregion


# (Ice-Boxed) todo:assessment -->  functionality of this module could go in REINFORCEplayingloop.py
#                                                                since VideoRecorder can be enable/disable:
# todo:refactor --> exp_spec must follow the saved computation grah: find a way to assert compatibility between those
def record_REINFORCE_agent_discrete(env='CartPole-v0', nb_of_clip_recorded=5):
    """
    Record playing loop of a previously trained REINFORCE agent in the 'CartPole-v0' environment

    """

    exp_spec = ExperimentSpec()

    cartpole_param_dict_2 = {
        'prefered_environment': 'CartPole-v0',
        'paramameter_set_name': 'RedLeader CartPole-v0',
        'batch_size_in_ts': 5000,
        'max_epoch': 50,
        'discounted_reward_to_go': True,
        'discout_factor': 0.999,
        'learning_rate': 1e-2,
        'nn_h_layer_topo': (62,),
        'random_seed': 82,
        'hidden_layers_activation': tf.nn.tanh,  # tf.nn.relu,
        'output_layers_activation': None,
        'render_env_every_What_epoch': 100,
        'print_metric_every_what_epoch': 2,
    }
    exp_spec.set_experiment_spec(cartpole_param_dict_2)
    playground = GymPlayground(environment_name=exp_spec.prefered_environment)

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    # *                                                                                                               *
    # *                                          Build computation graph                                              *
    # *                                                                                                               *
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    """ ---- Build the Policy_theta computation graph with theta as multi-layer perceptron ---- """
    # Placeholder
    observation_ph, action_ph, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(
        playground, None, obs_shape_constraint=None)

    # The policy & is neural net theta
    reinforce_policy = REINFORCE_policy(observation_ph, action_ph, Q_values_ph, exp_spec, playground)
    (policy_action_sampler, theta_mlp, pseudo_loss) = reinforce_policy

    saver = tf_cv1.train.Saver()

    with tf_cv1.Session() as sess:

        """ ---- Restore a trained REINFORCE agent computation graph ---- """
        saver.restore(sess, '../graph/saved_training/REINFORCE_agent-39')

        """ ---- Execute recording loop ---- """
        for run in range(nb_of_clip_recorded):
            current_observation = playground.env.reset()    # <-- fetch initial observation

            date_now = datetime.now()
            timestamp = "{}{}".format(date_now.minute, date_now.microsecond)
            recorder = VideoRecorder(playground.env, '../../video/REINFORCE_agent_cartpole_{}--{}.mp4'.format(run+1, timestamp))
            print("\n:: Start recording trajectory {}\n".format(run+1))

            """ ---- Simulator: time-steps ---- """
            while True:

                """ ---- Record one timestep ---- """
                recorder.capture_frame()

                """ ---- Agent: act in the environment ---- """
                step_observation = bloc.format_single_step_observation(current_observation)
                action_array = sess.run(policy_action_sampler, feed_dict={observation_ph: step_observation})

                action = bloc.format_single_step_action(action_array)
                observe_reaction, reward, done, _ = playground.env.step(action)
                current_observation = observe_reaction  # <-- (!)

                if done:
                    recorder.close()
                    break

    playground.env.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Command line arg for agent playing")
    parser.add_argument('--env', type=str, default='CartPole-v0')
    args = parser.parse_args()

    record_REINFORCE_agent_discrete(env=args.env)


