# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...
import tensorflow as tf
tf_cv1 = tf.compat.v1   # shortcut

from DRLimplementation.blocAndTools import buildingbloc as bloc
from DRLimplementation.blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from REINFORCEbrain import REINFORCE_policy
from DRLimplementation.blocAndTools.rl_vocabulary import rl_name
vocab = rl_name()
# endregion


def play_REINFORCE_agent_discrete(env='CartPole-v0'):
    """
    Execute playing loop of a previously trained REINFORCE agent in the 'CartPole-v0' environment

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
        saver.restore(sess, 'graph/saved_training/REINFORCE_agent-39')

        while True: #keep playing
        # for run in range(3):      #recorder version

            # recorder = VideoRecorder(playground.env, '../video/cartpole_{}.mp4'.format(run))
            current_observation = playground.env.reset()    # <-- fetch initial observation

            """ ---- Simulator: time-steps ---- """
            while True:

                playground.env.render()  # keep environment rendering turned OFF during unit test
                # recorder.capture_frame()

                """ ---- Agent: act in the environment ---- """
                step_observation = bloc.format_single_step_observation(current_observation)
                action_array = sess.run(policy_action_sampler, feed_dict={observation_ph: step_observation})

                action = bloc.format_single_step_action(action_array)
                observe_reaction, reward, done, _ = playground.env.step(action)
                current_observation = observe_reaction  # <-- (!)

                if done:
                    break


    # recorder.close()
    playground.env.close()



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Command line arg for agent playing")
    parser.add_argument('--env', type=str, default='CartPole-v0')
    args = parser.parse_args()

    play_REINFORCE_agent_discrete(env=args.env)


