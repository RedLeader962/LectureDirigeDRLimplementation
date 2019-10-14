# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

# region ::Import statement ...
import tensorflow as tf

from BasicPolicyGradient.REINFORCEbrain import REINFORCE_policy
from blocAndTools import buildingbloc as bloc
from blocAndTools.buildingbloc import ExperimentSpec, GymPlayground
from blocAndTools.rl_vocabulary import rl_name

vocab = rl_name()
tf_cv1 = tf.compat.v1   # shortcut
# endregion


def play_REINFORCE_agent_discrete(max_trajectories=20, test_run=False):
    """
    Execute playing loop of a previously trained REINFORCE agent in the 'CartPole-v0' environment

    :param max_trajectories:
    :type max_trajectories: int
    :param test_run:
    :type test_run: bool

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
        saver.restore(sess, 'BasicPolicyGradient/saved_training/REINFORCE_agent-39')
        print(":: Agent player >>> LOCK & LOAD\n"
              "           ↳ Execute {} run\n           ↳ Test run={}".format(max_trajectories, test_run)
              )

        print(":: Running trajectory >>> ", end=" ", flush=True)
        for run in range(max_trajectories):
            print(run+1, end=" ", flush=True)

            obs = playground.env.reset()    # <-- fetch initial observation
            # recorder = VideoRecorder(playground.env, '../video/cartpole_{}.mp4'.format(run))

            """ ---- Simulator: time-steps ---- """
            while True:

                if not test_run:     # keep environment rendering turned OFF during unit test
                    playground.env.render()
                    # recorder.capture_frame()

                """ ---- Agent: act in the environment ---- """
                step_observation = bloc.format_single_step_observation(obs)
                action_array = sess.run(policy_action_sampler, feed_dict={observation_ph: step_observation})

                action = bloc.format_single_step_action(action_array)
                obs_prime, reward, done, _ = playground.env.step(action)
                obs = obs_prime  # <-- (!)

                if done:
                    break

        print("END")

    # recorder.close()
    playground.env.close()
    print(":: Agent player >>> CLOSED")
