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

# (Priority) todo:refactor --> remove the module when refactoring to class is DONE:
raise DeprecationWarning

def play_REINFORCE_agent_discrete(exp_spec: ExperimentSpec, max_trajectories=20):
    """
    Execute playing loop of a previously trained REINFORCE agent in the 'CartPole-v0' environment

    :param exp_spec: Experiment specification
    :type exp_spec: ExperimentSpec
    :param max_trajectories: The number of trajectories the agent will execute (default=20)
    :type max_trajectories: int
    """


    playground = GymPlayground(environment_name=exp_spec.prefered_environment)

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    # *                                                                                                               *
    # *                                          Build computation graph                                              *
    # *                                                                                                               *
    # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    """ ---- Build the Policy_theta computation graph with theta as multi-layer perceptron ---- """
    # Placeholder
    observation_ph, action_ph, Q_values_ph = bloc.gym_playground_to_tensorflow_graph_adapter(playground,
                                                                                             obs_shape_constraint=None,
                                                                                             action_shape_constraint=None)

    # The policy & is neural net theta
    reinforce_policy = REINFORCE_policy(observation_ph, action_ph, Q_values_ph, exp_spec, playground)
    (policy_action_sampler, _, _) = reinforce_policy

    saver = tf_cv1.train.Saver()

    with tf_cv1.Session() as sess:
        saver.restore(sess, 'BasicPolicyGradient/saved_training/REINFORCE_agent-39')
        print(":: Agent player >>> LOCK & LOAD\n"
              "           ↳ Execute {} run\n           ↳ Test run={}".format(max_trajectories, exp_spec.isTestRun)
              )

        print(":: Running trajectory >>> ", end=" ", flush=True)
        for run in range(max_trajectories):
            print(run+1, end=" ", flush=True)

            obs = playground.env.reset()    # <-- fetch initial observation
            # recorder = VideoRecorder(playground.env, '../video/cartpole_{}.mp4'.format(run))

            """ ---- Simulator: time-steps ---- """
            while True:

                if not exp_spec.isTestRun:     # keep environment rendering turned OFF during unit test
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
