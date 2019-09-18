# !/usr/bin/env python
import gym
import pretty_printing
import numpy as np
import tensorflow as tf

from DRL_building_bloc import GymPlayground, ExperimentSpec
playground = GymPlayground()
exp_spec = ExperimentSpec()


env = gym.make('LunarLanderContinuous-v2')

action_space_doc = "Action is two floats [main engine, left-right engines].\n" \
                   "\tMain engine: -1..0 off, 0..+1 throttle from 50% to 100% power.\n" \
                   "\t\t\t\t(!) Engine can't work with less than 50% power.\n" \
                   "\tLeft-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off"

info_str = pretty_printing.environnement_doc_str(env, action_space_doc)
print(info_str)

"""
LunarLanderContinuous-v2

Environment doc/info:

    env: <TimeLimit<LunarLanderContinuous<LunarLanderContinuous-v2>>>

    Metadata: {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    REWARD range: (-inf, inf)

    ACTION SPACE:
        Type: Box(2,)
            Higher bound: [1. 1.]
            Lower bound: [-1. -1.]

    Action is two floats [main engine, left-right engines].
        Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power.
                    (!) Engine can't work with less than 50% power.
        Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

    OBSERVATION SPACE:
        Type: Box(8,)
            Higher bound: [inf inf inf inf inf inf inf inf]
            Lower bound: [-inf -inf -inf -inf -inf -inf -inf -inf]
"""

MAX_TIMESTEP = 400
episode_batch_size = 1
action_space_dimension = len(env.action_space.high)
observation_space_dimension = len(env.observation_space.high)

n_hidden = 2
n_outputs = action_space_dimension

# def policy(obs, output_dim):
#     obs
#     return None



# X = tf.placeholder(tf.float32, shape=[None, observation_space_dimension])
# initializer = tf.contrib.layers.variance_scaling_initializer()
# hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
# logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
# outputs = tf.nn.sigmoid(logits)



"""--Simulator code: episode------------------------------------------------------------------------------------"""
for i_episode in range(2):
    observation = env.reset()
    """------------------------------------------------------------------------Simulator code: episode---(end)--"""

    observations = np.zeros((playground.OBSERVATION_SPACE_SHAPE, exp_spec.timestep_max_per_trajectorie))
    actions = np.zeros((playground.ACTION_SPACE_SHAPE, exp_spec.timestep_max_per_trajectorie))
    rewards = np.zeros(exp_spec.timestep_max_per_trajectorie)

    """--Simulator code: time-step------------------------------------------------------------------------------"""
    for t_timeStep in range(exp_spec.timestep_max_per_trajectorie):
        # env.render()  # (!) keep render() turn OFF during unit test

        print(observation)

        action = env.action_space.sample()  # sample a random action from the action space (aka: a random agent)
        observation, reward, done, info = env.step(action)
        """------------------------------------------------------------------Simulator code: time-step---(end)--"""

        print("\ninfo: {}\n".format(info))

        actions[:, t_timeStep] = action
        observations[:, t_timeStep] = observation
        rewards[t_timeStep] = reward

        if done:
            """---End of episode-------------------------------------------------------------------------------"""
            print(
                "\n\n----------------------------------------------------------------------------------------"
                "\n Episode finished after {} timesteps".format(t_timeStep + 1))
            print("reward: {}".format(rewards))
            print("observation: {}".format(observations))
            break


        # negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=outputs)
        # episode_reward = rewards.sum() #q-values
        # weighted_negative_likelihood = tf.multiply(negative_likelihoods, rewards)
        # loss = tf.reduce_mean(weighted_negative_likelihood)
        # gradients = tf.gradients(loss, observations)
        #
        # gradients()