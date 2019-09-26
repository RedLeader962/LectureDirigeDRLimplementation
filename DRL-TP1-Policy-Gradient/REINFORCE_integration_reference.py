from datetime import datetime

import tensorflow as tf
tf_cv1 = tf.compat.v1   # shortcut
import numpy as np
import gym
from gym.spaces import Discrete, Box

import DRL_building_bloc as BLOC                                                       # \\\\\\ My bloc \\\\\\


"""
Based on REINFOCE with reward to go simplest implementation from SpinniUp 

Refactor to serve as a integration test reference
"""


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    # env = gym.make(env_name)                                                        # ////// Original bloc //////
    playground = BLOC.GymPlayground(env_name)                                       # \\\\\\    My bloc    \\\\\\
    env = playground.env                                                            # \\\\\\    My bloc    \\\\\\


    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    # obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)                  # ////// Original bloc //////
    obs_ph, act_ph = BLOC.gym_playground_to_tensorflow_graph_adapter(playground)
    # logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])                               # ////// Original bloc //////
    logits = BLOC.build_MLP_computation_graph(
        obs_ph, playground, hidden_layer_topology=tuple(hidden_sizes))                 # \\\\\\    My bloc    \\\\\\


    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    # act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)                        # ////// Original bloc //////
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(weights_ph * log_probs)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # \\\\\\    My bloc    \\\\\\
    date_now = datetime.now()
    run_str = "Run--{}h{}--{}-{}-{}".format(date_now.hour, date_now.minute, date_now.day, date_now.month, date_now.year)
    writer = tf_cv1.summary.FileWriter("./graph/integration_test/{}".format(run_str), tf_cv1.get_default_graph())

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    weights_ph: np.array(batch_weights)
                                 })
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        mean_return = np.mean(batch_rets)
        average_len = np.mean(batch_lens)

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, mean_return, average_len))

        yield (i, batch_loss, mean_return, average_len)


    print("\n>>> Close session\n")
    writer.close()
    playground.env.close()
    sess.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')

    epoch_generator = train(env_name=args.env_name, lr=args.lr, render=args.render)
    for epoch in epoch_generator:
        (i, batch_loss, mean_return, average_len) = epoch

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, mean_return, average_len))
