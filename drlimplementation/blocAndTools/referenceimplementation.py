from datetime import datetime

import tensorflow as tf

from DRLTP1PolicyGradient import REINFORCEbrain

tf_cv1 = tf.compat.v1   # shortcut
import numpy as np
from gym.spaces import Discrete, Box

from blocAndTools import buildingbloc as BLOC
from blocAndTools.visualisationtools import ConsolPrintLearningStats                                 # \\\\\\    My bloc    \\\\\\
from blocAndTools.samplecontainer import TrajectoryCollector, UniformBatchCollector                  # \\\\\\    My bloc    \\\\\\


"""
Integration test based on 'REINFORCE with reward to go' simplest implementation from SpinniUp at:
    https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/2_rtg_pg.py

Use in conjunction with: ../tests/test_integration/test_intergrationreinforce.py
"""

# ////// Original bloc //////
# def mlp(x, sizes, activation=tf.tanh, output_activation=None):
#     # Build a feedforward neural network.
#     for size in sizes[:-1]:
#         x = tf.layers.dense(x, units=size, activation=activation)
#     return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

# ////// Original bloc //////
# def reward_to_go(rews):
#     n = len(rews)
#     rtgs = np.zeros_like(rews)
#     for i in reversed(range(n)):
#         rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
#     return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    # env = gym.make(env_name)                                                          # ////// Original bloc //////

    REINFORCE_integration_test = {                                                      # \\\\\\    My bloc    \\\\\\
        'prefered_environment': env_name,
        'paramameter_set_name': 'REINFORCE integration test on CartPole-v0',
        'batch_size_in_ts': batch_size,
        'max_epoch': epochs,
        'discounted_reward_to_go': False,
        'discout_factor': 0.999,
        'learning_rate': lr,
        'nn_h_layer_topo': tuple(hidden_sizes),
        'random_seed': 42,
        'hidden_layers_activation': tf.nn.tanh,  # tf.nn.relu,
        'output_layers_activation': None,
        # 'output_layers_activation': tf.nn.sigmoid,
        'render_env_every_What_epoch': 100,
        'print_metric_every_what_epoch': 5,
    }
    playground = BLOC.GymPlayground(env_name)                                           # \\\\\\    My bloc    \\\\\\
    env = playground.env                                                                # \\\\\\    My bloc    \\\\\\
    exp_spec = BLOC.ExperimentSpec()                                                    # \\\\\\    My bloc    \\\\\\
    exp_spec.set_experiment_spec(REINFORCE_integration_test)                            # \\\\\\    My bloc    \\\\\\
    consol_print_learning_stats = ConsolPrintLearningStats(                             # \\\\\\    My bloc    \\\\\\
        exp_spec, exp_spec.print_metric_every_what_epoch)                               # \\\\\\    My bloc    \\\\\\

    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    # obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)                          # ////// Original bloc //////
    obs_ph, act_ph, weights_ph = BLOC.gym_playground_to_tensorflow_graph_adapter(playground)    # \\\\\\    My bloc    \\\\\\

    # logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])                                    # ////// Original bloc //////
    # logits = BLOC.build_MLP_computation_graph(obs_ph, playground,                        # \\\\\\    My bloc    \\\\\\
    #                                           hidden_layer_topology=tuple(hidden_sizes)) # \\\\\\    My bloc    \\\\\\


    # make action selection op (outputs int actions, sampled from policy)
    # actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)            # ////// Original bloc //////
    # actions, log_p_all = BLOC.policy_theta_discrete_space(logits, playground)            # \\\\\\    My bloc    \\\\\\

    # make loss function whose gradient, for the right data, is policy gradient
    # weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)                         # ////// Original bloc //////
    # act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)                               # ////// Original bloc //////
    # action_masks = tf.one_hot(act_ph, n_acts)                                            # ////// Original bloc //////
    # log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)          # ////// Original bloc //////
    # loss = -tf.reduce_mean(weights_ph * log_probs)                                       # ////// Original bloc //////

    # (!) First silent error cause by uneven batch size                                    # \\\\\\    My bloc    \\\\\\
    # loss = BLOC.discrete_pseudo_loss(log_p_all, act_ph, weights_ph, playground)          # \\\\\\    My bloc    \\\\\\

    reinforce_policy = REINFORCEbrain.REINFORCE_policy(obs_ph, act_ph,  # \\\\\\    My bloc    \\\\\\
                                                       weights_ph, exp_spec, playground)             # \\\\\\    My bloc    \\\\\\
    (actions, _, loss) = reinforce_policy                                                  # \\\\\\    My bloc    \\\\\\

    # make train op
    # train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)                   # ////// Original bloc //////
    train_op = BLOC.policy_optimizer(loss, learning_rate=exp_spec.learning_rate)           # \\\\\\    My bloc    \\\\\\

    # \\\\\\    My bloc    \\\\\\
    date_now = datetime.now()
    run_str = "Run--{}h{}--{}-{}-{}".format(date_now.hour, date_now.minute, date_now.day, date_now.month, date_now.year)
    writer = tf_cv1.summary.FileWriter("./test_integration/graph/{}".format(run_str), tf_cv1.get_default_graph())

    the_TRAJECTORY_COLLECTOR = TrajectoryCollector(exp_spec, playground)                   # \\\\\\    My bloc    \\\\\\
    the_UNI_BATCH_COLLECTOR = UniformBatchCollector(exp_spec.batch_size_in_ts)             # \\\\\\    My bloc    \\\\\\

    # ////// Original bloc //////
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())

    # \\\\\\    My bloc    \\\\\\
    tf_cv1.set_random_seed(exp_spec.random_seed)
    np.random.seed(exp_spec.random_seed)
    with tf_cv1.Session() as sess:
        sess.run(tf_cv1.global_variables_initializer())     # initialize random variable in the computation graph
        consol_print_learning_stats.start_the_crazy_experiment()


        # for training policy
        def train_one_epoch():
            consol_print_learning_stats.next_glorious_epoch()                              # \\\\\\    My bloc    \\\\\\

            # ////// Original bloc //////
            # # make some empty lists for logging.
            # batch_obs = []          # for observations
            # batch_acts = []         # for actions
            # batch_weights = []      # for reward-to-go weighting in policy gradient
            # batch_rets = []         # for measuring episode returns
            # batch_lens = []         # for measuring episode lengths
            # ep_rews = []            # list for rewards accrued throughout ep

            # reset episode-specific variables
            obs = env.reset()       # first obs comes from starting distribution
            done = False            # signal from environment that episode is over

            # render first episode of each epoch
            finished_rendering_this_epoch = False

            consol_print_learning_stats.next_glorious_trajectory()                       # \\\\\\    My bloc    \\\\\\

            # collect experience by acting in the environment with current policy
            while True:

                # rendering
                if (not finished_rendering_this_epoch) and render:
                    env.render()

                # save obs
                # batch_obs.append(obs.copy())  # <-- (!) (Critical)                     # ////// Original bloc //////


                # # act in the environment
                # act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]                # ////// Original bloc //////
                # obs, rew, done, _ = env.step(act)                                      # ////// Original bloc //////

                step_observation = BLOC.format_single_step_observation(obs)              # \\\\\\    My bloc    \\\\\\
                action_array = sess.run(actions, feed_dict={obs_ph: step_observation})   # \\\\\\    My bloc    \\\\\\
                act = BLOC.format_single_step_action(action_array)                       # \\\\\\    My bloc    \\\\\\
                obs, rew, done, _ = playground.env.step(act)                             # \\\\\\    My bloc    \\\\\\

                # ////// Original bloc //////
                # # save action, reward
                # batch_acts.append(act)
                # ep_rews.append(rew)

                # (Critical) | Login the observation S_t that trigered the action A_t is critical.  # \\\\\\    My bloc    \\\\\\
                #            | If the observation is the one at time S_t+1, the agent wont learn    # \\\\\\    My bloc    \\\\\\
                the_TRAJECTORY_COLLECTOR.collect(obs, act, rew)  # <-- (!) Second silent error      # \\\\\\    My bloc    \\\\\\

                if done:

                    # ////// Original bloc //////
                    # # if episode is over, record info about episode
                    # ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    # batch_rets.append(ep_ret)
                    # batch_lens.append(ep_len)

                    trj_return = the_TRAJECTORY_COLLECTOR.trajectory_ended()              # \\\\\\    My bloc    \\\\\\
                    trj_container = the_TRAJECTORY_COLLECTOR.pop_trajectory_and_reset()   # \\\\\\    My bloc    \\\\\\
                    the_UNI_BATCH_COLLECTOR.collect(trj_container)                        # \\\\\\    My bloc    \\\\\\

                    consol_print_learning_stats.trajectory_training_stat(
                        the_trajectory_return=trj_return, timestep=len(trj_container))    # \\\\\\    My bloc    \\\\\\

                    # the weight for each logprob(a_t|s_t) is reward-to-go from t
                    # batch_weights += list(reward_to_go(ep_rews))                        # ////// Original bloc //////
                    # batch_weights += BLOC.reward_to_go(ep_rews)                        # \\\\\\    My bloc    \\\\\\

                    # reset episode-specific variables
                    obs, done, ep_rews = env.reset(), False, []

                    consol_print_learning_stats.next_glorious_trajectory()              # \\\\\\    My bloc    \\\\\\

                    # won't render again this epoch
                    finished_rendering_this_epoch = True

                    # ////// Original bloc //////
                    # # end experience loop if we have enough of it
                    # if len(batch_obs) > batch_size:
                    #     break

                    if not the_UNI_BATCH_COLLECTOR.is_not_full():                        # \\\\\\    My bloc    \\\\\\
                        break                                                            # \\\\\\    My bloc    \\\\\\

            # ////// Original bloc //////
            # # take a single policy gradient update step
            # batch_loss, _ = sess.run([loss, train_op],
            #                          feed_dict={
            #                             obs_ph: np.array(batch_obs),
            #                             act_ph: np.array(batch_acts),
            #                             weights_ph: np.array(batch_weights)
            #                          })

            batch_container = the_UNI_BATCH_COLLECTOR.pop_batch_and_reset()              # \\\\\\    My bloc    \\\\\\
            (batch_rets, batch_lens) = batch_container.compute_metric()                  # \\\\\\    My bloc    \\\\\\
            batch_obs = batch_container.batch_observations                               # \\\\\\    My bloc    \\\\\\
            batch_acts = batch_container.batch_actions                                   # \\\\\\    My bloc    \\\\\\
            batch_weights = batch_container.batch_Qvalues                                # \\\\\\    My bloc    \\\\\\

            feed_dictionary = BLOC.build_feed_dictionary([obs_ph, act_ph, weights_ph],   # \\\\\\    My bloc    \\\\\\
                                                         [batch_obs,                     # \\\\\\    My bloc    \\\\\\
                                                          batch_acts, batch_weights])    # \\\\\\    My bloc    \\\\\\
            batch_loss, _ = sess.run([loss, train_op],                                   # \\\\\\    My bloc    \\\\\\
                                     feed_dict=feed_dictionary)                          # \\\\\\    My bloc    \\\\\\

            return batch_loss, batch_rets, batch_lens

        # training loop
        for i in range(epochs):
            batch_loss, batch_rets, batch_lens = train_one_epoch()
            mean_return = np.mean(batch_rets)
            average_len = np.mean(batch_lens)

            # ////// Original bloc //////
            # print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
            #       (i, batch_loss, mean_return, average_len))

            # \\\\\\    My bloc    \\\\\\
            consol_print_learning_stats.epoch_training_stat(
                epoch_loss=batch_loss,
                epoch_average_trjs_return=mean_return,
                epoch_average_trjs_lenght=average_len,
                number_of_trj_collected=0,
                total_timestep_collected=0
            )

            yield (i, batch_loss, mean_return, average_len)


    print("\n>>> Close session\n")
    writer.close()
    playground.env.close()
    tf_cv1.reset_default_graph()
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

