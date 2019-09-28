# coding=utf-8
from typing import Any, Union

import gym
from gym.wrappers import TimeLimit

import pretty_printing
import numpy as np
import tensorflow as tf

tf_cv1 = tf.compat.v1   # shortcut

import tensorflow_weak_warning_supressor as no_cpu_compile_warn
no_cpu_compile_warn.execute()

from vocabulary import rl_name
vocab = rl_name()


"""
Start TensorBoard in terminal:
    tensorboard --logdir=DRL-TP1-Policy-Gradient/graph/

In browser, go to:
    http://0.0.0.0:6006/ 
"""

class ExperimentSpec(object):
    def __init__(self, batch_size_in_ts=5000, max_epoch=2, discout_factor=0.99, learning_rate=1e-2,
                 neural_net_hidden_layer_topology: tuple = (32, 32), random_seed=42, discounted_reward_to_go=True,
                 environment_name='CartPole-v1', print_metric_every_what_epoch=5):
        """
        Gather the specification for a experiement
        
        note:
          |     EPOCH definition:
          |         In our casse, collecting and updating the gradient of a set of trajectories of
          |         size=trajectorie_batch_size is equal to one EPOCH


        # todo: add a param for the neural net configuration via a dict fed as a argument
        :param print_metric_every_what_epoch:
        :type print_metric_every_what_epoch:
        :param environment_name:
        :type environment_name:
        :param discounted_reward_to_go:
        :type discounted_reward_to_go:
        """

        self.paramameter_set_name = 'default'
        self.prefered_environment = environment_name

        self.batch_size_in_ts = batch_size_in_ts
        self.max_epoch = max_epoch
        self.discout_factor: float = discout_factor
        self.learning_rate = learning_rate
        self.discounted_reward_to_go = discounted_reward_to_go

        self.nn_h_layer_topo = neural_net_hidden_layer_topology
        self.random_seed = random_seed
        self.hidden_layers_activation: tf.Tensor = tf.nn.tanh
        self.output_layers_activation: tf.Tensor = tf.nn.sigmoid

        self.render_env_every_What_epoch = 100
        self.log_every_step = 1000
        self.print_metric_every_what_epoch = print_metric_every_what_epoch

        # (nice to have) todo --> add any NN usefull param:

        self._assert_param()

    def _assert_param(self):
        assert (0 <= self.discout_factor) and (self.discout_factor <= 1)
        assert isinstance(self.nn_h_layer_topo, tuple)

    def get_agent_training_spec(self):
        """
        Return specification related to the agent training
        :return: ( batch_size_in_ts, max_epoch, timestep_max_per_trajectorie )
        :rtype: (int, int, int)
        """
        return {
            'batch_size_in_ts': self.batch_size_in_ts,
            'max_epoch': self.max_epoch,
            'discout_factor': self.discout_factor,
            'learning_rate': self.learning_rate,
        }

    def get_neural_net_spec(self):
        """
        Return the specification related to the neural net construction
        :return:
        :rtype:
        """
        return {
            'nn_h_layer_topo': self.nn_h_layer_topo,
            'random_seed': self.random_seed ,
            'hidden_layers_activation': self.hidden_layers_activation,
            'output_layers_activation': self.output_layers_activation,
        }

    def set_experiment_spec(self, dict_param: dict):

        for str_k, v in dict_param.items():
            str_k: str
            self.__setattr__(str_k, v)

        self._assert_param()

        print("\n\n>>> Switching to parameter: {}".format(self.paramameter_set_name),
              self.get_agent_training_spec(),
              self.get_neural_net_spec(),
              "\n")
        return None


class GymPlayground(object):
    def __init__(self, environment_name='LunarLanderContinuous-v2', print_env_info=False):
        """
        Setup the learning playground for the agent (the environment in witch he will play) and gather relevant spec

        Note: 'LunarLanderContinuous-v2' (DEFAUT environment)
                env: <TimeLimit<LunarLanderContinuous<LunarLanderContinuous-v2>>>
                Metadata:
                    {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

                OBSERVATION SPACE:
                    Type: Box(8,)
                        Higher bound: [inf inf inf inf inf inf inf inf]
                        Lower bound: [-inf -inf -inf -inf -inf -inf -inf -inf]

                ACTION SPACE:
                    Type: Box(2,)
                        Higher bound: [1. 1.]
                        Lower bound: [-1. -1.]

                Action is two floats [main engine, left-right engines].
                    Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power.
                                (!) Engine can't work with less than 50% power.
                    Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

                REWARD range: (-inf, inf)

        Note: 'LunarLander-v2' (Discrete version):
                env: <TimeLimit<LunarLander<LunarLander-v2>>>
                Metadata:
                    {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

            OBSERVATION SPACE:
                Type: Box(8,)
                    Higher bound: [inf inf inf inf inf inf inf inf]
                    Lower bound: [-inf -inf -inf -inf -inf -inf -inf -inf]

            ACTION SPACE:
                Type: Discrete(4)
                    Higher bound: 1
                    Lower bound: 0

            REWARD range: (-inf, inf)

        :param print_env_info:
        :type print_env_info:
        :param environment_name: a gym environment
        :type environment_name: str
        """

        self.ENVIRONMENT_NAME = environment_name

        try:
            self._env = gym.make(self.ENVIRONMENT_NAME)
        except gym.error.Error as e:
            raise gym.error.Error("GymPlayground did not find the specified Gym environment.") from e

        info_str = ""
        if isinstance(self._env.action_space, gym.spaces.Box):
            info_str += "\n\n>>> Action space is Contiuous"
            self.ACTION_SPACE = self._env.action_space
            dimension = self.ACTION_SPACE.shape
            self.ACTION_CHOICES = [*dimension][-1]
        else:
            info_str += "\n\n>>> Action space is Discrete"
            self.ACTION_SPACE = self._env.action_space
            self.ACTION_CHOICES = self.ACTION_SPACE.n

        self.OBSERVATION_SPACE = self._env.observation_space

        if print_env_info:
            if self.ENVIRONMENT_NAME == 'LunarLanderContinuous-v2':
                action_space_doc = "\tAction is two floats [main engine, left-right engines].\n" \
                               "\tMain engine: -1..0 off, 0..+1 throttle from 50% to 100% power.\n" \
                               "\t\t\t\t(!) Engine can't work with less than 50% power.\n" \
                               "\tLeft-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off\n\n"
                info_str += pretty_printing.environnement_doc_str(self._env, action_space_doc=action_space_doc)
            else:
                info_str += pretty_printing.environnement_doc_str(self._env)

            print(info_str)

    @property
    def env(self) -> Union[TimeLimit, Any]:
        return self._env


    def get_environment_spec(self):
        """
        Return specification related to the gym environment
        :return: (OBSERVATION_SPACE, ACTION_SPACE, ACTION_CHOICES, ENVIRONMENT_NAME)
        :rtype: tuple
        """
        return self.OBSERVATION_SPACE, self.ACTION_SPACE, self.ACTION_CHOICES, self.ENVIRONMENT_NAME


def build_MLP_computation_graph(input_placeholder: tf.Tensor, playground: GymPlayground,
                                hidden_layer_topology: tuple = (32, 32), hidden_layers_activation: tf.Tensor = tf.nn.tanh,
                                output_layers_activation: tf.Tensor = None,
                                name_scope=vocab.Multi_Layer_Perceptron) -> tf.Tensor:
    """
    Builder function for Low Level TensorFlow API.
    Return a Multi Layer Perceptron computatin graph with topology:

        input_placeholder | *hidden_layer_topology | logits_layer

    The last layer is called the 'logits' (aka: the raw output of the MLP)

    In the context of deep learning, 'logits' is the equivalent of 'raw output' of our prediction.
    It will later be transform into probabilies using the 'softmax function'

    :param playground:
    :type playground:
    :param input_placeholder:
    :type input_placeholder: tf.Tensor
    :param hidden_layer_topology:
    :type hidden_layer_topology:
    :param hidden_layers_activation:
    :type hidden_layers_activation:
    :param output_layers_activation:
    :type output_layers_activation:
    :param name_scope:
    :type name_scope:
    :return: a well construct computation graph
    :rtype: tf.Tensor
    """
    assert isinstance(input_placeholder, tf.Tensor)
    assert isinstance(playground, GymPlayground)
    assert isinstance(hidden_layer_topology, tuple)

    # with tf.name_scope(name_scope) as scope:
    #     # Create input layer
    #     ops = keras.layers.Dense(hidden_layer_topology[0], input_shape=input_placeholder.shape,
    #                               activation=hidden_layers_activation, name=vocab.input_layer)
    #
    #     parent_layer = ops(input_placeholder)
    #
    #     # create & connect all hidden layer
    #     for id in range(len(hidden_layer_topology)):
    #         h_layer = keras.layers.Dense(hidden_layer_topology[id], activation=hidden_layers_activation, name='{}{}'.format(vocab.hidden_, id + 1))
    #         parent_layer = h_layer(parent_layer)
    #
    #     # create & connect the ouput layer: the logits
    #     logits = keras.layers.Dense(playground.ACTION_CHOICES, activation=output_layers_activation, name=vocab.logits)
    #
    #     return logits(parent_layer)

    with tf.variable_scope(name_or_scope=name_scope):
    # with tf.name_scope(name_scope) as scope:
        h_layer = input_placeholder

        # create & connect all hidden layer
        for id in range(len(hidden_layer_topology)):
            h_layer = tf_cv1.layers.dense(h_layer, hidden_layer_topology[id], activation=hidden_layers_activation, name='{}{}'.format(vocab.hidden_, id + 1))

        # create & connect the ouput layer: the logits
        logits = tf_cv1.layers.dense(h_layer, playground.ACTION_CHOICES, activation=output_layers_activation, name=vocab.logits)

    return logits


def continuous_space_placeholder(space: gym.spaces.Box, name=None, shape_constraint: tuple = None) -> tf.Tensor:
    assert isinstance(space, gym.spaces.Box)
    space_shape = space.shape
    if shape_constraint is not None:
        shape = (*shape_constraint, *space_shape)
    else:
        shape = (None, *space_shape)
    return tf_cv1.placeholder(dtype=tf.float32, shape=shape, name=name)


def discrete_space_placeholder(space: gym.spaces.Discrete, name=None, shape_constraint: tuple = None) -> tf.Tensor:
    assert isinstance(space, gym.spaces.Discrete)
    if shape_constraint is not None:
        shape = (*shape_constraint,)
    else:
        shape = (None,)
    return tf_cv1.placeholder(dtype=tf.int32, shape=shape, name=name)


def gym_playground_to_tensorflow_graph_adapter(playground: GymPlayground, action_shape_constraint: tuple = None,
                                               obs_shape_constraint: tuple = None) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    Configure handle for feeding value to the computation graph
            Continuous space    -->     dtype=tf.float32
            Discrete scpace     -->     dtype=tf.int32
    :param action_shape_constraint:
    :type action_shape_constraint:
    :param obs_shape_constraint:
    :type obs_shape_constraint:
    :param playground:
    :type playground: GymPlayground
    :return: input_placeholder, output_placeholder, Q_values_placeholder
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
    """
    assert isinstance(playground, GymPlayground), "\n\n>>> Expected a builded GymPlayground.\n\n"


    if isinstance(playground.env.observation_space, gym.spaces.Box):
        """observation space is continuous"""
        input_placeholder = continuous_space_placeholder(playground.OBSERVATION_SPACE, vocab.input_placeholder, obs_shape_constraint)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """observation space is discrete"""
        input_placeholder = discrete_space_placeholder(playground.OBSERVATION_SPACE, vocab.input_placeholder, obs_shape_constraint)
    else:
        raise NotImplementedError

    if isinstance(playground.env.action_space, gym.spaces.Box):
        """action space is continuous"""
        output_placeholder = continuous_space_placeholder(playground.ACTION_SPACE, vocab.output_placeholder, action_shape_constraint)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """action space is discrete"""
        output_placeholder = discrete_space_placeholder(playground.ACTION_SPACE, vocab.output_placeholder, action_shape_constraint)
    else:
        raise NotImplementedError

    if obs_shape_constraint is not None:
        shape = (*obs_shape_constraint,)
    else:
        shape = (None,)
    Q_values_ph = tf_cv1.placeholder(dtype=tf.float32, shape=shape, name=vocab.Qvalues_placeholder)

    return input_placeholder, output_placeholder, Q_values_ph


def policy_theta_discrete_space(logits_layer: tf.Tensor, playground: GymPlayground) -> (tf.Tensor, tf.Tensor):
    """Policy theta for discrete space --> actions are sampled from a categorical distribution

    :param logits_layer:
    :type logits_layer: tf.Tensor
    :param playground:
    :type playground: GymPlayground
    :return: (sampled_action, log_p_all)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    assert isinstance(playground.env.action_space, gym.spaces.Discrete)
    assert isinstance(logits_layer, tf.Tensor)
    assert logits_layer.shape.as_list()[-1] == playground.ACTION_CHOICES

    with tf.name_scope(vocab.policy_theta_D) as scope:
        # convert the logits layer (aka: raw output) to probabilities
        log_p_all = tf.nn.log_softmax(logits_layer)
        oversize_policy_theta = tf.random.categorical(logits_layer, num_samples=1)

        # Remove single-dimensional entries from the shape of the array since we only take one sample from the distribution
        sampled_action = tf.squeeze(oversize_policy_theta, axis=1)

        # # Compute the log probabilitie from sampled action
        # # todo --> sampled_action_log_probability unit test
        # sampled_action_mask = tf.one_hot(sampled_action, depth=playground.ACTION_CHOICES)
        # log_probabilities_matrix = tf.multiply(sampled_action_mask, log_p_all)
        # sampled_action_log_probability = tf.reduce_sum(log_probabilities_matrix, axis=1)


        return sampled_action, log_p_all


def policy_theta_continuous_space(logits_layer: tf.Tensor, playground: GymPlayground):
    """
    Policy theta for continuous space --> actions are sampled from a gausian distribution
    status: ice-box until next sprint
    """
    assert isinstance(playground.env.action_space, gym.spaces.Box)
    assert isinstance(logits_layer, tf.Tensor)
    assert logits_layer.shape.as_list()[-1] == playground.ACTION_CHOICES

    with tf.name_scope(vocab.policy_theta_C) as scope:
        # convert the logits layer (aka: raw output) to probabilities
        logits_layer = tf.identity(logits_layer, name='mu')

        raise NotImplementedError   # todo: implement
        # log_standard_deviation = NotImplemented  # (!) todo
        # standard_deviation = NotImplemented  # (!) todo --> compute standard_deviation
        # logit_layer_shape = tf.shape(logits_layer)
        # sampled_action = logits_layer + tf.random_normal(logit_layer_shape) * standard_deviation
        # return sampled_action, log_standard_deviation


def REINFORCE_policy(observation_placeholder: tf.Tensor, action_placeholder: tf.Tensor, Q_values_placeholder: tf.Tensor,
                     experiment_spec: ExperimentSpec, playground: GymPlayground) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    The learning agent. Base on the REINFORCE paper (aka: Vanila policy gradient)
    todo --> add references
    todo --> implement for continuous space (status: Ice-box until next sprint)

    :param observation_placeholder:
    :type observation_placeholder: tf.Tensor
    :param action_placeholder:
    :type action_placeholder: tf.Tensor
    :param Q_values_placeholder:
    :type Q_values_placeholder: tf.Tensor
    :param playground:
    :type playground: GymPlayground
    :param experiment_spec:
    :type experiment_spec: ExperimentSpec
    :return: (sampled_action, theta_mlp, pseudo_loss)
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
    """

    with tf.name_scope(vocab.REINFORCE) as scope:

        theta_mlp = build_MLP_computation_graph(observation_placeholder, playground,
                                                experiment_spec.nn_h_layer_topo,
                                                hidden_layers_activation=experiment_spec.hidden_layers_activation,
                                                output_layers_activation=experiment_spec.output_layers_activation,
                                                name_scope=vocab.theta_NeuralNet)

        # /---- discrete case -----
        if isinstance(playground.env.action_space, gym.spaces.Discrete):
            assert observation_placeholder.shape.as_list()[-1] == playground.OBSERVATION_SPACE.shape[0], \
                "the observation_placeholder is incompatible with environment, " \
                "{} != {}".format(observation_placeholder.shape.as_list()[-1], playground.OBSERVATION_SPACE.shape[0])

            sampled_action, log_p_all = policy_theta_discrete_space(theta_mlp, playground)

            pseudo_loss = discrete_pseudo_loss(log_p_all, action_placeholder, Q_values_placeholder, playground)

            # pseudo_loss = tf.reduce_mean(tf.stop_gradient(Q_values_placeholder) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=theta_mlp, labels=action_placeholder), name=vocab.pseudo_loss)


        # /---- continuous case -----
        elif isinstance(playground.env.action_space, gym.spaces.Box):
            raise NotImplementedError   # todo: implement <-- ice-box until next sprint

            # policy_theta, log_standard_deviation = policy_theta_continuous_space(theta_mlp, playground)
            # sampled_action = NotImplemented
            # sampled_action_log_probability = NotImplemented

        # /---- other gym environment -----
        else:
            print("\n>>> The agent implementation does not support environment space "
                  "{} yet.\n\n".format(playground.env.action_space))
            raise NotImplementedError

    return sampled_action, theta_mlp, pseudo_loss



# <editor-fold desc="::Reward to go related function ...">
def reward_to_go(rewards: list) -> list:
    assert isinstance(rewards, list)
    np_backward_rewards = np.array(rewards[::-1])
    reward_to_go = np.cumsum(np_backward_rewards)
    return list(reward_to_go[::-1])

def reward_to_go_np(rewards: np.ndarray) -> np.ndarray:
    assert isinstance(rewards, np.ndarray)
    np_backward_rewards = np.flip(rewards)
    reward_to_go = np.cumsum(np_backward_rewards)
    return np.flip(reward_to_go)


def discounted_reward_to_go(rewards: list, experiment_spec: ExperimentSpec) -> list:
    """
    Compute the discounted reward to go iteratively

    (Priority) todo:refactor --> refactor using a gamma mask and matrix product & sum, instead of loop
    (nice to have) todo:refactor --> fct signature: pass gamma (the discount factor) explicitely

    :param rewards:
    :type rewards:
    :param experiment_spec:
    :type experiment_spec:
    :return:
    :rtype:
    """
    gamma = experiment_spec.discout_factor
    assert (0 <= gamma) and (gamma <= 1)
    assert isinstance(rewards, list)

    backward_rewards = rewards[::-1]
    discounted_reward_to_go = np.zeros_like(rewards)

    for r in range(len(rewards)):
        exp = 0
        for i in range(r, len(rewards)):
            discounted_reward_to_go[i] += gamma**exp * backward_rewards[r]
            exp += 1

    return list(discounted_reward_to_go[::-1])


def discounted_reward_to_go_np(rewards: np.ndarray, experiment_spec: ExperimentSpec) -> np.ndarray:
    gamma = experiment_spec.discout_factor
    assert (0 <= gamma) and (gamma <= 1)
    assert rewards.ndim == 1, "Current implementation only support array of rank 1"
    assert isinstance(rewards, np.ndarray)

    np_backward_rewards = np.flip(rewards)
    discounted_reward_to_go = np.zeros_like(rewards)

    # todo --> Since flip return a view, test if iterate on a pre flip ndarray and than post flip before return would be cleaner

    # refactor --> using a gamma mask and matrix product & sum, instead of loop
    for r in range(len(rewards)):
        exp = 0
        for i in range(r, len(rewards)):
            discounted_reward_to_go[i] += gamma**exp * np_backward_rewards[r]
            exp += 1

    return np.flip(discounted_reward_to_go)
# </editor-fold>


def discrete_pseudo_loss(log_p_all, action_placeholder: tf.Tensor, Q_values_placeholder: tf.Tensor,
                         playground: GymPlayground) -> tf.Tensor:
    """
    Pseudo loss for discrete action space only using Softmax cross entropy with logits
    """

    # \\\\\\    My bloc    \\\\\\
    # with tf.name_scope(vocab.pseudo_loss) as scope:

    # Step 1: Compute the log probabilitie of the current policy over the action space
    action_mask = tf.one_hot(action_placeholder, playground.ACTION_CHOICES)
    log_probabilities_matrix = tf.multiply(action_mask, log_p_all)
    log_probabilities = tf.reduce_sum(log_probabilities_matrix, axis=1)

    # Step 2: Compute the pseudo loss
    # note: tf.stop_gradient(Q_values_placeholder) prevent the backpropagation into the Q_values_placeholder
    #   |   witch contain rewards_to_go. It treat the values of the tensor as constant during backpropagation.
    weighted_likelihoods = tf.multiply(
        tf.stop_gradient(Q_values_placeholder), log_probabilities)
    pseudo_loss = -tf.reduce_mean(weighted_likelihoods)
    return pseudo_loss

    # ////// Original bloc //////
    # action_masks = tf.one_hot(action_placeholder, playground.ACTION_CHOICES)
    # # log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    # log_probs = tf.reduce_sum(action_masks * log_p_all, axis=1)
    # loss = -tf.reduce_mean(Q_values_placeholder * log_probs)
    return loss


def policy_optimizer(pseudo_loss: tf.Tensor, learning_rate: ExperimentSpec) -> tf.Operation:
    """
    Define the optimizing methode for training the REINFORE agent
    """
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(pseudo_loss, name=vocab.optimizer)


def build_feed_dictionary(placeholders: list, arrays_of_values: list) -> dict:
    """
    Build a feed dictionary ready to use in a TensorFlow run session.

    It map TF placeholder to corresponding numpy array of values so be advise, order is important.

    :param placeholders: a list of tensorflow placeholder
    :type placeholders: [tf.Tensor, ...]
    :param arrays_of_values: a list of numpy array
    :type arrays_of_values: [np.ndarray, ...]
    :return: a feed dictionary
    :rtype: dict
    """
    # todo --> finish implementing
    # todo --> unit test
    assert isinstance(placeholders, list), "Wrong input type, placeholders must be a list of tensorflow placeholder"
    assert isinstance(arrays_of_values, list), "Wrong input type, arrays_of_values must be a list of numpy array"
    assert len(placeholders) == len(arrays_of_values), "placeholders and arrays_of_values must be of the same lenght"
    for placeholder in placeholders:
        assert isinstance(placeholder, tf.Tensor), "Wrong input type, placeholders must be a list of tensorflow placeholder"

    # (Iceboxed) todo:investigate?? --> it's probably not required anymore:
    # for ar in arrays_of_values:
    #     assert isinstance(ar, np.ndarray), "Wrong input type, arrays_of_values must be a list of numpy array"

    feed_dict = dict()
    for placeholder, array in zip(placeholders, arrays_of_values):
        feed_dict[placeholder] = array

    return feed_dict


# todo:validate --> possible source of graph data input error:
def format_single_step_observation(observation: np.ndarray):
    """ Single trajectorie batch size hack for the computation graph observation placeholder
            Ex:
                    observation.shape = (8,)
                vs
                    np.expand_dims(observation, axis=0).shape = (1, 8)
    """
    assert observation.ndim == 1, "Watch out!! observation array is of wrong dimension {}".format(observation.shape)
    batch_size_one_observation = np.expand_dims(observation, axis=0)
    return batch_size_one_observation

# todo:validate --> possible source of graph data input error:
def format_single_step_action(action_array: np.ndarray):
    # todo --> unitest
    action = None
    try:
        action = action_array[0]
    except:

        if isinstance(action_array, np.ndarray):
            assert action_array.ndim == 1, "action_array is of dimension > 1: {}".format(action_array.ndim)
            action = np.squeeze(action_array)
        else:
            action = action_array
        assert isinstance(action, int), ("something is wrong with the 'format_single_step_action'. "
                                        "Should output a int instead of {}".format(action))
    finally:
        return action


