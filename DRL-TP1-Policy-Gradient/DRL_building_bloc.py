# coding=utf-8
import copy

import gym
import pretty_printing
import numpy as np

import tensorflow as tf
from tensorflow import keras
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
    def __init__(self, timestep_max_per_trajectorie=20, trajectories_batch_size=10, max_epoch=2, discout_factor=1,
                 learning_rate=1e-2,
                 neural_net_hidden_layer_topology: list = [32, 32], random_seed=42):
        """
        Gather the specification for a experiement
        
        note:
          |     EPOCH definition:
          |         In our casse, collecting and updating the gradient of a set of trajectories of
          |         size=trajectorie_batch_size is equal to one EPOCH


        # todo: add a param for the neural net configuration via a dict fed as a argument
        """
        self.timestep_max_per_trajectorie = timestep_max_per_trajectorie         # horizon
        self.trajectories_batch_size = trajectories_batch_size
        self.max_epoch = max_epoch
        self.discout_factor = discout_factor
        self.learning_rate = learning_rate

        self.nn_h_layer_topo = neural_net_hidden_layer_topology
        self.random_seed = random_seed
        # todo: self.hidden_layer_activation_function
        # todo: self.output_layer_activation_function
        # todo: any NN usefull param

        assert isinstance(neural_net_hidden_layer_topology, list)

    def get_agent_training_spec(self):
        """
        Return specification related to the agent training
        :return: ( trajectories_batch_size, max_epoch, timestep_max_per_trajectorie )
        :rtype: (int, int, int)
        """
        return self.trajectories_batch_size, self.max_epoch, self.timestep_max_per_trajectorie

    def get_neural_net_spec(self):
        """
        Return the specification related to the neural net construction
        :return:
        :rtype:
        """
        return self.nn_h_layer_topo, self.random_seed

class GymPlayground(object):
    def __init__(self, environment_name='LunarLanderContinuous-v2'):
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

        :param environment_name: a gym environment
        :type environment_name: str
        """

        self.ENVIRONMENT_NAME = environment_name
        # self.env = None

        try:
            self.env = gym.make(self.ENVIRONMENT_NAME)
        except gym.error.Error as e:
            raise gym.error.Error("GymPlayground did not find the specified Gym environment.") from e


        if isinstance(self.env.action_space, gym.spaces.Box):
            print("\n\n>>> Action space is Contiuous")
            self.ACTION_SPACE_SHAPE = self.env.action_space.shape
        else:
            print("\n\n>>> Action space is Discrete")
            self.ACTION_SPACE_SHAPE = self.env.action_space.n

        self.OBSERVATION_SPACE_SHAPE = self.env.observation_space.shape

        if self.ENVIRONMENT_NAME == 'LunarLanderContinuous-v2':
            action_space_doc = "\tAction is two floats [main engine, left-right engines].\n" \
                           "\tMain engine: -1..0 off, 0..+1 throttle from 50% to 100% power.\n" \
                           "\t\t\t\t(!) Engine can't work with less than 50% power.\n" \
                           "\tLeft-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off\n\n"
            info_str = pretty_printing.environnement_doc_str(self.env, action_space_doc=action_space_doc)
        else:
            info_str = pretty_printing.environnement_doc_str(self.env)

        print(info_str)


    def get_environment_spec(self):
        """
        Return specification related to the gym environment
        :return: (OBSERVATION_SPACE_SHAPE, ACTION_SPACE_SHAPE, ENVIRONMENT_NAME)
        :rtype: tuple
        """
        return self.OBSERVATION_SPACE_SHAPE, self.ACTION_SPACE_SHAPE, self.ENVIRONMENT_NAME


def build_MLP_computation_graph(input_placeholder: tf.Tensor, action_placeholder_shape: tf.TensorShape,
                                hidden_layer_topology: list = [32, 32], hidden_layers_activation: tf.Tensor = tf.tanh,
                                output_layers_activation: tf.Tensor = tf.sigmoid) -> tf.Tensor:
    """
    Builder function for Low Level TensorFlow API.
    Return a Multi Layer Perceptron computatin graph with topology:

        input_placeholder | *hidden_layer_topology | logits_layer

    The last layer is called the 'logits' (aka: the raw output of the MLP)

    In the context of deep learning, 'logits' is the equivalent of 'raw output' of our prediction.
    It will later be transform into probabilies using the 'softmax function'

    :param input_placeholder:
    :type input_placeholder: tf.Tensor
    :param action_placeholder_shape:
    :type action_placeholder_shape:
    :param hidden_layer_topology:
    :type hidden_layer_topology:
    :param hidden_layers_activation:
    :type hidden_layers_activation:
    :param output_layers_activation:
    :type output_layers_activation:
    :return: a well construct computation graph
    :rtype: tf.Tensor
    """
    assert isinstance(input_placeholder, tf.Tensor)
    assert isinstance(action_placeholder_shape, tf.TensorShape)
    assert isinstance(hidden_layer_topology, list)

    with tf.name_scope(vocab.Multi_Layer_Perceptron) as scope:
        # Create input layer
        ops = keras.layers.Dense(hidden_layer_topology[0], input_shape=input_placeholder.shape,
                                  activation=hidden_layers_activation, name=vocab.input_layer)

        parent_layer = ops(input_placeholder)

        # create & connect all hidden layer
        for id in range(len(hidden_layer_topology)):
            h_layer = keras.layers.Dense(hidden_layer_topology[id], activation=hidden_layers_activation, name='{}{}'.format(vocab.hidden_, id + 1))
            parent_layer = h_layer(parent_layer)

        # create & connect the ouput layer: the logits
        logits = keras.layers.Dense(action_placeholder_shape.dims[-1], activation=output_layers_activation, name=vocab.logits)

        return logits(parent_layer)


def continuous_space_placeholder(space, name=None):
    return tf_cv1.placeholder(dtype=tf.float32, shape=(None, *space), name=name)

def discrete_space_placeholder(space: tuple, name=None):
    if isinstance(space, tuple):
        space = space[0]
    return tf_cv1.placeholder(dtype=tf.int32, shape=(None, space), name=name)


def gym_playground_to_tensorflow_graph_adapter(playground: GymPlayground) -> (tf.Tensor, tf.Tensor):
    """
    Configure handle for feeding value to the computation graph
            Continuous space    -->     dtype=tf.float32
            Discrete scpace     -->     dtype=tf.int32
    :param playground:
    :type playground: GymPlayground
    :return: input_placeholder, output_placeholder
    :rtype: (tf.Tensor, tf.Tensor)
    """
    assert isinstance(playground, GymPlayground), "\n\n>>> gym_playground_to_tensorflow_graph_adapter() expected a builded GymPlayground.\n\n"

    if isinstance(playground.env.observation_space, gym.spaces.Box):
        """observation space is continuous"""
        input_placeholder = continuous_space_placeholder(playground.OBSERVATION_SPACE_SHAPE, name=vocab.input_placeholder)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """observation space is discrete"""
        input_placeholder = discrete_space_placeholder(playground.OBSERVATION_SPACE_SHAPE, name=vocab.input_placeholder)
    else:
        raise NotImplementedError

    if isinstance(playground.env.action_space, gym.spaces.Box):
        """action space is continuous"""
        output_placeholder = continuous_space_placeholder(playground.ACTION_SPACE_SHAPE, name=vocab.output_placeholder)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """action space is discrete"""
        output_placeholder = discrete_space_placeholder(playground.ACTION_SPACE_SHAPE, name=vocab.output_placeholder)
    else:
        raise NotImplementedError

    return input_placeholder, output_placeholder


def policy_theta_discrete_space(logits_layer: tf.Tensor, action_placeholder_shape: tf.TensorShape, playground: GymPlayground):
    """
    Policy theta for discrete space --> actions are sampled from a categorical distribution
    """
    assert isinstance(playground.env.action_space, gym.spaces.Discrete)
    assert isinstance(logits_layer, tf.Tensor)
    assert isinstance(action_placeholder_shape, tf.TensorShape)
    logits_layer.shape.assert_is_compatible_with(action_placeholder_shape)

    with tf.name_scope(vocab.policy_theta_discrete) as scope:
        # convert the logits layer (aka: raw output) to probabilities
        log_probabilities = tf.nn.log_softmax(logits_layer)
        oversize_policy_theta = tf.random.categorical(log_probabilities, num_samples=1)

        # Remove single-dimensional entries from the shape of the array since we only take one sample from the distribution
        sampled_action = tf.squeeze(oversize_policy_theta, axis=1)

        return sampled_action, log_probabilities


def policy_theta_continuous_space(logits_layer: tf.Tensor, action_placeholder_shape: tf.TensorShape, playground: GymPlayground):
    """
    Policy theta for continuous space --> actions are sampled from a gausian distribution
    """
    assert isinstance(playground.env.action_space, gym.spaces.Box)
    assert isinstance(logits_layer, tf.Tensor)
    assert isinstance(action_placeholder_shape, tf.TensorShape)
    logits_layer.shape.assert_is_compatible_with(action_placeholder_shape)

    with tf.name_scope(vocab.policy_theta_continuous) as scope:
        # convert the logits layer (aka: raw output) to probabilities
        logits_layer = tf.identity(logits_layer, name='mu')

        raise NotImplementedError   # todo: implement
        log_standard_deviation = NotImplemented  # (!) todo
        standard_deviation = NotImplemented  # (!) todo --> compute standard_deviation
        logit_layer_shape = tf.shape(logits_layer)
        sampled_action = logits_layer + tf.random_normal(logit_layer_shape) * standard_deviation
        return sampled_action, log_standard_deviation


def REINFORCE_agent(observation_placeholder: tf.Tensor, action_placeholder: tf.Tensor, playground: GymPlayground,
                    experiment_spec: ExperimentSpec):
    """
    The learning agent. Base on the REINFORCE paper todo: citation
    (aka: Vanila policy gradient)
    """
    theta_mlp = build_MLP_computation_graph(observation_placeholder, action_placeholder.shape,
                                                 experiment_spec.nn_h_layer_topo)

    if isinstance(playground.env.action_space, gym.spaces.Discrete):
        policy_theta, log_probabilities = policy_theta_discrete_space(theta_mlp, action_placeholder.shape, playground)

        assert log_probabilities.shape.is_compatible_with(action_placeholder.shape), "the action_placeholder is incompatible with Discrete space, {} != {}".format(action_placeholder.shape, log_probabilities.shape)

        sampled_action = policy_theta

        # # compute the log probabilitie from sampled action
        # sampled_action_mask = tf.one_hot(sampled_action, depth=playground.ACTION_SPACE_SHAPE)
        # sampled_action_log_probability = tf.reduce_sum(log_probabilities * sampled_action_mask, axis=1)
        #
        # # compute the log probabilitie from action feed to the computetation graph
        # action_mask = tf.one_hot(action_placeholder, depth=playground.ACTION_SPACE_SHAPE)
        # feed_action_log_probability = tf.reduce_sum(log_probabilities * action_mask, axis=1)

        pseudo_loss = discrete_pseudo_loss(action_placeholder, theta_mlp)

    elif isinstance(playground.env.action_space, gym.spaces.Box):
        policy_theta, log_standard_deviation = policy_theta_continuous_space(theta_mlp, action_placeholder.shape, playground)

        assert policy_theta.shape.is_compatible_with(action_placeholder.shape), "the action_placeholder is incompatible with Continuous space, {} != {}".format(action_placeholder.shape, policy_theta.shape)

        sampled_action = NotImplemented
        sampled_action_log_probability = NotImplemented
        feed_action_log_probability = NotImplemented
        raise NotImplementedError   # todo: implement
    else:
        print("\n>>> The given playground {} is of action space type {}. The agent implementation does not support it yet\n\n".format(playground.ENVIRONMENT_NAME, playground.env.action_space))
        raise NotImplementedError

    return sampled_action, theta_mlp, pseudo_loss # sampled_action_log_probability, feed_action_log_probability


def discrete_pseudo_loss(action_placeholder, theta_mlp):
    q_values = NotImplemented # todo --> implement


    with tf.name_scope(vocab.pseudo_loss) as scope:
        negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits_v2(labels=action_placeholder, logits=theta_mlp,
                                                                          name='negative_likelihoods')

        # todo --> finish implement

        # weighted_negative_likelihoods = tf.multiply(negative_likelihoods, q_values)
        # pseudo_loss = tf.reduce_mean(weighted_negative_likelihoods)
        # return pseudo_loss
        return None  # todo --> temp hack3


# todo --> finish implementing
# todo --> unit test
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
    assert isinstance(placeholders, list), "Wrong input type, placeholders must be a list of tensorflow placeholder"
    assert isinstance(arrays_of_values, list), "Wrong input type, arrays_of_values must be a list of numpy array"
    assert len(placeholders) == len(arrays_of_values), "placeholders and arrays_of_values must be of the same lenght"
    for placeholder in placeholders:
        assert isinstance(placeholder, tf.Tensor), "Wrong input type, placeholders must be a list of tensorflow placeholder"
    for ar in arrays_of_values:
        assert isinstance(ar, np.ndarray), "Wrong input type, arrays_of_values must be a list of numpy array"

    feed_dict = dict
    for placeholder, array in zip(placeholders, arrays_of_values):
        feed_dict[placeholder] = array

    return feed_dict


class TrajectorieContainer(object):
    def __init__(self, observations: list, actions: list, rewards: list, dtype=None):
        """
        Container for events collected at every timestep of a single trajectorie

        Note from numpy about dtype:
            "This argument can only be used to 'upcast' the array.  For downcasting, use the .astype(t) method."
        """
        assert len(observations) == len(actions) == len(rewards)
        self.observations = np.array(observations, dtype=dtype)
        self.actions = np.array(actions, dtype=dtype)
        self.rewards = np.array(rewards, dtype=dtype)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, timestep: int):
        """
        Return the values (observation, action, reward) at the given timestep
        :param timestep: the exact timestep number (not the list index)
        :type timestep: int
        :return:  (observations, actions, rewards) at timestep t
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        """
        ts = timestep - 1
        obs = self.observations[ts]
        act = self.actions[ts]
        rew = self.rewards[ts]
        return obs, act, rew

    def unpack(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Unpack the full trajectorie as a tuple of numpy array
        :return: (observations, actions, rewards) arrays
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        """
        return self.observations, self.actions, self.rewards

class TimestepCollector(object):
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground):
        self._experiment_spec = experiment_spec
        # self._playground_spec = playground.get_environment_spec()
        self._observations = []
        self._actions = []
        self._rewards = []
        self.step_count = 0

    def __call__(self, observation, action, reward, *args, **kwargs) -> None:
        try:
            assert self.step_count < self._experiment_spec.timestep_max_per_trajectorie, \
                "Max timestep per trajectorie reached so the TimestepCollector is full."
            self._observations.append(observation)
            self._actions.append(action)
            self._rewards.append(reward)
            self.step_count += 1
        except AssertionError as ae:
            raise ae
        return None

    def append(self, observation, action, reward) -> None:
        """Collect observation, action, reward for one timestep"""
        self.__call__(observation, action, reward)
        return None

    def _normalize_the_collected_trajectorie_lenght(self) -> None:
        """
        Complete sampled trajectorie with dummy value to make all sampled trajectories of even lenght
        :return: None
        """
        raise NotImplementedError   # todo: implement for case batch size > 1

        t_timestep = len(self._observations)
        d = 0   # todo --> confirm chosen value do not affect training
        delta_t = self._experiment_spec.max_epoch - t_timestep
        for t in range(delta_t):
            self._observations.append(d)
            self._actions.append(d)
            self._rewards.append(d)
        return None

    def _reset(self):
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self.step_count = 0
        return None

    def get_collected_trajectorie_and_reset(self) -> TrajectorieContainer:
        """
        Return the sampled trajectorie as 3 np.ndarray (observations, actions, rewards)
        than reset the container ready for the next trajectorie sampling.

        :return: (observations, actions, rewards)
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        """

        # todo --> validate dtype for discrete case
        trajectorie_container = TrajectorieContainer(self._observations, self._actions, self._rewards)

        self._reset()
        return trajectorie_container

    def __del__(self):
        self._reset()


def reward_to_go(rewards: list):
    assert isinstance(rewards, list)
    np_backward_rewards = np.array(rewards[::-1])
    reward_to_go = np.cumsum(np_backward_rewards)
    return reward_to_go[::-1]


def discounted_reward_to_go(rewards: list, experiment_spec: ExperimentSpec):
    assert isinstance(rewards, list)
    discount = experiment_spec.discout_factor   # lambda
    assert (0 <= discount) and (discount <= 1)
    backward_rewards = rewards[::-1]
    discounted_reward_to_go = np.zeros_like(rewards)

    for r in range(len(rewards)):
        exp = 0
        for i in range(r, len(rewards)):
            discounted_reward_to_go[i] += discount**exp * backward_rewards[r]
            exp += 1

    return discounted_reward_to_go[::-1]


# ice-box
class TrajectoriesBatchContainer(object):
    """Iterable container by timestep increment for storage & retrieval of component of a sampled trajectory"""
    def __init__(self, max_trajectory_lenght: int, playground: GymPlayground):
        self.max_trajectory_lenght = max_trajectory_lenght
        self.playground = playground

        raise NotImplementedError   # todo: implement --> ice-box: implement for case batch size > 1

# ice-box
def epoch_buffer(trajectory_placeholder):
    raise NotImplementedError   # todo: implement


