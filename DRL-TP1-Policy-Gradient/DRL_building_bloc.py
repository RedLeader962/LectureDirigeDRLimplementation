# coding=utf-8

import tensorflow_weak_warning_supressor as no_cpu_compile_warn
no_cpu_compile_warn.execute()

import gym
import pretty_printing
import numpy as np
import abc

import tensorflow as tf
from tensorflow import keras
tf_cv1 = tf.compat.v1   # shortcut


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
                 neural_net_hidden_layer_topology: tuple = (32, 32), random_seed=42):
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

        assert isinstance(neural_net_hidden_layer_topology, tuple)

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

        try:
            self.env = gym.make(self.ENVIRONMENT_NAME)
        except gym.error.Error as e:
            raise gym.error.Error("GymPlayground did not find the specified Gym environment.") from e


        if isinstance(self.env.action_space, gym.spaces.Box):
            print("\n\n>>> Action space is Contiuous")
            self.ACTION_SPACE = self.env.action_space
            dimension = self.ACTION_SPACE.shape
            self.ACTION_CHOICES = [*dimension][-1]
        else:
            print("\n\n>>> Action space is Discrete")
            self.ACTION_SPACE = self.env.action_space
            self.ACTION_CHOICES = self.ACTION_SPACE.n

        self.OBSERVATION_SPACE = self.env.observation_space

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
        :return: (OBSERVATION_SPACE, ACTION_SPACE, ACTION_CHOICES, ENVIRONMENT_NAME)
        :rtype: tuple
        """
        return self.OBSERVATION_SPACE, self.ACTION_SPACE, self.ACTION_CHOICES, self.ENVIRONMENT_NAME


def build_MLP_computation_graph(input_placeholder: tf.Tensor, playground: GymPlayground,
                                hidden_layer_topology: tuple = (32, 32), hidden_layers_activation: tf.Tensor = tf.tanh,
                                output_layers_activation: tf.Tensor = tf.sigmoid,
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

    with tf.name_scope(name_scope) as scope:
        # Create input layer
        ops = keras.layers.Dense(hidden_layer_topology[0], input_shape=input_placeholder.shape,
                                  activation=hidden_layers_activation, name=vocab.input_layer)

        parent_layer = ops(input_placeholder)

        # create & connect all hidden layer
        for id in range(len(hidden_layer_topology)):
            h_layer = keras.layers.Dense(hidden_layer_topology[id], activation=hidden_layers_activation, name='{}{}'.format(vocab.hidden_, id + 1))
            parent_layer = h_layer(parent_layer)

        # create & connect the ouput layer: the logits
        logits = keras.layers.Dense(playground.ACTION_CHOICES, activation=output_layers_activation, name=vocab.logits)

        return logits(parent_layer)


def continuous_space_placeholder(space: gym.spaces.Box, name=None) -> tf.Tensor:
    assert isinstance(space, gym.spaces.Box)
    space_shape = space.shape
    return tf_cv1.placeholder(dtype=tf.float32, shape=(None, *space_shape), name=name)


def discrete_space_placeholder(space: gym.spaces.Discrete, name=None) -> tf.Tensor:
    assert isinstance(space, gym.spaces.Discrete)
    return tf_cv1.placeholder(dtype=tf.int32, shape=(None,), name=name)


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
        input_placeholder = continuous_space_placeholder(playground.OBSERVATION_SPACE, name=vocab.input_placeholder)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """observation space is discrete"""
        input_placeholder = discrete_space_placeholder(playground.OBSERVATION_SPACE, name=vocab.input_placeholder)
    else:
        raise NotImplementedError

    if isinstance(playground.env.action_space, gym.spaces.Box):
        """action space is continuous"""
        output_placeholder = continuous_space_placeholder(playground.ACTION_SPACE, name=vocab.output_placeholder)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """action space is discrete"""
        output_placeholder = discrete_space_placeholder(playground.ACTION_SPACE, name=vocab.output_placeholder)
    else:
        raise NotImplementedError

    return input_placeholder, output_placeholder


def policy_theta_discrete_space(logits_layer: tf.Tensor, playground: GymPlayground) -> (tf.Tensor, tf.Tensor):
    """Policy theta for discrete space --> actions are sampled from a categorical distribution

    :param logits_layer:
    :type logits_layer: tf.Tensor
    :param playground:
    :type playground: GymPlayground
    :return: (sampled_action, log_probabilities)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    assert isinstance(playground.env.action_space, gym.spaces.Discrete)
    assert isinstance(logits_layer, tf.Tensor)
    assert logits_layer.shape.as_list()[-1] == playground.ACTION_CHOICES

    with tf.name_scope(vocab.policy_theta_D) as scope:
        # convert the logits layer (aka: raw output) to probabilities
        log_probabilities = tf.nn.log_softmax(logits_layer)
        oversize_policy_theta = tf.random.categorical(logits_layer, num_samples=1)

        # Remove single-dimensional entries from the shape of the array since we only take one sample from the distribution
        sampled_action = tf.squeeze(oversize_policy_theta, axis=1)

        # # Compute the log probabilitie from sampled action
        # # todo --> sampled_action_log_probability unit test
        # sampled_action_mask = tf.one_hot(sampled_action, depth=playground.ACTION_CHOICES)
        # log_probabilities_matrix = tf.multiply(sampled_action_mask, log_probabilities)
        # sampled_action_log_probability = tf.reduce_sum(log_probabilities_matrix, axis=1)


        return sampled_action, log_probabilities


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
                                                experiment_spec.nn_h_layer_topo, name_scope=vocab.theta_NeuralNet)

        # /---- discrete case -----
        if isinstance(playground.env.action_space, gym.spaces.Discrete):
            assert observation_placeholder.shape.as_list()[-1] == playground.OBSERVATION_SPACE.shape[0], \
                "the observation_placeholder is incompatible with environment, " \
                "{} != {}".format(observation_placeholder.shape.as_list()[-1], playground.OBSERVATION_SPACE.shape[0])

            sampled_action, log_probabilities = policy_theta_discrete_space(theta_mlp, playground)

            pseudo_loss = discrete_pseudo_loss(theta_mlp, action_placeholder, Q_values_placeholder, playground)

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
def reward_to_go(rewards: list) -> np.ndarray:
    assert isinstance(rewards, list)
    np_backward_rewards = np.array(rewards[::-1])
    reward_to_go = np.cumsum(np_backward_rewards)
    return reward_to_go[::-1]

def reward_to_go_np(rewards: np.ndarray) -> np.ndarray:
    assert isinstance(rewards, np.ndarray)
    np_backward_rewards = np.flip(rewards)
    reward_to_go = np.cumsum(np_backward_rewards)
    return np.flip(reward_to_go)


def discounted_reward_to_go(rewards: list, experiment_spec: ExperimentSpec) -> np.ndarray:
    assert isinstance(rewards, list)
    gamma = experiment_spec.discout_factor
    assert (0 <= gamma) and (gamma <= 1)
    backward_rewards = rewards[::-1]
    discounted_reward_to_go = np.zeros_like(rewards)

    for r in range(len(rewards)):
        exp = 0
        for i in range(r, len(rewards)):
            discounted_reward_to_go[i] += gamma**exp * backward_rewards[r]
            exp += 1

    return discounted_reward_to_go[::-1]


def discounted_reward_to_go_np(rewards: np.ndarray, experiment_spec: ExperimentSpec) -> np.ndarray:
    assert rewards.ndim == 1, "Current implementation only support array of rank 1"
    assert isinstance(rewards, np.ndarray)
    gamma = experiment_spec.discout_factor
    assert (0 <= gamma) and (gamma <= 1)

    np_backward_rewards = np.flip(rewards)
    discounted_reward_to_go = np.zeros_like(rewards)
    # todo --> Since flip return a view, test if iterate on a pre flip ndarray and than post flip before return would be cleaner

    # todo --> refactor using a gamma mask and matrix product & sum, instead of loop
    for r in range(len(rewards)):
        exp = 0
        for i in range(r, len(rewards)):
            discounted_reward_to_go[i] += gamma**exp * np_backward_rewards[r]
            exp += 1

    return np.flip(discounted_reward_to_go)
# </editor-fold>


def discrete_pseudo_loss(theta_mlp: tf.Tensor, action_placeholder: tf.Tensor, Q_values_placeholder: tf.Tensor, playground: GymPlayground) -> tf.Tensor:
    """
    Pseudo loss for discrete action space only using Softmax cross entropy with logits

    :param playground:
    :type playground:
    :param theta_mlp:
    :type theta_mlp:
    :param action_placeholder:
    :type action_placeholder: tf.Tensor
    :param Q_values_placeholder:
    :type Q_values_placeholder: tf.Tensor
    :return: pseudo_loss
    :rtype: tf.Tensor
    """

    assert action_placeholder.shape.is_compatible_with(Q_values_placeholder.shape), "action_placeholder shape is not compatible with Q_values_placeholder shape, {} != {}".format(action_placeholder, Q_values_placeholder)

    with tf.name_scope(vocab.pseudo_loss) as scope:

        # Step 1: Compute the log probabilitie of the current policy over the action space
        action_mask = tf.one_hot(action_placeholder, playground.ACTION_CHOICES)
        log_probabilities_matrix = tf.multiply(action_mask, tf.nn.log_softmax(theta_mlp))
        log_probabilities = tf.reduce_sum(log_probabilities_matrix, axis=1)

        # Step 2: Compute the pseudo loss
        # note: tf.stop_gradient(Q_values_placeholder) prevent the backpropagation into the Q_values_placeholder
        #   |   witch contain rewards_to_go. It treat the values of the tensor as constant during backpropagation.

        weighted_likelihoods = tf.multiply(
            tf.stop_gradient(Q_values_placeholder), log_probabilities)
        pseudo_loss = tf.reduce_mean(weighted_likelihoods)
        return pseudo_loss


def policy_optimizer(pseudo_loss: tf.Tensor, exp_spec: ExperimentSpec) -> tf.Operation:
    """
    Define the optimizing methode for training the REINFORE agent

    :param exp_spec:
    :type exp_spec: ExperimentSpec
    :param pseudo_loss:
    :type pseudo_loss: tf.Tensor
    :return: policy_optimizer_op
    :rtype: tf.Operation
    """
    return tf_cv1.train.AdamOptimizer(learning_rate=exp_spec.learning_rate).minimize(pseudo_loss, name=vocab.optimizer)


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
    for ar in arrays_of_values:
        assert isinstance(ar, np.ndarray), "Wrong input type, arrays_of_values must be a list of numpy array"

    feed_dict = dict()
    for placeholder, array in zip(placeholders, arrays_of_values):
        feed_dict[placeholder] = array

    return feed_dict


class TrajectoryContainer(object):
    def __init__(self, observations: list, actions: list, rewards: list, experiment_spec: ExperimentSpec, discounted=True, dtype=None):
        """
        Container for storage & retrieval events collected at every timestep of a single trajectorie

        todo --> validate dtype for discrete case

        Note: about dtype (source: Numpy doc)
         |      "This argument can only be used to 'upcast' the array.
         |          For downcasting, use the .astype(t) method."

        """
        assert len(observations) == len(actions) == len(rewards), "{} vs {} vs {} !!!".format(observations, actions, rewards)
        self.observations = np.array(observations, dtype=dtype)
        self.actions = np.array(actions, dtype=dtype)
        self.rewards = np.array(rewards, dtype=dtype)
        self.discounted = discounted
        self.trajectory_return = np.sum(self.rewards)

        if discounted:
            self.Q_values = discounted_reward_to_go_np(self.rewards, experiment_spec=experiment_spec)
        else:
            self.Q_values = reward_to_go_np(self.rewards)

        assert len(self.Q_values) == len(self.rewards)

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

    def unpack(self) -> (np.ndarray, np.ndarray, np.ndarray, [np.ndarray, np.ndarray], np.float, int):
        """
        Unpack the full trajectorie as a tuple of numpy array

            Note: Q_values is a numpy ndarray view

        :return: (observations, actions, rewards, Q_values, trajectory_return, trajectory_lenght)
        :rtype: (np.ndarray, np.ndarray, np.ndarray, [np.ndarray, np.ndarray], np.float, int)
        """
        return self.observations, self.actions, self.rewards, \
               self.Q_values, self.trajectory_return, self.__len__()

    def __repr__(self):
        str = "\n::trajectory_container/\n"
        str += ".observations=\n{}\n\n".format(self.observations)
        str += ".actions=\n{}\n\n".format(self.actions)
        str += ".rewards=\n{}\n\n".format(self.rewards)
        str += ".discounted --> {}\n".format(self.discounted)
        str += ".Q_values=\n{}\n\n".format(self.Q_values)
        str += ".trajectory_return=\n{}\n\n".format(self.trajectory_return)
        str += "len(trajectory) --> {} ::\n\n".format(self.__len__())
        return str


class TimestepCollector(object):
    """
    Batch collector for time step events agregation
    """
    def __init__(self, experiment_spec: ExperimentSpec, playground: GymPlayground):
        self._exp_spec = experiment_spec
        # self._playground_spec = playground.get_environment_spec()
        self._observations = []
        self._actions = []
        self._rewards = []
        self._q_values = []
        self.step_count = 0

    def __call__(self, observation: np.ndarray, action, reward: float, *args, **kwargs) -> None:
        """ Collect observation, action, reward for one timestep

        :type observation: np.ndarray
        :type action: int or float
        :type reward: float
        """
        try:
            assert self.step_count < self._exp_spec.timestep_max_per_trajectorie, \
                "Max timestep per trajectorie reached so the TimestepCollector is full."
            self._observations.append(observation)
            self._actions.append(action)
            self._rewards.append(reward)
            self.step_count += 1

        except AssertionError as ae:
            raise ae
        return None

    def collect(self, observation: np.ndarray, action, reward: float) -> None:
        """ Collect observation, action, reward for one timestep

        :type observation: np.ndarray
        :type action: int or float
        :type reward: float
        """
        self.__call__(observation, action, reward)
        return None

    def _reset(self):
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self.step_count = 0
        return None

    def get_collected_timestep_and_reset_collector(self, discounted_q_values=True) -> TrajectoryContainer:
        """
            1.  Return the sampled trajectory in a TrajectoryContainer
            2.  Reset the container ready for the next trajectorie sampling.

        :return: A TrajectoryContainer with the full trajectory
        :rtype: TrajectoryContainer
        """
        # todo --> validate dtype for discrete case
        trajectory_container = TrajectoryContainer(self._observations, self._actions, self._rewards, self._exp_spec,
                                                   discounted=discounted_q_values)
        self._reset()
        return trajectory_container

    def __del__(self):
        self._reset()

class EpochContainer(object):
    """Container for storage & retrieval of batch of sampled trajectories"""
    def __init__(self, experiment_spec: ExperimentSpec):
        self.experiment_spec = experiment_spec
        self.trajectories = []
        self.timestep_total = 0
        self._number_of_collected_trajectory = 0

        # note: Optimization consideration --> why collect numpy ndarray in python list?
        #   |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
        #   |       to a ndarray than it is to append ndarray to each other

    def __call__(self, trajectory: TrajectoryContainer,  *args, **kwargs) -> None:
        self.trajectories.append(trajectory)
        self.timestep_total += trajectory.__len__()
        self._number_of_collected_trajectory += 1
        return None

    def collect(self, trajectory: TrajectoryContainer) -> None:
        self.__call__(trajectory)
        return None

    def __len__(self):
        return self._number_of_collected_trajectory

    def total_timestep_collected(self) -> int:
        return self.timestep_total

    def unpack(self) -> (list, list, list, list, list, int, int):
        """
        Unpack the full epoch batch of collected trajectories in lists of numpy ndarray

        todo: (implement & test) precompute ndarray concatenation in a single ndarray
          |        [ndarray, ... , ndarray] --> big ndarray

        :return: (trjs_observations, trjs_actions, trjs_Qvalues,
                    trjs_returns, trjs_lenghts, nb_of_collected_trjs, timestep_total)
        :rtype: (list, list, list, list, list, int, int)
        """

        trjs_observations = []
        trjs_actions = []
        trjs_reward = []
        trjs_Qvalues = []
        trjs_returns = []
        trjs_lenghts = []

        """TrajectoryContainer unpacking reference:
           
                       (observations, actions, rewards, Q_values, 
                                trajectory_return, trajectory_lenght) = TrajectoryContainer.unpack()
        """
        for trj in self.trajectories:
            trajectory = trj.unpack()

            trjs_observations.append(trajectory[0])
            trjs_actions.append(trajectory[1])
            # trjs_reward.append(trajectory[2])
            trjs_Qvalues.append(trajectory[3])
            trjs_returns.append(trajectory[4])
            trjs_lenghts.append(trajectory[5])

        _number_of_collected_trj = self.total_timestep_collected()
        _timestep_total = self.__len__()

        # np.array(V).copy()

        # Reset the container
        self._reset()
        return trjs_observations, trjs_actions, trjs_Qvalues, trjs_returns, trjs_lenghts, _number_of_collected_trj, _timestep_total

    def _reset(self) -> None:
        self.trajectories = []
        self.timestep_total = 0
        self._number_of_collected_trajectory = 0
        return None



class TrajectoriesCollector(object):
    def __init__(self, experiment_spec: ExperimentSpec):
        self.epoch_container = EpochContainer(experiment_spec=experiment_spec)
        
    def collect(self, trajectory: TrajectoryContainer) -> None:
        self.epoch_container.collect(trajectory)
        return None

    def get_collected_trajectories_and_reset_collector(self) -> {list, list, list, list, list, int, int}:
        """
            Step:
                    1. Concatenate each trajectory component in a long ndarray
                    2. Compute the relevant metric
                    3. Return the computed value in a dictionarie
                    4. Reset the container ready for the next batch.

        :key: 'trjs_obss', 'trjs_acts', 'trjs_Qvalues', 'trjs_returns', 'trjs_len', 'epoch_average_return', 'epoch_average_lenghts'

        :return: a dictionarie of concatenated trajectories component
        :rtype: dict
        """
        trajectories_lists = self.epoch_container.unpack()
        _, _, _, _, _, nb_trjs, timestep_total = trajectories_lists
        component_name = ('trjs_obss', 'trjs_acts', 'trjs_Qvalues', 'trjs_returns', 'trjs_len')

        """EpochContainer unpacking reference:
        
                (trjs_observations, trjs_actions, trjs_Qvalues, trjs_returns, 
                            trjs_lenghts, nb_of_collected_trjs, timestep_total) = epoch_container.unpack()
        """

        # For each component, convert his list of trajectories to a long ndarray of trajectories:
        trajectories_dict = {K: np.array(V).copy() for K, V in zip(component_name, trajectories_lists[:-2])}

        # Compute metric:
        trajectories_dict['epoch_average_return'] = np.mean(trajectories_dict['trjs_returns'])
        trajectories_dict['epoch_average_lenghts'] = np.mean(trajectories_dict['trjs_len'])

        return trajectories_dict

    def get_number_of_trajectories_collected(self):
        return self.epoch_container.__len__()

    def get_total_timestep_collected(self) -> int:
        return self.epoch_container.total_timestep_collected()



def format_single_step_observation(observation: np.ndarray):
    """ Single trajectorie batch size hack for the computation graph observation placeholder
                observation.shape = (8,)
            vs
                np.expand_dims(observation, axis=0).shape = (1, 8)
    """
    assert observation.ndim == 1, "Watch out!! observation array is of wrong dimension {}".format(observation.shape)
    batch_size_one_observation = np.expand_dims(observation, axis=0)
    return batch_size_one_observation


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
        assert isinstance(action, int), "something is wrong with the 'format_single_step_action'. " \
                                        "Should output a int instead of {}".format(action)
    finally:
        return action