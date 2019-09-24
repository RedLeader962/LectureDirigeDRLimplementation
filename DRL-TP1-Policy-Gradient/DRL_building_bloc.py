# coding=utf-8
import sys
import time

import matplotlib.pyplot as plt
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
    def __init__(self, timestep_max_per_trajectorie=None, trajectories_batch_size=10, max_epoch=2, discout_factor=1,
                 learning_rate=1e-2, neural_net_hidden_layer_topology: tuple = (32, 32), random_seed=42,
                 discounted_reward_to_go=True, environment_name='CartPole-v1', print_metric_every_what_epoch=5):
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

        self.timestep_max_per_trajectorie = timestep_max_per_trajectorie

        self.trajectories_batch_size = trajectories_batch_size
        self.max_epoch = max_epoch
        self.discout_factor: float = discout_factor
        self.learning_rate = learning_rate
        self.discounted_reward_to_go = discounted_reward_to_go

        self.nn_h_layer_topo = neural_net_hidden_layer_topology
        self.random_seed = random_seed
        self.hidden_layers_activation: tf.Tensor = tf.tanh
        self.output_layers_activation: tf.Tensor = tf.sigmoid

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
        :return: ( trajectories_batch_size, max_epoch, timestep_max_per_trajectorie )
        :rtype: (int, int, int)
        """
        return {
            'timestep_max_per_trajectorie': self.timestep_max_per_trajectorie,
            'trajectories_batch_size': self.trajectories_batch_size,
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

    todo --> refactor using a gamma mask and matrix product & sum, instead of loop

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

    # (Iceboxed) todo:investigate?? --> it's probably not required anymore:
    # for ar in arrays_of_values:
    #     assert isinstance(ar, np.ndarray), "Wrong input type, arrays_of_values must be a list of numpy array"

    feed_dict = dict()
    for placeholder, array in zip(placeholders, arrays_of_values):
        feed_dict[placeholder] = array

    return feed_dict


class TrajectoryContainer(object):
    def __init__(self, observations: list, actions: list, rewards: list, experiment_spec: ExperimentSpec,
                 discounted: bool = True) -> None:
        """
        Container for storage & retrieval events collected at every timestep of a single trajectorie

        todo --> validate dtype for discrete case

        Note: about dtype (source: Numpy doc)
         |      "This argument can only be used to 'upcast' the array.
         |          For downcasting, use the .astype(t) method."

         :param observations:
         :type observations: list
         :param actions:
         :type actions: list
         :param rewards:
         :type rewards: list
         :param experiment_spec:
         :type experiment_spec: ExperimentSpec
         :param discounted:
         :type discounted: bool

        """
        assert isinstance(observations, list) and isinstance(actions, list) and isinstance(rewards, list), "wrong argument type"
        assert len(observations) == len(actions) == len(rewards), "{} vs {} vs {} !!!".format(observations, actions, rewards)
        # self.observations = np.array(observations, dtype=dtype)
        # self.actions = np.array(actions, dtype=dtype)
        # self.rewards = np.array(rewards, dtype=dtype)
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.the_trajectory_return = np.sum(self.rewards, axis=None)
        self.discounted = discounted

        if discounted:
            self.Q_values = discounted_reward_to_go(self.rewards, experiment_spec=experiment_spec)
        else:
            self.Q_values = reward_to_go(self.rewards)

        assert len(self.Q_values) == len(self.rewards)

    def __len__(self):
        return len(self.actions)

    # def __getitem__(self, timestep: int):
    #     """
    #     Return the values (observation, action, reward) at the given timestep
    #
    #     :param timestep: the exact timestep number (not the list index)
    #     :type timestep: int
    #     :return:  (observations, actions, rewards) at timestep t
    #     :rtype: (np.ndarray, np.ndarray, np.ndarray)
    #     """
    #     ts = timestep - 1
    #     obs = self.observations[ts]
    #     act = self.actions[ts]
    #     rew = self.rewards[ts]
    #     return obs, act, rew

    def unpack(self) -> (list, list, list, list, float, int):
        """
        Unpack the full trajectorie as a tuple of numpy array

            Note: Q_values is a numpy ndarray view

        :return: (observations, actions, rewards, Q_values, the_trajectory_return, trajectory_lenght)
        :rtype: (list, list, list, list, int, int)
        """
        tc = self.observations, self.actions, self.rewards, self.Q_values, int(self.the_trajectory_return), self.__len__()
        return tc

    def __repr__(self):
        myRep = "\n::trajectory_container/\n"
        myRep += ".observations=\n{}\n\n".format(self.observations)
        myRep += ".actions=\n{}\n\n".format(self.actions)
        myRep += ".rewards=\n{}\n\n".format(self.rewards)
        myRep += ".discounted --> {}\n".format(self.discounted)
        myRep += ".Q_values=\n{}\n\n".format(self.Q_values)
        myRep += ".the_trajectory_return=\n{}\n\n".format(self.the_trajectory_return)
        myRep += "len(trajectory) --> {} ::\n\n".format(self.__len__())
        return myRep


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
        trajectory_container = TrajectoryContainer(self._observations.copy(), self._actions.copy(),
                                                   self._rewards.copy(), self._exp_spec, discounted=discounted_q_values)

        self._reset()
        return trajectory_container

    def __del__(self):
        self._reset()


class EpochContainer(object):
    def __init__(self, trajectories_list: list):
        """
        Container for storage & retrieval of a batch of sampled trajectories

        :param trajectories_list: a list of TrajectoryContainer instance fulled with collected timestep events
        :type trajectories_list: [TrajectoryContainer, ...]
        """
        self.trjs_observations = []
        self.trjs_actions = []
        self.trjs_reward = []
        self.trjs_Qvalues = []
        self.trjs_returns = []
        self.trjs_lenghts = []
        self._number_of_collected_trajectory = len(trajectories_list)
        self._total_timestep_collected = 0

        for aTrajectory_container in trajectories_list:
            assert isinstance(aTrajectory_container, TrajectoryContainer), \
                "The list must contain object of type TrajectoryContainer"

            unpacked = aTrajectory_container.unpack()
            observations, actions, rewards, Q_values, trajectory_return, trajectory_lenght = unpacked

            self.trjs_observations += observations
            self.trjs_actions += actions
            self.trjs_reward += rewards
            self.trjs_Qvalues += Q_values
            self.trjs_returns.append(trajectory_return)
            self.trjs_lenghts.append(trajectory_lenght)
            self._total_timestep_collected += len(aTrajectory_container)

    def __len__(self) -> int:
        return self._number_of_collected_trajectory

    def get_total_timestep_collected(self):
        return self._total_timestep_collected

    def compute_metric(self) -> (float, float):
        """
        Compute relevant metric over this container stored sample

        :return: (epoch_average_trjs_return, epoch_average_trjs_lenght)
        :rtype: (float, float)
        """
        epoch_average_trjs_return = np.mean(self.trjs_returns).copy()
        epoch_average_trjs_lenght = np.mean(self.trjs_lenghts).copy()
        return epoch_average_trjs_return, epoch_average_trjs_lenght

    def unpack_all(self) -> (list, list, list, list, list, int, int):
        """
        Unpack the full epoch batch of collected trajectories in lists of numpy ndarray

        :return: (trjs_observations, trjs_actions, trjs_Qvalues,
                    trjs_returns, trjs_lenghts, total_timestep_collected, nb_of_collected_trjs )
        :rtype: (list, list, list, list, list, int, int)
        """

        # (icebox) todo:assessment --> if the copy method still required?:
        #                                   it does only if the list content are numpy ndarray
        trajectories_copy = (self.trjs_observations.copy(), self.trjs_actions.copy(), self.trjs_Qvalues.copy(),
                             self.trjs_returns.copy(), self.trjs_lenghts, self._total_timestep_collected, self.__len__())

        return trajectories_copy


class TrajectoriesCollector(object):
    def __init__(self):
        self.trajectories_list = []
        self.timestep_total = 0
        self._number_of_collected_trajectory = 0

        # note: Optimization consideration --> why collect numpy ndarray in python list?
        #   |   It's a order of magnitude faster to collect ndarray in a list and then convert the list
        #   |       to a ndarray than it is to append ndarray to each other

    def __call__(self, trajectory: TrajectoryContainer, *args, **kwargs) -> None:
        self.trajectories_list.append(trajectory)
        self.timestep_total += trajectory.__len__()
        self._number_of_collected_trajectory += 1
        return None

    def collect(self, trajectory: TrajectoryContainer) -> None:
        self.__call__(trajectory)
        return None

    def get_number_of_trajectories_collected(self):
        return self.trajectories_list.__len__()

    def get_total_timestep_collected(self) -> int:
        return self.timestep_total

    def get_collected_trajectories_and_reset_collector(self) -> EpochContainer:
        """
            Step:
                    1. Concatenate each trajectory component in a long ndarray
                    2. Compute the relevant metric
                    3. Return the computed value in a dictionarie
                    4. Reset the container ready for the next batch.


        :return: a EpochContainer of concatenated trajectories component
        :rtype: EpochContainer
        """
        container = EpochContainer(self.trajectories_list)

        self.trajectories_list = []
        self.timestep_total = 0
        self._number_of_collected_trajectory = 0

        return container


# todo:validate --> possible source of graph data input error:
def format_single_step_observation(observation: np.ndarray):
    """ Single trajectorie batch size hack for the computation graph observation placeholder
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
        assert isinstance(action, int), "something is wrong with the 'format_single_step_action'. " \
                                        "Should output a int instead of {}".format(action)
    finally:
        return action


class CycleIndexer(object):
    def __init__(self, cycle_lenght: int = 10):
        self.i = 0
        self.j = cycle_lenght
        self.cycle_lenght = cycle_lenght

    def __next__(self):
        if self.i == self.cycle_lenght:
            self.reset()
        else:
            self.i += 1
            self.j -= 1

        return self.i, self.j

    def reset(self):
        self.i = 0
        self.j = self.cycle_lenght


class ConsolPrintLearningStats(object):
    def __init__(self, print_metric_every_what_epoch=5, consol_span=90):
        self.cycle_indexer = CycleIndexer(cycle_lenght=10)
        self.epoch = 0
        self.trj = 0
        self.the_trajectory_return = None
        self.timestep = None
        self.number_of_trj_collected = None
        self.total_timestep_collected = None
        self.epoch_loss = None
        self.average_trjs_return = None
        self.average_trjs_lenght = None
        self.print_metric_every = print_metric_every_what_epoch
        self.span = consol_span

        self.current_stats_batch_pseudo_loss = 0.0
        self.last_stats_batch_mean_pseudo_lost = 0.0

        self.current_batch_return = 0
        self.last_batch_return = 0

        self.collected_experiment_stats = {
            'smoothed_average_return': [],
            'smoothed_average_peusdo_loss': [],
        }

    def _assert_all_property_are_feed(self) -> bool:
        if ((self.number_of_trj_collected is not None) and (self.total_timestep_collected is not None) and
                (self.epoch_loss is not None) and (self.average_trjs_return is not None) and
                (self.average_trjs_lenght is not None)):
            return True

    def start_the_crazy_experiment(self, message=("3", "2", "1", "READY")) -> None:
        print("\n\n")
        self.anim_line(start_anim_at_a_new_line=True, keep_cursor_at_same_line_on_exit=False)
        self.anim_line(nb_of_cycle=1, keep_cursor_at_same_line_on_exit=True)

        for m in message:
            print("\r{:^{span}}".format(m, span=self.span), end="", flush=True)
            time.sleep(0.2)
        # print("\r{:^{span}}".format("?", span=self.span), end="", flush=True)
        # time.sleep(0.01)
        print(
            "\r{:=<{span}}\r".format("=== EXPERIMENT START ", span=self.span), end="", flush=True)
        return None

    def print_experiment_stats(self):
        print("\n\n\n{:^{span}}".format("Experiment stoped", span=self.span))
        stats_str = "Collected {} trajectories over {} epoch".format(self.trj, self.epoch)
        print("{:^{span}}".format(
            stats_str, span=self.span), end="\n\n", flush=True)
        print("{:=>{span}}".format(" EXPERIMENT END ===", span=self.span), end="\n", flush=True)
        self.anim_line(caracter=">", nb_of_cycle=1, start_anim_at_a_new_line=False)
        self.anim_line(caracter="<", nb_of_cycle=1, keep_cursor_at_same_line_on_exit=False)

        ultra_basic_ploter(self.collected_experiment_stats['smoothed_average_return'], self.collected_experiment_stats['smoothed_average_peusdo_loss'],)
        # print("\n\nCollected experiment stats:\n{}".format(self.collected_experiment_stats))
        return None

    def anim_line(self, caracter=">", nb_of_cycle=2,
                  start_anim_at_a_new_line=False,
                  keep_cursor_at_same_line_on_exit=True):
        if start_anim_at_a_new_line:
            print("\n")

        for c in range(nb_of_cycle):
            for i in range(self.span):
                print(caracter, end="", flush=True)
                time.sleep(0.005)

            if (c == nb_of_cycle -1) and not keep_cursor_at_same_line_on_exit:
                print("\n", end="", flush=True)
            else:
                print("\r", end="", flush=True)

    def next_glorious_epoch(self) -> None:
        self.epoch += 1

        if (self.epoch - 1) % self.print_metric_every == 0:
            print("\n\n{:-<{span}}\n".format(":: Epoch ", span=self.span), end="", flush=True)
        return None

    def next_glorious_trajectory(self) -> (int, int):
        """
        Incremente the cycle_index_i and decremente the cycle_index_j.
        Both index are returned for convience.

        :return: cycle_index_i, cycle_index_j
        :rtype: int, int
        """
        self.trj += 1
        return self.cycle_indexer.__next__()

    def epoch_training_stat(self, epoch_loss, epoch_average_trjs_return, epoch_average_trjs_lenght,
                            number_of_trj_collected, total_timestep_collected) -> None:
        """
        Call after a traing update as been done, at the end of a epoch.
        """
        self.number_of_trj_collected = number_of_trj_collected
        self.total_timestep_collected = total_timestep_collected
        self.epoch_loss = epoch_loss
        self.average_trjs_return = epoch_average_trjs_return
        self.average_trjs_lenght = epoch_average_trjs_lenght

        self.current_stats_batch_pseudo_loss += self.epoch_loss

        self.current_batch_return += epoch_average_trjs_return



        if (self.epoch) % self.print_metric_every == 0:
            mean_stats_batch_loss = self.current_stats_batch_pseudo_loss / self.print_metric_every
            mean_stats_batch_return = self.current_batch_return / self.print_metric_every
            print(
                "\r\t ↳ {:^3}".format(self.epoch),
                ":: Collected {} trajectories for a total of {} timestep.".format(
                    self.number_of_trj_collected, self.total_timestep_collected),
                "\n\t\t↳ pseudo loss: {:>6.2f} ".format(self.epoch_loss),
                "| average trj return: {:>6.2f} | average trj lenght: {:>6.2f}".format(
                    self.average_trjs_return, self.average_trjs_lenght),
                end="\n", flush=True)

            print("\n\tAverage pseudo lost: {:>6.3f} (over the past {} epoch)".format(
                mean_stats_batch_loss, self.print_metric_every))
            if abs(mean_stats_batch_loss) < abs(self.last_stats_batch_mean_pseudo_lost):
                print("\t\t↳ is lowering ⬊  ...  goooood :)", end="", flush=True)
            elif mean_stats_batch_loss > self.last_stats_batch_mean_pseudo_lost:
                print("\t\t↳ is rising ⬈", end="", flush=True)

            self.collected_experiment_stats['smoothed_average_peusdo_loss'].append(mean_stats_batch_loss)
            self.collected_experiment_stats['smoothed_average_return'].append(mean_stats_batch_return)

            self.current_stats_batch_pseudo_loss = 0
            self.last_stats_batch_mean_pseudo_lost = mean_stats_batch_loss

            self.current_batch_return = 0
            self.last_batch_return = mean_stats_batch_return
        return None

    def trajectory_training_stat(self, the_trajectory_return, timestep) -> None:
        """
        Print formated learning metric & stat

        :param the_trajectory_return:
        :type the_trajectory_return: float
        :param timestep:
        :type timestep: int
        :return:
        :rtype: None
        """
        print("\r\t ↳ {:^3} :: Trajectory {:>4}  ".format(self.epoch, self.trj),
              ">"*self.cycle_indexer.i, " "*self.cycle_indexer.j,
              "  got return {:>8.2f}   after  {:>4}  timesteps".format(
                  the_trajectory_return, timestep + 1),
              sep='', end='', flush=True)

        self.the_trajectory_return = the_trajectory_return
        self.timestep = timestep
        return None




def ultra_basic_ploter(epoch_average_return: list, epoch_average_loss: list) -> None:

    fig, ax = plt.subplots(figsize=(8, 6))

    x_axes = len(epoch_average_return)
    ax.plot(x_axes, epoch_average_return, label='Average Return')
    ax.plot(x_axes, epoch_average_loss, label='Average loss')

    # plt.ylabel('Average Return')
    plt.xlabel('Epoch')

    ax.grid(True)
    ax.legend(loc='best')

    plt.show()