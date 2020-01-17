# coding=utf-8
from datetime import datetime
from typing import Any, Union, Tuple
import gym
from gym.wrappers import TimeLimit
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
from tensorflow.python import keras
import numpy as np

from blocAndTools.rl_vocabulary import rl_name

deprecation._PRINT_DEPRECATION_WARNINGS = False
vocab = rl_name()
tf_cv1 = tf.compat.v1  # shortcut


class ExperimentSpec:
    
    def __init__(self, algo_name=None, comment=None, batch_size_in_ts=5000, max_epoch=2, discout_factor=0.99,
                 learning_rate=1e-2, theta_nn_hidden_layer_topology: tuple = (32, 32), random_seed=0,
                 discounted_reward_to_go=True, environment_name='CartPole-v1', expected_reward_goal=None,
                 print_metric_every_what_epoch=5, isTestRun=False, show_plot=False, log_metric_interval=100):
        """
        Gather the specification for a experiement regarding NN and algo training hparam plus some environment detail
        
        note:
          |     EPOCH definition:
          |         In our casse, collecting and updating the gradient of a set of trajectories of
          |         size=batch_size_in_ts is equal to one EPOCH

        """
        # todo: add a param for the neural net configuration via a dict fed as a argument
        # (nice to have) todo:implement --> set_experiment_spec_JSON (taking json as argument):

        self.algo_name = algo_name
        self.comment = comment
        self.paramameter_set_name = 'default'
        self.rerun_tag = None
        self.rerun_idx = 0

        self.isTestRun = isTestRun
        self.prefered_environment = environment_name
        self.expected_reward_goal = expected_reward_goal
        self.show_plot = show_plot

        self.batch_size_in_ts = batch_size_in_ts
        self.max_epoch = max_epoch
        self.discout_factor: float = discout_factor
        self.learning_rate = learning_rate
        self.discounted_reward_to_go = discounted_reward_to_go

        self.theta_nn_h_layer_topo = theta_nn_hidden_layer_topology
        self.random_seed = random_seed
        self.theta_hidden_layers_activation: tf.Tensor = tf.nn.tanh
        self.theta_output_layers_activation: tf.Tensor = None

        self.render_env_every_What_epoch = 100
        self.log_metric_interval = log_metric_interval
        self.print_metric_every_what_epoch = print_metric_every_what_epoch

        self._assert_param()
    
    def _assert_param(self):
        
        if isinstance(self.discout_factor, list):
            for each in self.discout_factor:
                assert (0 <= each) and (each <= 1)
        else:
            assert (0 <= self.discout_factor) and (self.discout_factor <= 1)
        
        if isinstance(self.theta_nn_h_layer_topo, list):
            for each in self.theta_nn_h_layer_topo:
                assert isinstance(each, tuple)
        else:
            assert isinstance(self.theta_nn_h_layer_topo, tuple)
    
    def __getitem__(self, item: str):
        """
        Use ExperimentSpecification instance like a dictionary
        It's the way to access new spec added via set_experiment_spec()

        ```python

            exp_spec.set_experiment_spec({'critique_loop_len': 80})
            exp_spec['critique_loop_len']
            > 80

        ```
        :param item: a specification keyword
        :return: the specification value
        """
        return self.__dict__[item]
    
    def set_experiment_spec(self, dict_param: dict, print_change=True) -> None:
        """
        Change any spec value and/or append aditional spec with value

        :param print_change:
        :type print_change:
        :param dict_param: A dictionary of spec: value
        :type dict_param: dict
        """
        for str_k, v in dict_param.items():
            str_k: str
            self.__setattr__(str_k, v)

        self._assert_param()

        if print_change:
            print("\n\n:: Switching to parameter: {}\n".format(self.paramameter_set_name))
            print(self.__repr__())
        return None
    
    def __repr__(self):
        class_name = "ExperimentSpec"
        repr_str = data_container_class_representation(self, class_name, space_from_margin=3)
        return repr_str
    
    def get_agent_training_spec(self):
        """
        Utility fct: Return specification related to the agent training
        (!) is non exhaustive.
        :return: ( batch_size_in_ts, max_epoch, timestep_max_per_trajectorie )
        :rtype: (int, int, int)
        """
        # (Ice-Boxed) todo:assessment --> is it still usefull?: remove if not
        return {
            'batch_size_in_ts': self.batch_size_in_ts,
            'max_epoch':        self.max_epoch,
            'discout_factor':   self.discout_factor,
            'learning_rate':    self.learning_rate,
            'isTestRun':        self.isTestRun
            }
    
    def get_neural_net_spec(self):
        """
        Utility fct: Return the specification related to the neural net construction
        (!) is non exhaustive.
        """
        # (Ice-Boxed) todo:assessment --> is it still usefull?: remove if not
        return {
            'theta_nn_h_layer_topo':          self.theta_nn_h_layer_topo,
            'random_seed':                    self.random_seed,
            'theta_hidden_layers_activation': self.theta_hidden_layers_activation,
            'theta_output_layers_activation': self.theta_output_layers_activation,
            }


def data_container_class_representation(class_instance, class_name: str, space_from_margin=0) -> str:
    """
    Utility function for automatic representation of container type class
    Handle dynamically property added at run time
    
    .. Example::
    
        def __repr__(self):
            repr_str = data_container_class_representation(self, class_name='ExperimentSpec', space_from_margin=3)
            return repr_str
        
    
    :param class_instance:
    :param class_name:
    :param space_from_margin:
    :return:
    """
    m_sp = " " * space_from_margin
    item_space = " " * 3
    repr_str = m_sp + class_name + "{\n"
    for k, v in class_instance.__dict__.items():
        repr_str += m_sp + item_space + "\'{}\': {}\n".format(k, v)
    repr_str += m_sp + "}"
    return repr_str


def list_representation(list_instance: list, list_name: str):
    tr_str = "%s\n[" % list_name
    for trainable in list_instance:
        tr_str += '\t' + str(trainable) + '\n'
    tr_str += ']\n'
    return tr_str


class GymPlayground(object):
    
    def __init__(self, environment_name='LunarLanderContinuous-v2', print_env_info=False):
        """
        Setup the learning playground for the agent (the environment in witch he will play) and gather relevant spec

        Note: 'LunarLanderContinuous-v2' (DEFAUT continuous environment)
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

        self._env = self._make_gym_env()

        info_str = ""
        if isinstance(self._env.action_space, gym.spaces.Box):
            info_str += "\n\n:: Action space is Contiuous\n"
            self.ACTION_SPACE = self._env.action_space
            dimension = self.ACTION_SPACE.shape
            self.ACTION_CHOICES = [*dimension][-1]
        else:
            info_str += "\n\n:: Action space is Discrete\n"
            self.ACTION_SPACE = self._env.action_space
            self.ACTION_CHOICES = self.ACTION_SPACE.n

        if isinstance(self._env.observation_space, gym.spaces.Box):
            self.OBSERVATION_SPACE = self._env.observation_space
            obs_dimension = self.OBSERVATION_SPACE.shape
            self.OBSERVATION_DIM = [*obs_dimension][-1]
        else:
            raise NotImplementedError("GymPlayground does not support non continuous observation space yet!")

        # (nice to have) todo:fixme!! --> update folder path:
        # if print_env_info:
        #     if self.ENVIRONMENT_NAME == 'LunarLanderContinuous-v2':
        #         action_space_doc = "\tAction is two floats [main engine, left-right engines].\n" \
        #                        "\tMain engine: -1..0 off, 0..+1 throttle from 50% to 100% power.\n" \
        #                        "\t\t\t\t(!) Engine can't work with less than 50% power.\n" \
        #                        "\tLeft-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine,
        #                        -0.5..0.5 off\n\n"
        #         info_str += env_spec_pretty_printing.environnement_doc_str(self._env,
        #         action_space_doc=action_space_doc)
        #     else:
        #         info_str += env_spec_pretty_printing.environnement_doc_str(self._env)

        print(info_str)
    
    def _make_gym_env(self):
        try:
            return gym.make(self.ENVIRONMENT_NAME)
        except gym.error.Error as e:
            raise gym.error.Error("GymPlayground did not find the specified Gym environment.") from e
    
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


def build_MLP_computation_graph(input_placeholder: tf.Tensor, output_dim, hidden_layer_topology: tuple = (32, 32),
                                hidden_layers_activation: tf.Tensor = tf.nn.tanh,
                                output_layers_activation: tf.Tensor = None,
                                reuse=None,
                                name=vocab.Multi_Layer_Perceptron) -> tf.Tensor:
    """
    Builder function for Low Level TensorFlow API.
    Return a Multi Layer Perceptron computatin graph with topology:

        input_placeholder | *hidden_layer_topology | logits_layer

    The last layer is called the 'logits' (aka: the raw output of the MLP)

    In the context of deep learning, 'logits' is the equivalent of 'raw output' of our prediction.
    It will later be transform into probabilities using the 'softmax function'


    :return: a well construct computation graph
    :rtype: tf.Tensor
    """
    assert isinstance(input_placeholder, tf.Tensor)
    assert isinstance(hidden_layer_topology, tuple)

    with tf_cv1.variable_scope(name_or_scope=name, reuse=reuse):
        h_layer = input_placeholder

        # # (!) the kernel_initializer random initializer choice make a big difference on the learning performance
        kernel_init = tf_cv1.initializers.he_normal
        # kernel_init = None

        # create & connect all hidden layer
        for l_id in range(len(hidden_layer_topology)):
            h_layer = tf_cv1.layers.dense(h_layer, hidden_layer_topology[l_id],
                                          activation=hidden_layers_activation,
                                          reuse=reuse,
                                          kernel_initializer=kernel_init(),
                                          name='{}{}'.format(vocab.hidden_, l_id + 1))

        logits = tf_cv1.layers.dense(h_layer, output_dim,
                                     activation=output_layers_activation,
                                     reuse=reuse,
                                     kernel_initializer=kernel_init(),
                                     name=vocab.logits)

    return logits


def build_KERAS_MLP_computation_graph(input_placeholder: tf.Tensor, output_dim, hidden_layer_topology: tuple = (32, 32),
                                      hidden_layers_activation: tf.Tensor = tf.nn.tanh,
                                      output_layers_activation: tf.Tensor = None,
                                      reuse=None,
                                      name='keras_' + vocab.Multi_Layer_Perceptron) -> tf.Tensor:
    """
    Builder function for KERAS TensorFlow API.
    
    Note: Be advise, the argument 'reuse' is kept for signature compatibility purpose.
      |     Keras use a different strategy for reuse than regular tensorflow Dense layer.
    
    Return a Multi Layer Perceptron computation graph with topology:

        input_placeholder | *hidden_layer_topology | logits_layer

    The last layer is called the 'logits' (aka: the raw output of the MLP)

    In the context of deep learning, 'logits' is the equivalent of 'raw output' of our prediction.
    It will later be transform into probabilities using the 'softmax function'


    :return: a well construct computation graph
    :rtype: tf.Tensor
    """
    error_msg = ("\t:: Be advise that the argument 'reuse' was keept only for signature compatibility "
                 "purpose with 'buildingbloc.build_MLP_computation_graph()'.\n"
                 "\t\t\t\t\t   Keras implementation of 'Dense' layers use a different strategy to handle 'reuse' task "
                 "than regular tensorflow 'Dense' layer")
    assert reuse is None, error_msg
    assert isinstance(input_placeholder, tf.Tensor)
    assert isinstance(hidden_layer_topology, tuple)

    with tf_cv1.variable_scope(name_or_scope=name, reuse=reuse):
        h_layer = input_placeholder

        # # (!) the kernel_initializer random initializer choice make a big difference on the learning performance
        kernel_init = tf_cv1.initializers.he_normal
        # kernel_init = None
        
        # create & connect all hidden layer
        for l_id in range(len(hidden_layer_topology)):
            dense = keras.layers.Dense(hidden_layer_topology[l_id],
                                       activation=hidden_layers_activation,
                                       kernel_initializer=kernel_init(),
                                       name='{}{}'.format(vocab.hidden_, l_id + 1))
            h_layer = dense(h_layer)
        
        logits = keras.layers.Dense(output_dim,
                                    activation=output_layers_activation,
                                    kernel_initializer=kernel_init(),
                                    name=vocab.logits)
    
    return logits(h_layer)


def continuous_space_placeholder(space: gym.spaces.Box, shape_constraint: tuple = None, name=None) -> tf.Tensor:
    assert isinstance(space, gym.spaces.Box)
    space_shape = space.shape
    if shape_constraint is not None:
        shape = (*shape_constraint, *space_shape)
    else:
        shape = (None, *space_shape)
    return tf_cv1.placeholder(dtype=tf.float32, shape=shape, name=name)


def discrete_space_placeholder(space: gym.spaces.Discrete, shape_constraint: tuple = None, dtype=tf.int32,
                               name=None) -> tf.Tensor:
    assert isinstance(space, gym.spaces.Discrete), "{}".format(space)
    if shape_constraint is not None:
        shape = (*shape_constraint,)
    else:
        shape = (None,)
    
    return tf_cv1.placeholder(dtype=dtype, shape=shape, name=name)


def gym_playground_to_tensorflow_graph_adapter(playground: GymPlayground, obs_shape_constraint: tuple = None,
                                               action_shape_constraint: tuple = None,
                                               Q_name: str = vocab.Qvalues_ph,
                                               obs_ph_name=vocab.obs_ph
                                               ) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    Configure placeholder for feeding value to the computation graph
            Continuous space    -->     dtype=tf.float32
            Discrete scpace     -->     dtype=tf.int32

    :return: obs_ph, act_ph, Q_values_placeholder
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
    """
    assert isinstance(playground, GymPlayground), "\n\n>>> Expected a builded GymPlayground.\n\n"

    obs_ph = build_observation_placeholder(playground, obs_shape_constraint, obs_ph_name)
    
    act_ph = build_action_placeholder(playground, action_shape_constraint)
    
    if obs_shape_constraint is not None:
        shape = (*obs_shape_constraint,)
    else:
        shape = (None,)
    
    Q_values_ph = tf_cv1.placeholder(dtype=tf.float32, shape=shape, name=Q_name)
    
    return obs_ph, act_ph, Q_values_ph


def build_action_placeholder(playground: GymPlayground, action_shape_constraint: tuple = None,
                             name: str = vocab.act_ph):
    assert isinstance(playground, GymPlayground), "\n\n>>> Expected a builded GymPlayground.\n\n"
    
    if isinstance(playground.env.action_space, gym.spaces.Box):
        """action space is continuous"""
        act_ph = continuous_space_placeholder(playground.ACTION_SPACE, action_shape_constraint, name=name)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """action space is discrete"""
        act_ph = discrete_space_placeholder(playground.ACTION_SPACE, action_shape_constraint, name=name)
    else:
        raise NotImplementedError
    return act_ph


def build_observation_placeholder(playground: GymPlayground, obs_shape_constraint: tuple = None,
                                  name: str = vocab.obs_ph):
    assert isinstance(playground, GymPlayground), "\n\n>>> Expected a builded GymPlayground.\n\n"
    
    if isinstance(playground.env.observation_space, gym.spaces.Box):
        """observation space is continuous"""
        obs_ph = continuous_space_placeholder(playground.OBSERVATION_SPACE, obs_shape_constraint, name=name)
    elif isinstance(playground.env.action_space, gym.spaces.Discrete):
        """observation space is discrete"""
        obs_ph = discrete_space_placeholder(playground.OBSERVATION_SPACE, obs_shape_constraint, dtype=tf_cv1.float32,
                                            name=name)
    else:
        raise NotImplementedError
    return obs_ph


def policy_theta_discrete_space(logits_layer: tf.Tensor, playground: GymPlayground, name=vocab.policy_theta_D) -> (
        tf.Tensor, tf.Tensor):
    """Policy theta for discrete space --> actions are sampled from a categorical distribution

    :param logits_layer:
    :type logits_layer: tf.Tensor
    :param playground:
    :type playground: GymPlayground
    :param name:
    :type name:
    :return: (sampled_action, log_p_all)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    assert isinstance(playground.env.action_space, gym.spaces.Discrete)
    assert isinstance(logits_layer, tf.Tensor)
    assert logits_layer.shape.as_list()[-1] == playground.ACTION_CHOICES
    
    with tf.name_scope(name=name) as scope:
        # convert the logits layer (aka: raw output) to probabilities
        log_p_all = tf.nn.log_softmax(logits_layer)
        oversize_policy_theta = tf.random.categorical(logits_layer, num_samples=1)
        
        # Remove single-dimensional entries from the shape of the array since we only take one sample from the
        # distribution
        sampled_action = tf.squeeze(oversize_policy_theta, axis=1, )
        
        # (Ice-Boxed) todo:implement --> sampled_action_log_probability unit test:
        # # Compute the log probabilitie from sampled action
        # sampled_action_mask = tf.one_hot(sampled_action, depth=playground.ACTION_CHOICES)
        # log_probabilities_matrix = tf.multiply(sampled_action_mask, log_p_all)
        # sampled_action_log_probability = tf.reduce_sum(log_probabilities_matrix, axis=1)
        
        return sampled_action, log_p_all


# (Ice-Boxed) todo:implement --> implement policy_theta for continuous space: ice-boxed until next sprint
def policy_theta_continuous_space(logits_layer: tf.Tensor, playground: GymPlayground, name=vocab.policy_theta_C):
    """
    Policy theta for continuous space --> actions are sampled from a gausian distribution
    status: ice-box until next sprint
    """
    assert isinstance(playground.env.action_space, gym.spaces.Box)
    assert isinstance(logits_layer, tf.Tensor)
    assert logits_layer.shape.as_list()[-1] == playground.ACTION_CHOICES

    with tf.name_scope(name=name) as scope:
        # convert the logits layer (aka: raw output) to probabilities
        logits_layer = tf.identity(logits_layer, name='mu')

        raise NotImplementedError  # todo: implement
        # log_standard_deviation = NotImplemented  # (!) todo
        # standard_deviation = NotImplemented  # (!) todo --> compute standard_deviation
        # logit_layer_shape = tf.shape(logits_layer)
        # sampled_action = logits_layer + tf.random_normal(logit_layer_shape) * standard_deviation
        # return sampled_action, log_standard_deviation


def discrete_pseudo_loss(log_p_all, action_placeholder: tf.Tensor, Q_values_placeholder: tf.Tensor,
                         playground: GymPlayground, name=None) -> tf.Tensor:
    """
    Pseudo loss for discrete action space
    """
    with tf.name_scope(name) as scope:
        # Step 1: Compute the log probabilitie of the current policy over the action space
        action_mask = tf.one_hot(action_placeholder, playground.ACTION_CHOICES)
        log_probabilities_matrix = tf.multiply(action_mask, log_p_all)
        log_probabilities = tf.reduce_sum(log_probabilities_matrix, axis=1)

        # Step 2: Compute the pseudo loss
        # note: tf.stop_gradient(Q_values_placeholder) prevent the backpropagation into the Q_values_placeholder
        #   |   witch contain rewards_to_go. It treat the values of the tensor as constant during backpropagation.
        # weighted_likelihoods = tf.multiply(
        #     tf.stop_gradient(Q_values_placeholder), log_probabilities)
        weighted_likelihoods = log_probabilities * tf.stop_gradient(Q_values_placeholder)
        pseudo_loss = -tf.reduce_mean(weighted_likelihoods)
        return pseudo_loss


def policy_optimizer(pseudo_loss: tf.Tensor, learning_rate: list or tf_cv1.Tensor, global_gradient_step=None,
                     name=vocab.policy_optimizer) -> tf.Operation:
    """
    Define the optimizing methode for training the REINFORE agent
    """
    return tf_cv1.train.AdamOptimizer(learning_rate=learning_rate).minimize(pseudo_loss,
                                                                            global_step=global_gradient_step, name=name)


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


def setup_commented_run_dir_str(exp_spec: ExperimentSpec, agent_root_dir: str) -> str:
    date_now = datetime.now()

    experiment_name = exp_spec.paramameter_set_name
    comment = exp_spec.comment

    if comment is not None:
        comment = "(" + comment + ")"
        cleaned_comment = comment.replace(" ", "+")
        experiment_name = experiment_name + comment
    else:
        cleaned_comment = ""

    cleaned_name = experiment_name.replace(" ", "_")

    tag = exp_spec.rerun_tag
    if tag is not None:
        exp_str = "Exp-{}-{}".format(tag, cleaned_comment)
        runs_dir = "{}/graph/{}".format(agent_root_dir, exp_str)
        tag_i = "{}-{}".format(tag, exp_spec.rerun_idx)
        run_str = "Run-{}-{}-d{}h{}m{}s{}".format(tag_i, cleaned_name, date_now.day, date_now.hour, date_now.minute,
                                                  date_now.second)
    else:
        runs_dir = "{}/graph".format(agent_root_dir)
        run_str = "Run--{}-d{}h{}m{}s{}".format(cleaned_name, date_now.day, date_now.hour, date_now.minute,
                                                date_now.second)

    run_dir = "{}/{}".format(runs_dir, run_str)
    return run_dir


def learning_rate_scheduler(max_gradient_step_expected: int, learning_rate: float, lr_decay_rate: float = 1e-1,
                            name_sufix: str = None) -> Tuple[tf_cv1.Tensor, tf_cv1.Variable]:
    """
    Create a learning rate sheduler, a global step counter variable and a TF summary for to keep track in TensorBoard
    To turn OFF scheduler: set lr_decay_rate=1
    ex:
        (lr=1e-1, decay_rate=1e-1, max_epoch=20) --> ~1e-2 after 20 epoch (0.011220184543019637)
        (lr=1e-1, decay_rate=1e-3, max_epoch=20) --> ~1e-4 after 20 epoch (0.0001412537544622755)


    How to use it:
        Ex: Momentum optimizer with Learning Rate Scheduling

                decayed_learning_rate, global_step = learning_rate_scheduler(X_train_nb_of_samples,
                                                                             initial_learning_rate,
                                                                             batch_size, max_epoch,
                                                                             lr_decay_rate)
                optimizer = tf.train.MomentumOptimizer(decayed_learning_rate, momentum=self.momentum, use_nesterov=True)
                self._training_op = optimizer.minimize(loss, global_step=global_step, name="training_op")

    Check TensorFlow v1 exponential_decay doc for aditional info
        https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/exponential_decay

    For behavior intuition, check: /exploration_and_benchmarking/behavior-learning_rate_scheduler.py

    :param learning_rate:               the initial learning rate
    :type learning_rate: float
    :param lr_decay_rate:               learning-rate decay rate
    :type lr_decay_rate: float
    :param max_gradient_step_expected:  Optional - Will be computed this way ```batch_size * max_epoch``` if None
    :param name_sufix:                  a sufix to be added to the op name ex: 'policy_', 'critic_'
    :return:                            the decayed learning rate , the gradient step counter
    :rtype:                             Tuple[tf_cv1.Tensor, tf_cv1.Variable]
    """
    # assert isinstance(lr_decay_rate, float)

    if name_sufix is not None:
        name_sufix += '_'

    # decay_step = max_gradient_step_expected // batch_size * max_epoch * 0.1

    gradient_step_counter = _gradient_step_counter_op(name=name_sufix)
    decayed_learning_rate = tf_cv1.train.exponential_decay(learning_rate=learning_rate,
                                                           global_step=gradient_step_counter,
                                                           decay_steps=max_gradient_step_expected,
                                                           decay_rate=lr_decay_rate)
    tf_cv1.summary.scalar(name_sufix + 'learning_rate', decayed_learning_rate, family=vocab.learning_rate)
    return decayed_learning_rate, gradient_step_counter


def _gradient_step_counter_op(name=None) -> tf_cv1.Variable:
    """
    Create a gradient step counter variable that keep track of how much training step as elapse since begining of
    training.
    Usage: Pass a _gradient_step_counter_op instance to a TF optimizer minimize(). It will incremente it at each update.

    usage:
        tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
            .minimize(myLoss, global_step=_gradient_step_counter_op)
    """
    return tf_cv1.Variable(0, trainable=False, name=name + 'gradient_step_counter')


def he_initialization(input_op: tf.Tensor, nb_of_neuron: int, seed=None) -> tf.Tensor:
    """
    He initialization
    The best initialization strategy for Relu (and variance of Relu) activation function

    How to use it:
        In a hidden layer
            Weight = tf.get_variable(name='W', initializer=he_initialization(input_op, nb_of_neuron, seed))
            bias   = tf.get_variable(name='b', shape=(nb_of_neuron), initializer=tf.initializers.zeros)

    Code from my project Study_on_Deep_Reinforcement_Learning/TensorFlow_exploration/multilayer_perceptron
    /LowLevel_TF_multilayer_perceptron.py

    ** Important: the weight random initializer choice make a big difference on the learning performance
    :param input_op: a tensorFlow operation
    :param nb_of_neuron: nb of neuron in the layer
    :param seed:
    :return: a tensor initialize with random value folowing the He initialization method
    """
    nb_of_input_unit = int(input_op.get_shape()[1])
    stddev = 2 / np.sqrt(nb_of_input_unit + nb_of_neuron)
    initialized_tensor = tf.truncated_normal((nb_of_input_unit, nb_of_neuron), stddev=stddev, seed=seed)
    return initialized_tensor


def gym_environment_reward_assesment(env: Union[TimeLimit, Any], sample_size: int = 10000) -> float:
    """Tool to asses the reward size of a gym environment
    
    Exemple of use with a buildingbloc.GymPlayground object:
        >>> playgroundLunar = GymPlayground('LunarLanderContinuous-v2')
        >>> myEnv = playgroundLunar.env.env
        >>> theAverageLunarLanderEnvReward = gym_environment_reward_assesment(myEnv, sample_size=1000)
        
    :param env: a Gym instanciated environment
    :param sample_size: reward sample size
    :return: the average environment reward
    """
    env.reset()
    rewards = [env.step(env.action_space.sample())[1] for step in range(sample_size)]
    return np.mean(rewards)
