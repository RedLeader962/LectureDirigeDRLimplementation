# coding=utf-8
import pytest
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
from datetime import datetime

from DRLimplementation.ActorCritic import ActorCriticAgent
from blocAndTools.buildingbloc import ExperimentSpec
from blocAndTools.visualisationtools import ConsolPrintLearningStats

tf_cv1 = tf.compat.v1   # shortcut
deprecation._PRINT_DEPRECATION_WARNINGS = False

"""
Start TensorBoard in terminal:
    tensorboard --logdir=DRLimplementation/tests/test_Z_integration/test_integrationActorCritic
    
In browser, go to:
    http://0.0.0.0:6006/ 
"""

AGENT_ROOT_DIR = "test_Z_integration/test_integrationActorCritic"

CARTPOLE_HPARAM = {
    'prefered_environment': 'CartPole-v0',
    'paramameter_set_name': 'Integration test CartPole-v0',
    'MonteCarloTarget': True,
    'isTestRun': False,
    'batch_size_in_ts': 5000,
    'max_epoch': 50,
    'discounted_reward_to_go': True,
    'discout_factor': 0.999,
    'learning_rate': 1e-2,
    'theta_nn_h_layer_topo': (62, ),
    'random_seed': 82,
    'theta_hidden_layers_activation': tf.nn.tanh,        # tf.nn.relu,
    'theta_output_layers_activation': None,
    'render_env_every_What_epoch': 100,
    'print_metric_every_what_epoch': 2,
}

CARTPOLE_HPARAM_FAIL = CARTPOLE_HPARAM.copy()
CARTPOLE_HPARAM_FAIL['theta_nn_h_layer_topo'] = (3,)


def init_spec_and_ActorCriticAgent(hparam):
    tf_cv1.reset_default_graph()

    exp_spec = ExperimentSpec()
    exp_spec.set_experiment_spec(hparam)
    actorcritic_agent = ActorCriticAgent(exp_spec, agent_root_dir=AGENT_ROOT_DIR)

    consol_print_learning_stats = ConsolPrintLearningStats(exp_spec, exp_spec.print_metric_every_what_epoch)

    def epoch_generator():
        return actorcritic_agent._training_epoch_generator(consol_print_learning_stats, render_env=False)

    return epoch_generator, exp_spec, consol_print_learning_stats


@pytest.fixture
def setup_ActorCritic_train_algo_generator_with_PASSING_spec():
    nb_of_try = 2
    env_max_return = 200.000

    """ ---- setup summary collection for TensorBoard ---- """
    date_now = datetime.now()
    run_str = "Run--{}h{}--{}-{}-{}".format(date_now.hour, date_now.minute, date_now.day,
                                            date_now.month, date_now.year)
    writer = tf_cv1.summary.FileWriter("{}/graph/runs/{}".format(AGENT_ROOT_DIR, run_str),
                                       tf_cv1.get_default_graph())

    epoch_generator, exp_spec, consol_print_learning_stats = init_spec_and_ActorCriticAgent(hparam=CARTPOLE_HPARAM)
    yield epoch_generator, nb_of_try, env_max_return, exp_spec
    consol_print_learning_stats.print_experiment_stats(print_plot=not exp_spec.isTestRun)
    writer.close()


@pytest.fixture
def setup_ActorCritic_train_algo_generator_with_FAILING_spec():
    nb_of_try = 2
    env_max_return = 200.000

    epoch_generator, exp_spec, consol_print_learning_stats = init_spec_and_ActorCriticAgent(hparam=CARTPOLE_HPARAM_FAIL)
    yield epoch_generator, nb_of_try, env_max_return, exp_spec
    consol_print_learning_stats.print_experiment_stats(print_plot=not exp_spec.isTestRun)


def training_loop(epoch_generator, env_max_return):
    """
    Utility fct for Actor-Critic type algorithm integration testing
    """
    agent_learned = False

    for epoch_stats in epoch_generator:
        epoch, batch_loss, mean_return, average_len = epoch_stats

        if mean_return == env_max_return:
            agent_learned = True
            break

    return epoch_stats, agent_learned


def test_integration_ActorCriticAgent_train_PASS(setup_ActorCritic_train_algo_generator_with_PASSING_spec):
    epoch_generator, nb_of_try, env_max_return, exp_spec = setup_ActorCritic_train_algo_generator_with_PASSING_spec

    error_str = ""
    agent_learned = False

    for run in range(nb_of_try):
        """ Repeate the test if it fail. It's a work around reference the probabilistic nature of the algo. """

        trl = training_loop(epoch_generator(), env_max_return)
        print("\nrun:{} -- {}".format(run, trl))
        epoch_stats, agent_learned = trl

        epoch, batch_loss, mean_return, average_len = epoch_stats

        if agent_learned:
            print("\n:: Good to go!!! The agent did learn something\n"
                  "epoch: {} \t loss: {:.3f} \t return: {:.3f} \t ep_len: {:.3f}\n\n".format(
                epoch, batch_loss, mean_return, average_len))
            break
        elif not agent_learned:
            error_str += ("\t\tRun {}\n"
                          "\t\t  |\tLoss:\t\t\t{:.3f}\n\t\t  |\tMean return:\t{:.3f} < {} !!\n").format(
                run, batch_loss, mean_return, env_max_return)

    assert agent_learned, ("\n\n"
                           ":: The agent FAILED to learned enough in {} epoch\n"
                           "    - Test run over {} run\n"
                           "    - Env: {} with NN hidden {}\n"
                           "    - Required mean return {}\n"
                           "\n{}\n\n").format(
        exp_spec.max_epoch, nb_of_try, exp_spec.prefered_environment,
        exp_spec.theta_hidden_layers_activation, env_max_return, error_str)


@pytest.mark.skip(reason="Was required to check that each run was unique & done in isolation")
def test_integration_ActorCriticAgent_train_ALL_RUN_DIFFERENT(setup_ActorCritic_train_algo_generator_with_PASSING_spec):
    epoch_generator, nb_of_try, env_max_return, exp_spec = setup_ActorCritic_train_algo_generator_with_PASSING_spec

    loss_at_run_end = []

    repeate_run = 5

    for run in range(repeate_run):
        """ Repeate the test if it fail. It's a work around reference the probabilistic nature of the algo. """

        epoch_stats, agent_learned = training_loop(epoch_generator(), env_max_return)
        (epoch, batch_loss, mean_return, average_len) = epoch_stats

        loss_at_run_end.append(batch_loss)

    assert len(set(loss_at_run_end)) == repeate_run, ">>> Some run where probably not executed in isolation!"


