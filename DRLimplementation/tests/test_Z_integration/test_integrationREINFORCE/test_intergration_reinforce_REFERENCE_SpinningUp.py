# coding=utf-8

import pytest
from tests.test_Z_integration.test_integrationREINFORCE import reference_implementation_SpinningUp as integration_ref

"""
Start TensorBoard in terminal:
    cd DRLimplementation
    tensorboard --logdir=DRLimplementation/tests/test_Z_integration/test_integrationREINFORCE/graph
    
In browser, go to:
    http://0.0.0.0:6006/ 
"""

# pytestmark = pytest.mark.skip("all tests still WIP")  # (Priority) todo:implement --> coverage: then remove line

@pytest.fixture
def setup_train_algo_generator_with_PASSING_spec():
    env_name = 'CartPole-v0'
    env_max_return = 200.000
    max_epochs = 50
    hidden_sizes = [62]     # (!)
    nb_of_try = 2

    def epoch_generator():
        return integration_ref.train(env_name=env_name,
                                     epochs=max_epochs,
                                     hidden_sizes=hidden_sizes)

    return epoch_generator, nb_of_try, env_name, env_max_return, max_epochs, hidden_sizes

@pytest.fixture
def setup_train_algo_generator_with_FAILING_spec():
    env_name = 'CartPole-v0'
    env_max_return = 200.000
    max_epochs = 50
    hidden_sizes = [3]      # (!)
    nb_of_try = 2

    def epoch_generator():
        return integration_ref.train(env_name=env_name,
                                     epochs=max_epochs,
                                     hidden_sizes=hidden_sizes)

    return epoch_generator, nb_of_try, env_name, env_max_return, max_epochs, hidden_sizes

def training_loop(epoch_generator, env_max_return, max_epochs):
    """
    Utility fct for REINFORCE type algorithm integration testing
    """
    agent_learned = False

    for epoch_stats in epoch_generator:
        i, batch_loss, mean_return, average_len = epoch_stats

        if mean_return == env_max_return:
            agent_learned = True
            break

    return epoch_stats, agent_learned

@pytest.mark.skip(reason="Problem SOLVED")
def test_bloc_integration_to_working_REINFORCE_algo_PASS(setup_train_algo_generator_with_PASSING_spec):
    epoch_generator, nb_of_try, env_name, env_max_return, max_epochs, hidden_sizes = setup_train_algo_generator_with_PASSING_spec

    error_str = ""

    for run in range(nb_of_try):
        """ Repeate the test if it fail. It's a work around reference the probabilistic nature of the algo. """

        trl = training_loop(epoch_generator(), env_max_return, max_epochs)
        print("\nrun:{} -- {}".format(run, trl))
        epoch_stats, agent_learned = trl

        i, batch_loss, mean_return, average_len = epoch_stats

        if agent_learned:
            print("\n:: Good to go!!! The agent did learn something\n"
                  "epoch: {} \t loss: {:.3f} \t return: {:.3f} \t ep_len: {:.3f}\n\n".format(
                i, batch_loss, mean_return, average_len))
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
        max_epochs, nb_of_try, env_name, hidden_sizes, env_max_return, error_str)


@pytest.mark.skip(reason="Was required to check that each run was unique & done in isolation")
def test_training_loop_ALL_RUN_DIFFERENT(setup_train_algo_generator_with_PASSING_spec):
    epoch_generator, nb_of_try, env_name, env_max_return, max_epochs, hidden_sizes = setup_train_algo_generator_with_PASSING_spec

    loss_at_run_end = []

    repeate_run = 5

    for run in range(repeate_run):
        """ Repeate the test if it fail. It's a work around reference the probabilistic nature of the algo. """

        epoch_stats, agent_learned = training_loop(epoch_generator(), env_max_return, max_epochs)
        (i, batch_loss, mean_return, average_len) = epoch_stats

        loss_at_run_end.append(batch_loss)

    assert len(set(loss_at_run_end)) == repeate_run, ">>> Some run where probably not executed in isolation!"


