# coding=utf-8

import pytest
import REINFORCE_integration_reference as integration_ref


def test():
    env_name = 'CartPole-v0'
    env_max_return=200.000
    max_epochs = 50
    # hidden_sizes = [4, 4, 4]
    # hidden_sizes = [32]
    hidden_sizes = [3]

    epoch_generator = integration_ref.train(
        env_name=env_name,
        epochs=max_epochs,
        hidden_sizes=hidden_sizes)

    for epoch in epoch_generator:
        (i, batch_loss, mean_return, average_len) = epoch

        if mean_return == env_max_return:

            print("\n >>> Good to go! The agent did learn something\n"
                  'epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f\n\n' %
                  (i, batch_loss, mean_return, average_len))
            break
        else:
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (i, batch_loss, mean_return, average_len))

            assert i < (max_epochs -1), "\n\n\t>>> The agent did not learn enough in {} epoch. \n" \
                                        "\t>>> \tLoss:\t\t\t{:.2f}\n\t>>> \tMean return:\t{:.2f} < {}\n".format(
                max_epochs, batch_loss, mean_return, env_max_return)

