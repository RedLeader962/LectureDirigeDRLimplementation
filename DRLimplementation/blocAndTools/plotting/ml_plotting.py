#!/usr/bin/env python
import os

from matplotlib import pyplot as plt
from matplotlib import gridspec as gs


def plot_learning_curve(show_plot, nb_of_neuron_by_hidden_layer, layers_activation, initial_learning_rate, momentum,
                        lr_decay_rate, max_epoch, batch_size, early_stopping, gradient_clipping_threshold, l1_scale,
                        l2_scale, description, _early_stop_flag, _last_epoch, _accuracy_train_log,
                        _accuracy_validation_log, test_accuracy_score, exp_log_dir, dnn_object_id):  # pragma: no cover
    
    # plt.style.use('dark_background')
    # plt.style.use('seaborn-darkgrid')
    # plt.style.use('seaborn')
    # plt.style.use('ggplot')
    plt.style.use('bmh')  # <-- very clean
    # plt.style.use('classic')
    
    fig = plt.figure(figsize=(9, 6))
    plot_gs = gs.GridSpec(5, 1)
    fig.text(x=0.59, y=0.35, fontsize='large',
             s="hiden layer:{}   activation: {}\n"
               "layers cells:{}\n"
               "learning rate:{}\n"
               "momentum:{}\n"
               "lr decay rate {}\n"
               "max epoch:{}  batch:{}\n"
               "early stopping: {}\n"
               "gradient clipping: {}\n"
               "regularisation L1:{} L2: {}".format(len(nb_of_neuron_by_hidden_layer),
                                                    layers_activation.__name__,
                                                    nb_of_neuron_by_hidden_layer,
                                                    round(initial_learning_rate, 6),
                                                    momentum,
                                                    round(lr_decay_rate, 6),
                                                    max_epoch, batch_size, early_stopping,
                                                    gradient_clipping_threshold,
                                                    l1_scale, l2_scale))
    if early_stopping and _early_stop_flag:
        fig.text(x=0.59, y=0.26, fontsize='large', s="Early stopping at epoch {}".format(_last_epoch))

    # if description:
    #     fig.text(x=0.5, y=0.04, fontsize='large', s="{}".format(description), ha='center')

    """Learning curve"""
    ax_1 = fig.add_subplot(plot_gs[0:4, 0])
    ax_1.plot(range(len(_accuracy_train_log)), _accuracy_train_log, color='blue', label='Train')
    ax_1.plot(range(len(_accuracy_validation_log)), _accuracy_validation_log, color='green', label='Validation')
    ax_1.set_title("Multilayer Perceptron learning accuracy", fontsize='xx-large')
    ax_1.set_ylim(0, 1.2)
    ax_1.set_xlabel("step")
    ax_1.set_ylabel("accuracy")
    ax_1.legend(loc='best')
    ax_1.grid(True)

    """model prediction"""

    fig.text(x=0.5, y=0.1, fontsize='xx-large', ha='center',
             s="Test accuracy: {}%".format(test_accuracy_score * 100))

    fig.text(x=0.01, y=0.01, s="{}---{}".format(description, dnn_object_id))
    # fig.tight_layout()

    try:
        os.makedirs("{}/plot_png".format(exp_log_dir), exist_ok=False)
    except FileExistsError:
        pass
    finally:
        fig.savefig('{}/plot_png/{}---{}.png'.format(exp_log_dir, description.replace(" ", "_"), dnn_object_id),
                    dpi='figure',
                    format='png')

    if show_plot:
        plt.show()

    plt.close('all')
    return None


def plot_learning_curve_and_prediction(X_test, y_pred, show_plot, nb_of_neuron_by_hidden_layer, layers_activation,
                                       initial_learning_rate, momentum, lr_decay_rate, max_epoch, batch_size,
                                       early_stopping, gradient_clipping_threshold, l1_scale, l2_scale, description,
                                       _early_stop_flag, _last_epoch, _accuracy_train_log, _accuracy_validation_log,
                                       test_accuracy_score, exp_log_dir, dnn_object_id):

    # plt.style.use('dark_background')
    # plt.style.use('seaborn-darkgrid')
    # plt.style.use('seaborn')
    # plt.style.use('ggplot')
    plt.style.use('bmh')  # <-- very clean
    # plt.style.use('classic')

    fig = plt.figure(figsize=(9, 11))
    plot_gs = gs.GridSpec(2, 4)
    fig.text(x=0.59, y=0.55, fontsize='large',
             s="hiden layer:{}   activation: {}\n"
               "layers cells:{}\n"
               "learning rate:{}\n"
               "momentum:{}\n"
               "lr decay rate {}\n"
               "max epoch:{}  batch:{}\n"
               "early stopping: {}\n"
               "gradient clipping: {}\n"
               "regularisation L1:{} L2: {}".format(len(nb_of_neuron_by_hidden_layer),
                                                    layers_activation.__name__,
                                                    nb_of_neuron_by_hidden_layer,
                                                    round(initial_learning_rate, 6),
                                                    momentum,
                                                    round(lr_decay_rate, 6),
                                                    max_epoch, batch_size, early_stopping,
                                                    gradient_clipping_threshold,
                                                    l1_scale, l2_scale))
    if early_stopping and _early_stop_flag:
        fig.text(x=0.59, y=0.46, fontsize='large', s="Early stopping at epoch {}".format(_last_epoch))

    # if description:
    #     fig.text(x=0.5, y=0.04, fontsize='large', s="description: {}".format(description), ha='center')

    """Learning curve"""
    ax_1 = fig.add_subplot(plot_gs[0, :])
    ax_1.plot(range(len(_accuracy_train_log)), _accuracy_train_log, color='blue', label='Train')
    ax_1.plot(range(len(_accuracy_validation_log)), _accuracy_validation_log, color='green', label='Validation')
    ax_1.set_title("Multilayer Perceptron learning accuracy", fontsize='xx-large')
    ax_1.set_ylim(0, 1.2)
    ax_1.set_xlabel("step")
    ax_1.set_ylabel("accuracy")
    ax_1.legend(loc='best')
    ax_1.grid(True)

    """Test image and model prediction"""
    ax_2 = fig.add_subplot(plot_gs[1, 0])
    ax_2.imshow(X_test[0].reshape(28, 28), cmap="Greys")
    ax_2.set_xlabel("Test prediction: {}".format(y_pred[0]), fontsize='large')
    ax_3 = fig.add_subplot(plot_gs[1, 1])
    ax_3.imshow(X_test[1].reshape(28, 28), cmap="Greys")
    ax_3.set_xlabel("Test prediction: {}".format(y_pred[1]), fontsize='large')
    ax_4 = fig.add_subplot(plot_gs[1, 2])
    ax_4.imshow(X_test[2].reshape(28, 28), cmap="Greys")
    ax_4.set_xlabel("Test prediction: {}".format(y_pred[2]), fontsize='large')
    ax_5 = fig.add_subplot(plot_gs[1, 3])
    ax_5.imshow(X_test[3].reshape(28, 28), cmap="Greys")
    ax_5.set_xlabel("Test prediction: {}".format(y_pred[3]), fontsize='large')
    fig.text(x=0.5, y=0.1, fontsize='xx-large', ha='center',
             s="Test accuracy: {}%".format(test_accuracy_score * 100))

    fig.text(x=0.01, y=0.01, s="{}---{}".format(description, dnn_object_id))
    fig.tight_layout()

    try:
        os.makedirs("{}/plot_png".format(exp_log_dir), exist_ok=False)
    except FileExistsError:
        pass
    finally:
        fig.savefig('{}/plot_png/{}---{}.png'.format(exp_log_dir, description.replace(" ", "_"), dnn_object_id),
                    dpi='figure',
                    format='png')

    if show_plot:
        plt.show()

    plt.close('all')
    return None
