# coding=utf-8

from collections import namedtuple
from typing import Type
from enum import Enum

class TargetType(Enum):
    MonteCarlo = 1
    Bootstrap = 2

class NetworkType(Enum):
    Split = 1
    Shared = 2


_rl_vocab_list = [
    'REINFORCE', 'ActorCritic', 'DQN',
    'Discrete', 'Continuous',
    'trajectorie', 'timestep', 'epoch', 'horizon',
    'Trajectory_lenght',
    'observation', 'action', 'policy', 'transition_dynamic',
    'obs_t', 'obs_tPrime'
             'Qvalues', 'Vvalues'
                        'reward', 'reward_to_go', 'G',
    'V_estimate', 'Advantage',
    'TD_target', 'TD_error',
    'V_pi', 'Q_pi', 'A_pi',
    'policy_theta', 'policy_theta_D', 'policy_theta_C',
    'shared_network', 'split_network', 'two_input_network',
    'actor_network', 'theta_NeuralNet',
    'critic_network', 'phi_NeuralNet', 'critic_t', 'critic_tPrime',
    'pseudo_loss', 'actor_loss', 'critic_loss',
    'likelihood', 'negative_likelihood',
    'discout_factor',
    'baseline',
    'objective', 'loss', 'inference',
    'optimizer', 'policy_optimizer', 'critic_optimizer', 'policy_training', 'critic_training',
    'learning_rate',
    'entropy', 'KL',
    'Multi_Layer_Perceptron',
    'graph_input', 'obs_ph', 'obs_t_ph', 'obs_tPrime_ph',
    'act_ph', 'rew_ph', 'Qvalues_ph', 'advantage_ph', 'target_ph',
    'input_layer', 'hidden_', 'output_layer', 'logits',
    'sampled_action', 'sampled_action_log_pr', 'action_space_log_pr',
    'policy_mu', 'policy_log_std', 'sampled_action_log_likelihood',
    'policy_phi', 'Q_theta_1', 'Q_theta_2', 'V_psi', 'V_psi_frozen',
    'actor_kl_loss', 'Q_theta_1_loss', 'Q_theta_2_loss', 'V_psi_loss', 'v_psi_frozen_update_ops',
    ]

"""
A standardize vocabulary to use when refering node in a computation graph

    Recommandation: Name all operation in a computation graph

    Pro: 
        - The better your name scopes, the better your visualization in TensorBoard
        - It's easier to get correct operation from graph after restoration
            ref: quiver:///notes/A77E0215-5AA8-47C1-875A-2468C51E54C9
    
            'x = tf.placeholder(..., name=rl_name.action)'
            
            'graph.get_operation_by_name(rl_name.action)' --> return a Operation
            or
            'graph.get_operation_by_name(rl_name.x).outputs[0]' --> return a Tensor  
                
            Placeholder tensor only has one output, thus the :0 part.
            
"""
rl_name: namedtuple = namedtuple('RLvocabulary', _rl_vocab_list, defaults=_rl_vocab_list)


