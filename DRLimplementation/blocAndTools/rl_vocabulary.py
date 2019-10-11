# coding=utf-8

from collections import namedtuple

_rl_vocab_list = [
    'REINFORCE', 'ActorCritic', 'DQN',
    'Discrete', 'Continuous',
    'trajectorie', 'timestep', 'epoch', 'horizon',
    'observation', 'action', 'policy', 'transition_dynamic', 'reward', 'reward_to_go', 'Qvalues', 'G',
    'policy_theta', 'policy_theta_D', 'policy_theta_C',
    'theta_NeuralNet',
    'sampled_action', 'sampled_action_log_pr', 'action_space_log_pr',
    'pseudo_loss', 'likelihood', 'negative_likelihood'
    'V_pi', 'Q_pi', 'A_pi',
    'discout_factor',
    'baseline',
    'objective', 'loss', 'inference', 'optimizer',
    'learning_rate',
    'entropy', 'KL',
    'Multi_Layer_Perceptron',
    'input_placeholder', 'output_placeholder', 'Qvalues_placeholder',
    'input_layer', 'hidden_', 'output_layer', 'logits'
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
rl_name = namedtuple('RLvocabulary', _rl_vocab_list, defaults=_rl_vocab_list)



