# coding=utf-8

from collections import namedtuple

_rl_vocab_list = [
    'trajectorie', 'timestep', 'epoch', 'horizon',
    'observation', 'action', 'policy', 'transition_dynamic', 'reward', 'reward_to_go', 'G',
    'V_pi', 'Q_pi', 'A_pi',
    'discout_factor',                           # gamma
    'baseline',
    'objective', 'loss', 'inference', 'optimizer',
    'learning_rate',
    'entropy', 'KL',
    'Multi_Layer_Perceptron',
    'input_placeholder', 'output_placeholder',
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
rl_name = namedtuple('RLvocabulary', _rl_vocab_list, defaults=_rl_vocab_list).__call__()



