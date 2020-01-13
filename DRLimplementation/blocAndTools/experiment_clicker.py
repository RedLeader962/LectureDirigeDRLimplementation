# coding=utf-8

class ExperimentClicker(object):
    """
    A counter for tracking experiment evolution
    """
    __slots__ = ['_gradient_step', '_target_update_step']
    
    def __init__(self):
        self._gradient_step = 0
        self._target_update_step = 0
    
    """ ---- Gradient step & target update ---- """
    
    def step_all(self) -> None:
        self._gradient_step += 1
        self._target_update_step += 1
        return None
    
    """ ---- Gradient step ---- """
    
    def gradient_step(self) -> None:
        self._gradient_step += 1
        return None
    
    def reset_gradient_step_count(self) -> None:
        self._gradient_step = 0
        return None
    
    @property
    def gradient_step_count(self):
        return self._gradient_step
    
    """ ---- Target update ---- """
    
    def target_update_step(self) -> None:
        self._target_update_step += 1
        return None
    
    def reset_target_update_count(self) -> None:
        self._target_update_step = 0
        return None
    
    @property
    def target_update_count(self):
        return self._target_update_step
