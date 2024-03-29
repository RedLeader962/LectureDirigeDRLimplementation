# coding=utf-8

class DiscreteTimestepCounter(object):
    """
    A time step counter for tracking discrete time evolution
    2 level: GLOBAL timestep & LOCAL timestep
    """
    __slots__ = ['_local_step_idx', '_global_step_idx', '_per_epoch_step_idx']

    def __init__(self):
        self._local_step_idx = 0
        self._per_epoch_step_idx = 0
        self._global_step_idx = 0

    """ ---- Global & local ---- """

    def step_all(self) -> None:
        self._local_step_idx += 1
        self._per_epoch_step_idx += 1
        self._global_step_idx += 1
        return None

    """ ---- Global step ---- """

    def global_step(self) -> None:
        self._global_step_idx += 1
        return None

    def reset_global_count(self) -> None:
        self._global_step_idx = 0
        return None
    
    @property
    def global_count(self):
        return self._global_step_idx
    
    """ ---- Local step ---- """
    
    def local_step(self) -> None:
        self._local_step_idx += 1
        return None

    def reset_local_count(self) -> None:
        self._local_step_idx = 0
        return None

    @property
    def local_count(self):
        return self._local_step_idx

    """ ---- per epoch step ---- """

    def per_epoch_step(self) -> None:
        self._per_epoch_step_idx += 1
        return None

    def reset_per_epoch_count(self) -> None:
        self._per_epoch_step_idx = 0
        return None

    @property
    def per_epoch_count(self):
        return self._per_epoch_step_idx
