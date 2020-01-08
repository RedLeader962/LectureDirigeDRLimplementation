# coding=utf-8
from dataclasses import dataclass
import numpy as np


@dataclass()
class BatchStats:
    batch_id: int
    step_collected: int
    trajectory_collected: int
    mean_return: float
    max_return: float
    min_return: float
    std_return: float
    mean_trj_lenght: float
    max_trj_lenght: float
    min_trj_lenght: float
    std_trj_lenght: float

    def __init__(self, batch_id: int, batch_trjs_returns: list, batch_trjs_lenghts: list, trajectory_collected: int,
                 step_collected: int) -> None:
        """Compute and store statistic about a batch of trajectories return and lenght such as:
            mean, max, min, standard deviation
        
            :param batch_id:
            :param batch_trjs_returns:
            :param batch_trjs_lenghts:
            :param trajectory_collected: used for test purposes
            :param step_collected:
        """
        super().__init__()
    
        _trj_returns = np.array(batch_trjs_returns)
        _trj_lenghts = np.array(batch_trjs_lenghts)
    
        assert _trj_returns.ndim == 1, "The batch_trjs_returns is not of dimension 1"
        assert _trj_lenghts.ndim == 1, "The batch_trjs_lenghts is not of dimension 1"
        assert len(
            _trj_returns) == trajectory_collected, "Nb of batch_trjs_returns collected differ from the container " \
                                                   "trj_count"
        assert len(
            _trj_lenghts) == trajectory_collected, "Nb of batch_trjs_lenghts collected differ from the container " \
                                                   "trj_count"
    
        self.mean_return = _trj_returns.mean()
        self.max_return = _trj_returns.max()
        self.min_return = _trj_returns.min()
        self.std_return = _trj_returns.std()
    
        self.mean_trj_lenght = _trj_lenghts.mean()
        self.max_trj_lenght = _trj_lenghts.max()
        self.min_trj_lenght = _trj_lenghts.min()
        self.std_trj_lenght = _trj_lenghts.std()
    
        self.batch_id = batch_id
        self.step_collected = step_collected
        self.trajectory_collected = trajectory_collected
