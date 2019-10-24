# coding=utf-8
from dataclasses import dataclass

@dataclass()
class BatchStats:
    mean_return: float
    max_return: float
    min_return: float
    std_return: float
    mean_trj_lenght: float
    max_trj_lenght: float
    min_trj_lenght: float
    std_trj_lenght: float
    batch_id: int
    step_collected: int
    trajectory_collected: int
