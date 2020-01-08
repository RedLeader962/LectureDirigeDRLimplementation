# coding=utf-8
from dataclasses import dataclass


@dataclass()
class BasicTrajectoryLogger:
    the_return: float = 0.0
    lenght: int = 0
    
    def push(self, reward) -> None:
        self.the_return += reward
        self.lenght += 1
        return None
