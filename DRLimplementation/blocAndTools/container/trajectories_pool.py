# coding=utf-8
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf

import copy

tf_cv1 = tf.compat.v1  # shortcut

from blocAndTools.buildingbloc import GymPlayground, ExperimentSpec


class TimestepSample:
    __slots__ = ['_container_id', 'obs_t', 'act_t', 'obs_t_prime', 'rew_t', 'done_t']
    
    def __init__(self, container_id: int, playground: GymPlayground):
        self._container_id = container_id
        
        self.obs_t = np.zeros((1, playground.OBSERVATION_DIM), dtype=np.float32)
        self.act_t = np.zeros((1, playground.ACTION_CHOICES), dtype=np.float32)
        self.obs_t_prime = np.zeros((1, playground.OBSERVATION_DIM), dtype=np.float32)
        self.rew_t = 0.0
        self.done_t = 0.0
    
    def replace(self, obs_t: np.ndarray, act_t: np.ndarray, obs_t_prime: np.ndarray,
                rew_t: float, done_t: float) -> None:
        self.obs_t = obs_t
        self.act_t = act_t
        self.obs_t_prime = obs_t_prime
        self.rew_t = rew_t
        self.done_t = done_t
    
    def __repr__(self):
        # Is required explicitely since the use of '__slots__' remove the dict representation from the class object
        myRep = "\n::TimestepSample/\n"
        myRep += "._container_id=\n\n".format(self._container_id)
        myRep += ".obs_t=\n\n".format(self.obs_t)
        myRep += ".act_t=\n\n".format(self.act_t)
        myRep += ".obs_t_prime=\n\n".format(self.obs_t_prime)
        myRep += ".rew_t=\n\n".format(self.rew_t)
        myRep += ".done_t=\n\n".format(self.done_t)
        return myRep
    
    def __eq__(self, other):
        return self._contain_the_same(obs_t=other.obs_t, act_t=other.act_t, obs_t_prime=other.obs_t_prime,
                                      rew_t=other.rew_t, done_t=other.done_t)
    
    def _contain_the_same(self, obs_t, act_t, obs_t_prime, rew_t, done_t):
        check = [np.array_equal(self.obs_t, obs_t), np.array_equal(self.act_t, act_t),
                 np.array_equal(self.obs_t_prime, obs_t_prime),
                 self.rew_t == rew_t, self.done_t == done_t]
        return all(check)


class SampleBatch:
    __slots__ = ['obs_t', 'act_t', 'obs_t_prime', 'rew_t', 'done_t', '_BATCH_SIZE', '_timestepsample_testutility']
    
    def __init__(self, batch_size: int, playground: GymPlayground):
        self.obs_t = [np.zeros((batch_size, playground.OBSERVATION_DIM), dtype=np.float32) for _ in range(batch_size)]
        self.act_t = [np.zeros((batch_size, playground.ACTION_CHOICES), dtype=np.float32) for _ in range(batch_size)]
        self.obs_t_prime = [np.zeros((batch_size, playground.OBSERVATION_DIM), dtype=np.float32) for _ in
                            range(batch_size)]
        self.rew_t = [0.0 for _ in range(batch_size)]
        self.done_t = [0.0 for _ in range(batch_size)]
        self._BATCH_SIZE = batch_size
        
        self._timestepsample_testutility = TimestepSample(container_id=-1, playground=playground)
    
    def __setitem__(self, key, value: TimestepSample):
        assert isinstance(value, TimestepSample)
        self.obs_t[key] = value.obs_t
        self.act_t[key] = value.act_t
        self.obs_t_prime[key] = value.obs_t_prime
        self.rew_t[key] = value.rew_t
        self.done_t[key] = value.done_t
    
    def __getitem__(self, key):
        return self.obs_t[key], self.act_t[key], self.obs_t_prime[key], self.rew_t[key], self.done_t[key]
    
    def get_timestepsample(self, key):
        self._timestepsample_testutility.replace(obs_t=self.obs_t[key], act_t=self.act_t[key],
                                                 obs_t_prime=self.obs_t_prime[key], rew_t=self.rew_t[key],
                                                 done_t=self.done_t[key])
        return self._timestepsample_testutility
    
    def __contains__(self, timestepsample: TimestepSample):
        is_in = [self.get_timestepsample(idx) == timestepsample for idx in range(self._BATCH_SIZE)]
        return any(is_in)
    
    def __eq__(self, other):
        check = [self.get_timestepsample(idx) in other for idx in range(self._BATCH_SIZE)]
        return all(check)
    
    def swap_with_selected_sample(self, samples: List[TimestepSample]):
        assert len(samples) == self._BATCH_SIZE
        for i, v in enumerate(samples):
            self.__setitem__(i, v)
        # return self # <-- (!) This was EVIL
        return copy.deepcopy(self)
    
    def __repr__(self):
        # Is required explicitely since the use of '__slots__' remove the dict representation from the class object
        myRep = "\n::SampleBatch/\n"
        myRep += ".obs_t=\n{}\n\n".format(self.obs_t)
        myRep += ".act_t=\n{}\n\n".format(self.act_t)
        myRep += ".obs_t_prime=\n{}\n\n".format(self.obs_t_prime)
        myRep += ".rew_t=\n{}\n\n".format(self.rew_t)
        myRep += ".done_t=\n{}\n\n".format(self.done_t)
        myRep += "._BATCH_SIZE=\n{}\n\n".format(self._BATCH_SIZE)
        return myRep


class TrajectoriesPool(object):
    
    def __init__(self, capacity: int, batch_size: int, playground: GymPlayground):
        """

        :param capacity: Nb of collected step to keep. Once reached, old step will start being overwriten by new one.
        :param batch_size:
        :param playground: the environment from which sampled step are collected
        """
        self._pool = [TimestepSample(container_id=i + 1, playground=playground) for i in range(capacity)]
        self._idx = 0
        self.CAPACITY = capacity
        self._load = 0
        self._BATCH_SIZE = batch_size
        self._sample_batch = SampleBatch(batch_size, playground)
        self._pool_full_message_delivered = False
    
    def collect_OAnORD(self, obs_t: np.ndarray, act_t: np.ndarray, obs_t_prime: np.ndarray,
                       rew_t: float, done_t: int) -> None:
        """
        Collect obs_t, act_t, obs_t_prime, rew_t and done_t for one timestep
        """
        self._pool[self._idx].replace(obs_t=obs_t, act_t=act_t, obs_t_prime=obs_t_prime, rew_t=rew_t, done_t=done_t)
        
        """
        Move index position ready for next sample collection.
        The index loop over container when _load == capacity
        """
        self._idx = (self._idx + 1) % self.CAPACITY
        
        if not self._pool_full():
            self._load += 1
    
    def _pool_full(self) -> bool:
        if self._load < self.CAPACITY:
            return False
        else:
            if self._pool_full_message_delivered:
                self._pool_full_message_delivered = True
                print(":: Replay pool is full --> ", self._load)
            return True

    @property
    def size(self) -> int:
        return self._load

    def sample_from_pool(self) -> SampleBatch:
        selected_sample = self.sample_from_pool_as_list()
        return self._sample_batch.swap_with_selected_sample(selected_sample)

    def sample_from_pool_as_list(self) -> List[TimestepSample]:
        pool_slice = self._pool[0:self._load]
        selected_sample = random.sample(pool_slice, self._BATCH_SIZE)
        return selected_sample


class PoolManager(object):
    
    def __init__(self, exp_spec: ExperimentSpec, playground: GymPlayground):
        self._trajectories_pool = TrajectoriesPool(exp_spec['pool_capacity'], exp_spec.batch_size_in_ts, playground)
        self._rewards = []
        self._curent_trj_lenght = 0
        self._step_count_since_begining_of_training = 0
        self._trajectories_collected = 0
    
    @property
    def current_pool_size(self) -> int:
        return self._trajectories_pool.size
    
    def timestep_collected_so_far(self) -> int:
        return self._step_count_since_begining_of_training
    
    def trj_collected_so_far(self) -> int:
        return self._trajectories_collected
    
    def collect_OAnORD(self, obs_t: np.ndarray, act_t: np.ndarray, obs_t_prime: np.ndarray,
                       rew_t: float, done_t: int) -> None:
        """ Collect observation, action, next observation, reward and terminal cue for one timestep

        :param obs_t: np.ndarray
        :param act_t: np.ndarray
        :param obs_t_prime: np.ndarray
        :param rew_t: float
        :param done_t: bool
        """
        self._trajectories_pool.collect_OAnORD(obs_t=obs_t, act_t=act_t, obs_t_prime=obs_t_prime,
                                               rew_t=rew_t, done_t=done_t)
        self._rewards.append(rew_t)
        self._curent_trj_lenght += 1
        self._step_count_since_begining_of_training += 1
        return None

    def sample_from_pool(self) -> SampleBatch:
        return self._trajectories_pool.sample_from_pool()

    def trajectory_ended(self) -> Tuple[float, int]:
        """ Must be called at each trajectory end

        Compute:
            1. the trajectory return
            2. the trajectory lenght base on collected samples

        :return: the trajectory return
        """
        trajectory_return = self._compute_trajectory_return()
        trajectory_lenght = self._curent_trj_lenght
        self._reset()
        self._trajectories_collected += 1
        return trajectory_return, trajectory_lenght
    
    def _compute_trajectory_return(self) -> float:
        trj_return = float(np.sum(self._rewards, axis=None))
        return trj_return
    
    def _reset(self):
        self._rewards = []
        self._curent_trj_lenght = 0
        return None
