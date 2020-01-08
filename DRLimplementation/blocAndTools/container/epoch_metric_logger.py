# coding=utf-8
from dataclasses import dataclass
from typing import Union, List

import numpy as np

"""
Metric recommandation from OpenAI SpiningUp:

    1. look at the mean/std/min/max for
        - cumulative rewards,
        - episode lengths,
        - and value function estimates,
    2. losses of the objectives,
    3. details of any exploration parameters:
        - like mean entropy for stochastic policy optimization,
        - or current epsilon for epsilon-greedy as in DQN.
    4. Also, watch videos of your agent's performance every now and then;
        this will give you some insights you wouldn't get otherwise.
"""


@dataclass()
class EpochMetricLogger:
    trjs_return: List[float] = list
    trjs_lenght: List[float] = list
    agent_eval_trjs_return: List[float] = list
    agent_eval_trjs_lenght: List[float] = list
    v_loss: List[float] = list
    q1_loss: List[float] = list
    q2_loss: List[float] = list
    pi_loss: List[float] = list
    pi_log_likelihood: List[float] = list
    policy_pi: List[float] = list
    policy_mu: List[float] = list
    v_values: List[float] = list
    q1_values: List[float] = list
    q2_values: List[float] = list
    _epoch_id: int = None
    _trj_collected: int = 0
    _eval_trj_collected: int = 0

    def append_trajectory_metric(self, trj_return, trj_lenght) -> None:
        self.trjs_return.append(trj_return)
        self.trjs_lenght.append(trj_lenght)
        self._trj_collected += 1
        return None

    def append_agent_eval_trj_metric(self, eval_trj_return, eval_trj_lenght) -> None:
        self.agent_eval_trjs_return.append(eval_trj_return)
        self.agent_eval_trjs_lenght.append(eval_trj_lenght)
        self._eval_trj_collected += 1
        return None

    def append_loss(self, critic_v_loss, critic_q1_loss, critic_q2_loss, actor_kl_loss) -> None:
        self.v_loss.append(critic_v_loss)
        self.q1_loss.append(critic_q1_loss)
        self.q2_loss.append(critic_q2_loss)
        self.pi_loss.append(actor_kl_loss)
        return None

    def append_policy_metric(self, pi_log_likelihood, policy_pi, policy_mu) -> None:
        self.pi_log_likelihood.append(pi_log_likelihood)
        self.policy_pi.append(policy_pi)
        self.policy_mu.append(policy_mu)
        return None
    
    def append_approximator_values(self, v_value, q1_value, q2_value) -> None:
        self.v_values.append(v_value)
        self.q1_values.append(q1_value)
        self.q2_values.append(q2_value)
        return None
    
    def append_all_epoch_metric(self, critic_v_loss, critic_q1_loss, critic_q2_loss, actor_kl_loss,
                                pi_log_likelihood, policy_pi, policy_mu,
                                v_value, q1_value, q2_value) -> None:
        self.append_loss(critic_v_loss, critic_q1_loss, critic_q2_loss, actor_kl_loss)
        self.append_policy_metric(pi_log_likelihood, policy_pi, policy_mu)
        self.append_approximator_values(v_value, q1_value, q2_value)
        return None

    @property
    def total_training_timestep_collected(self):
        return np.asarray(self.trjs_lenght).sum()

    @property
    def mean_trjs_return(self):
        return np.asarray(self.trjs_return).mean()

    @property
    def mean_trjs_lenght(self):
        return np.asarray(self.trjs_lenght).mean()

    @property
    def agent_eval_mean_trjs_return(self):
        return np.asarray(self.agent_eval_trjs_return).mean()

    @property
    def agent_eval_mean_trjs_lenght(self):
        return np.asarray(self.agent_eval_trjs_lenght).mean()

    @property
    def mean_v_loss(self):
        return np.asarray(self.v_loss).mean()

    @property
    def mean_q1_loss(self):
        return np.asarray(self.q1_loss).mean()

    @property
    def mean_q2_loss(self):
        return np.asarray(self.q2_loss).mean()
    
    @property
    def mean_pi_loss(self):
        return np.asarray(self.pi_loss).mean()
    
    @property
    def mean_pi_log_likelihood(self):
        return np.asarray(self.pi_log_likelihood).mean()
    
    @property
    def mean_policy_pi(self):
        return np.asarray(self.policy_pi).mean()
    
    @property
    def mean_policy_mu(self):
        return np.asarray(self.policy_mu).mean()
    
    @property
    def mean_v_values(self):
        return np.asarray(self.v_values).mean()
    
    @property
    def mean_q1_values(self):
        return np.asarray(self.q1_values).mean()
    
    @property
    def mean_q2_values(self):
        return np.asarray(self.q2_values).mean()

    @property
    def nb_trj_collected(self):
        return self._trj_collected
    
    def new_epoch(self, epoch_id):
        self.reset()
        self._epoch_id = epoch_id
    
    def reset(self):
        self.trjs_return.clear()
        self.trjs_lenght.clear()
        self.agent_eval_trjs_return.clear()
        self.agent_eval_trjs_lenght.clear()
        self.v_loss.clear()
        self.q1_loss.clear()
        self.q2_loss.clear()
        self.pi_loss.clear()
        self.pi_log_likelihood.clear()
        self.policy_pi.clear()
        self.policy_mu.clear()
        self.v_values.clear()
        self.q1_values.clear()
        self.q2_values.clear()
        self._epoch_id = None
        self._trj_collected = 0
        self._eval_trj_collected = 0
