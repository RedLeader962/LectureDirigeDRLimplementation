# coding=utf-8

import sys
import os

from SoftActorCritic.SoftActorCriticAgent import SoftActorCriticAgent
from SoftActorCritic.SoftActorCriticBrain import (
    apply_action_bound, build_gaussian_policy_graph,
    build_critic_graph_q_theta, build_critic_graph_v_psi,
    actor_train, critic_q_theta_train, critic_v_psi_train, critic_learning_rate_scheduler,
    )

name = "SoftActorCritic"


def solve_PYTHONPATH():
    """Solve: broken import path problem occuring when script are invocated from command line

    This problematic behavior is a side effect of importing module/package outside the current package tree.
    This is not problematic when your working in a controled environment (aka a IDE with hardcoded PYTHONPATH).
    However, when your script will be execute from the command line (outside your IDE) all hell will break loose.

    (CRITICAL) todo --> appended to each package __init__ containing module that will be invocated from command line:
    (nice to have) todo:refactor --> make one, root level, parameterizable function:
                                                                see DRLimplementation/BasicPolicyGradient/__init__.py
    """
    absolute_current_dir = os.getcwd()
    print(":: On INIT - cwd: {}".format(absolute_current_dir))

    if os.path.basename(absolute_current_dir) != "DRLimplementation":
        # meaning the module invocating the script is not at project root level

        while True:
            absolute_parent_dir, curent_dir = os.path.split(absolute_current_dir)

            if os.path.basename(absolute_parent_dir) == "DRLimplementation":
                assert os.path.exists(absolute_parent_dir), "Something is wrong with path resolution"
                sys.path.insert(0, absolute_parent_dir)
                print(":: insert to sys.path: {}".format(absolute_parent_dir))
                break
            else:
                absolute_current_dir = absolute_parent_dir


solve_PYTHONPATH()
