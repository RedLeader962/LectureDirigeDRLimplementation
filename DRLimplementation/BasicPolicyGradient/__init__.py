# coding=utf-8
"""
Invoke REINFORCE agent using

    python -m BasicPolicyGradient [--help] [--train] [--render_training] [--discounted] [--play_for] [--test_run]

"""

import sys
import os

from BasicPolicyGradient.REINFORCEbrain import REINFORCE_policy
from BasicPolicyGradient.REINFORCEplayingloop import play_REINFORCE_agent_discrete
from BasicPolicyGradient.REINFORCEtrainingloop import train_REINFORCE_agent_discrete


name = "BasicPolicyGradient"

def solve_PYTHONPATH():
    """Solve: broken import path problem occuring when script are invocated from command line

    This problematic behavior is a side effect of importing module/package outside the current package tree.
    This is not problematic when your working in a controled environment (aka a IDE with hardcoded PYTHONPATH).
    However, when your script will be execute from the command line (outside your IDE) all hell will break loose.

    (CRITICAL) todo --> appended to each package __init__ containing module that will be invocated from command line:
    (nice to have) todo:refactor --> make one, root level, parameterizable function:
                                                                        see DRLimplementation/ActorCritic/__init__.py
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