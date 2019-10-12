# coding=utf-8
name = "DRLTP1PolicyGradient"

import sys
import os

# --- HACK -------------------------------------------------------------------------------------------------------------
#   Solve: broken import path problem occuring when script are invocated from command line
#
#   This problematic behavior is a side effect of importing module/package outside the current package tree.
#   This is not problematic when your working in a controled environment (aka a IDE with hardcoded PYTHONPATH).
#   However, when your script will be execute from the command line (outside your IDE) all hell will break loose.
#
#   (CRITICAL) todo --> appended to each package __init__ containing module that will be invocated from command line:
#

absolute_current_dir = os.getcwd()

while True:
    absolute_parent_dir, curent_dir = os.path.split(absolute_current_dir)

    print(":: On INIT - (abs) parent dir: {}".format(absolute_parent_dir))

    if os.path.basename(absolute_parent_dir) == "drlimplementation":
        assert os.path.exists(absolute_parent_dir), "Something is wrong with path resolution"
        sys.path.insert(0, absolute_parent_dir)
        print(":: insert to sys.path: {}".format(absolute_parent_dir))
        break
    else:
        absolute_current_dir = absolute_parent_dir

# ----------------------------------------------------------------------------------------------------------------end---
