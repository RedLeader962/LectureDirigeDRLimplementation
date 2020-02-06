# coding=utf-8
import numpy as np
import timeit

HIGH = 1000
NUMBER = 10000000

"""
Time experiment: pass_by_index VS pass_by_insert
Results:
    pass_by_index:          15.298919873
    pass_by_insert:         140.12976299299999
    
Time experiment: reset_by_indexing VS new_instance
Results:
    reset_by_indexing:      7.514461331999996
    new_instance:           13.380060698000001
    
"""

# --- pass_by_index VS pass_by_insert ----------------------------------------------------------------------------------

z1 = np.zeros(HIGH, dtype=np.float)
z2 = np.zeros(HIGH, dtype=np.float)


def pass_by_index():
    idx: int = np.random.randint(low=0, high=HIGH - 1)
    z1[idx] = np.float(idx)
    return None


def pass_by_insert():
    idx: int = np.random.randint(low=0, high=HIGH - 1)
    np.insert(arr=z2, obj=idx, values=np.float(idx))
    return None


print("pass_by_index: ", timeit.timeit('pass_by_index()', number=NUMBER, globals=globals()))

print("pass_by_insert: ", timeit.timeit('pass_by_insert()', number=NUMBER, globals=globals()))

# --- reset ndarray ----------------------------------------------------------------------------------------------------

z3 = np.ones(HIGH, dtype=np.float)
z_new = np.ones(HIGH, dtype=np.float)


def reset_by_indexing():
    z3[0:HIGH] = 0.0
    return z3


def new_instance():
    z_new = np.zeros(HIGH, dtype=np.float)
    return z_new


print("reset_by_indexing: ", timeit.timeit('reset_by_indexing()', number=NUMBER, globals=globals()))

print("new_instance: ", timeit.timeit('new_instance()', number=NUMBER, globals=globals()))
