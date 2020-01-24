#!/usr/bin/env python

import timeit
import numpy as np
import sys


"""
LESSON LEARNED: numpy array indexing (write) is slower compare to (read + write) list of same size

Experiment result:

    Sizeof (with 4000000 items):
    
       numpy array	32.000096 Mb
       python list	32.000064 Mb
    
    Timeit result:
       5 loops,
    
       mod_numpy_array: 1.97616 usec per loop
       append_to_list: 1.53627 usec per loop
       fix_size_list: 0.73891 usec per loop

"""


QUATRE_MILLIONS = 4000000
NUMBER_OF_LOOP = 5


# --- Sizeof -------------------------------------------------------------------------------------------
a1 = np.zeros(QUATRE_MILLIONS)
b1 = [0] * QUATRE_MILLIONS

print("\nSizeof (with {} items):\n".format((QUATRE_MILLIONS)))
print("\t{}\t{} Mb".format("numpy array", sys.getsizeof(a1)*1e-6))
print("\t{}\t{} Mb".format("python list", sys.getsizeof(b1)*1e-6))


# --- Speed -------------------------------------------------------------------------------------------


def mod_numpy_array():
    a = np.zeros(QUATRE_MILLIONS, dtype=int)
    b = np.zeros(QUATRE_MILLIONS, dtype=int)
    c = np.zeros(QUATRE_MILLIONS, dtype=int)

    for i in range(QUATRE_MILLIONS):
        a[i] = 1
        b[i] = 1
        c[i] = 1

    assert len(a) == QUATRE_MILLIONS


def append_to_list():
    a = list()
    b = list()
    c = list()

    for i in range(QUATRE_MILLIONS):
        a.append(1)
        b.append(1)
        c.append(1)

    assert len(a) == QUATRE_MILLIONS


def fix_size_list():
    a = [0] * QUATRE_MILLIONS
    b = [0] * QUATRE_MILLIONS
    c = [0] * QUATRE_MILLIONS

    for i in range(QUATRE_MILLIONS):
        a[i] = 1
        b[i] = 1
        c[i] = 1

    assert len(b) == QUATRE_MILLIONS


result = np.zeros(3)

result[0] = timeit.timeit("mod_numpy_array()", number=NUMBER_OF_LOOP, globals=globals())
result[1] = timeit.timeit("append_to_list()", number=NUMBER_OF_LOOP, globals=globals())
result[2] = timeit.timeit("fix_size_list()", number=NUMBER_OF_LOOP, globals=globals())

result = result/NUMBER_OF_LOOP

print("\nTimeit result:\n\t{} loops,\n\n"
      "\tmod_numpy_array: {:2.5f} usec per loop\n"
      "\tappend_to_list: {:2.5f} usec per loop\n"
      "\tfix_size_list: {:2.5f} usec per loop\n".format(NUMBER_OF_LOOP, result[0], result[1], result[2]))
