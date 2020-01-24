#!/usr/bin/env python

import timeit
import numpy as np
import sys

"""
LESSON LEARNED:
    - given that we want to compute data at the end,
        - writing to a pre fixed size list is close in speed to writing to a numpy array;
        -appending to a list is ~45 time slower then writing to a pre fixed size container;

Experiment result:

    Sizeof (with 250000 items):
        numpy array	2.000096 Mb
        python list	2.000064 Mb
    
    Timeit result:
        5 loops,
    
        Write directly to numpy array then compute:                     5.91130 usec per loop
        Append to list, convert to numpy array then compute:            266.94095 usec per loop
        Write to fixed size list, convert to numpy array then compute:  8.11612 usec per loop
"""

NB_COLLECTED_SAMPLE = 250000
NUMBER_OF_LOOP = 5

BATCH_SIZE = 100

# --- Sizeof -------------------------------------------------------------------------------------------
a1 = np.zeros(NB_COLLECTED_SAMPLE)
b1 = [0] * NB_COLLECTED_SAMPLE

print("\nSizeof (with {} items):\n".format((NB_COLLECTED_SAMPLE)))
print("\t{}\t{} Mb".format("numpy array", sys.getsizeof(a1) * 1e-6))
print("\t{}\t{} Mb".format("python list", sys.getsizeof(b1) * 1e-6))


# --- Speed -------------------------------------------------------------------------------------------


def mod_numpy_array():
    a = np.zeros(NB_COLLECTED_SAMPLE, dtype=int)
    b = np.zeros(NB_COLLECTED_SAMPLE, dtype=int)
    c = np.zeros(NB_COLLECTED_SAMPLE, dtype=int)
    
    for _ in range(BATCH_SIZE):
        for i in range(NB_COLLECTED_SAMPLE):
            a[i] = 1
            b[i] = 1
            c[i] = 1
        
        sumA = np.sum(a)
        sumB = np.sum(b)
        sumC = np.sum(c)
        
        assert sumA == sumB
        assert sumB == sumC


def append_to_list():
    a = list()
    b = list()
    c = list()
    
    for _ in range(BATCH_SIZE):
        for i in range(NB_COLLECTED_SAMPLE):
            a.append(1)
            b.append(1)
            c.append(1)
        
        sumA = np.sum(a)
        sumB = np.sum(b)
        sumC = np.sum(c)
        
        assert sumA == sumB
        assert sumB == sumC


def fix_size_list():
    a = [0] * NB_COLLECTED_SAMPLE
    b = [0] * NB_COLLECTED_SAMPLE
    c = [0] * NB_COLLECTED_SAMPLE
    
    for _ in range(BATCH_SIZE):
        for i in range(NB_COLLECTED_SAMPLE):
            a[i] = 1
            b[i] = 1
            c[i] = 1
        
        sumA = np.sum(a)
        sumB = np.sum(b)
        sumC = np.sum(c)
        
        assert sumA == sumB
        assert sumB == sumC


result = np.zeros(3)

result[0] = timeit.timeit("mod_numpy_array()", number=NUMBER_OF_LOOP, globals=globals())
result[1] = timeit.timeit("append_to_list()", number=NUMBER_OF_LOOP, globals=globals())
result[2] = timeit.timeit("fix_size_list()", number=NUMBER_OF_LOOP, globals=globals())

result = result / NUMBER_OF_LOOP

print("\nTimeit result:\n\t{} loops,\n\n"
      "\tWrite directly to numpy array then compute:                    {:2.5f} usec per loop\n"
      "\tAppend to list, convert to numpy array then compute:           {:2.5f} usec per loop\n"
      "\tWrite to fixed size list, convert to numpy array then compute: {:2.5f} usec per loop\n".format(NUMBER_OF_LOOP,
                                                                                                        result[0],
                                                                                                        result[1],
                                                                                                        result[2]))
