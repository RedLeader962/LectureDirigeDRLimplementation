# coding=utf-8

import numpy as np


def fct_to_yield():
    a = np.arange(20)

    for each in a:
        yield ("-->", a[each])
        print("we are back")

    yield ("\n>>>>DONE", len(a))

    print("--TARMINÉ!!!!!")


def execute():

    generator = fct_to_yield()

    for each_idx in generator:
        field_A, field_B = each_idx
        print("{} {}".format(field_A, field_B))


if __name__ == "__main__":
    execute()
