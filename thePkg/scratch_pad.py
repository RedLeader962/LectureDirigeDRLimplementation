#!/usr/bin/env python


class myClass:
    name: str

    def __init__(self):
        self.name = 'moi'

    def __str__(self):
        return self.name


def myIncrement(x):
    return x + 1


def myfunct() -> object:
    """
    :rtype: object

    """
    my = myClass()
    people = dict(name='bozo', job='clown')

    return people['name'] + my.__str__()


if __name__ == '__main__':
    print(myfunct())


