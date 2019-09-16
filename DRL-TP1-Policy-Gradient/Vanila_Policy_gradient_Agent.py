# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

from DRL_building_bloc import Playground

import tensorflow as tf
from tensorflow import keras





if __name__ == '__main__':
    play = Playground()

    print(">>> Environment {} ready to go\n".format(play.env))
