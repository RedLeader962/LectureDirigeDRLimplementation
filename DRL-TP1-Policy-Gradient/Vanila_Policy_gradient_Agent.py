# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

from DRL_building_bloc import GymPlayground

import tensorflow as tf
from tensorflow import keras





if __name__ == '__main__':
    play = GymPlayground()

    print(">>> Environment {} ready to go\n".format(play.env))
