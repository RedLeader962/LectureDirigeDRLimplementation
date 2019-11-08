#!/usr/bin/env python

import timeit
import numpy as np
import sys
import time


"""
LESSON LEARNED: 
    - numpy array indexing (READ) is 2X slower compare to (READ) on same size list
    - overrided method are a little bit slower to execute than a original one
    - execution behave differently with respect to container size (size: 4000 vs 4000000)
    - result change drastically if TensorFlow is running in a other python process

Experiment result: 

    === 4 000 000 items ==========================================================================
    
    --- Timeit WRITE result (with 4000000 items) -------------------------------------------------
        250 step per episode, over 20 timeit loops
    
        populate_numpy_array: 											8.24359 usec per loop
        populate_append_to_list: 										13.24392 usec per loop
        populate_fix_size_list: 										10.36419 usec per loop
        populate_EpisodeData(EpisodeData): 								3.31320 usec per loop
        populate_EpisodeData(EpisodeDataFixSizeList): 					2.65059 usec per loop
    
        Overrided (WRITE):
        overrided_populate_EpisodeData(EpisodeData): 					3.66189 usec per loop
        overrided_populate_EpisodeData(EpisodeDataFixSizeList): 		2.91738 usec per loop
    
    
    --- Timeit READ result (with 4000000 items) --------------------------------------------------
        250 step per episode, over 20 timeit loops
    
        numpy_array_container: 											72.16929 usec per loop
        append_to_list: 												24.34082 usec per loop
        fix_size_list: 													23.58640 usec per loop
        read_and_write_EpisodeData(EpisodeData): 						28.45965 usec per loop
        read_and_write_EpisodeData(EpisodeDataFixSizeList): 			28.79260 usec per loop
    
        Overrided (READ):
        overrided_read_and_write_EpisodeData(EpisodeData): 				26.65566 usec per loop
        overrided_read_and_write_EpisodeData(EpisodeDataFixSizeList): 	26.11270 usec per loop
    
    
    --- Timeit READ&WRITE result (with 4000000 items) ---------------------------------------------
        250 step per episode, over 20 timeit loops
    
        numpy_array_container: 											80.41288 usec per loop
        append_to_list: 												37.58474 usec per loop
        fix_size_list: 													33.95059 usec per loop
        read_and_write_EpisodeData(EpisodeData): 						31.77285 usec per loop
        read_and_write_EpisodeData(EpisodeDataFixSizeList): 			31.44319 usec per loop
    
        Overrided (READ):
        overrided_read_and_write_EpisodeData(EpisodeData): 				30.31756 usec per loop
        overrided_read_and_write_EpisodeData(EpisodeDataFixSizeList): 	29.03008 usec per loop
    
    
    --- Sizeof (with 4000000 items) --------------------------------------------------------------
        numpy array							512.00011 Mb
        python list							512.00102 Mb
        EpisodeData							1287.36000 Mb
        EpisodeDataFixSizeList				1282.49600 Mb
        

    
"""

OBS = np.array([1, 2, 3, 4, 5, 6, 7, 8])
Q_VALUES = np.array([1, 2, 3, 4, 5, 6])

# === Speed ============================================================================================================

# --- experiment 4 -----------------------------------------------------------------------------------------------------


class EpisodeData(object):

    def __init__(self):
        # self.episode_id = int()
        self.episode_nb_of_step = 0
        self.cumulated_reward = list()
        self.reward_delta = list()
        self.obs = list()
        self.q_values = list()

    def feed_step_data(self, episode_step_idx, cumulated_reward, reward_delta, obs, q_values):
        self.episode_nb_of_step += 1
        self.cumulated_reward.append(cumulated_reward)
        self.reward_delta.append(reward_delta)
        self.obs.append(obs)
        self.q_values.append(q_values)
        return None

    def __setitem__(self, episode_step_idx, values):
        (cumulated_reward, reward_delta, obs, q_values) = values
        self.episode_nb_of_step += 1
        self.cumulated_reward.append(cumulated_reward)
        self.reward_delta.append(reward_delta)
        self.obs.append(obs)
        self.q_values.append(q_values)
        return None

    def read_step_data(self, idx):
        return self.cumulated_reward[:idx+1], self.reward_delta[:idx+1], self.obs[idx], self.q_values[idx]

    def __getitem__(self, item):
        return self.cumulated_reward[:item+1], self.reward_delta[:item+1], self.obs[item], self.q_values[item]


class EpisodeDataFixSizeList(object):

    def __init__(self):
        # self.episode_id = int()
        self.episode_nb_of_step = 0
        self.cumulated_reward = [0]*MAX_STEP_PER_EPISODE
        self.reward_delta = [0]*MAX_STEP_PER_EPISODE
        self.obs = [0]*MAX_STEP_PER_EPISODE
        self.q_values = [0]*MAX_STEP_PER_EPISODE

    def feed_step_data(self, episode_step_idx, cumulated_reward, reward_delta, obs, q_values):
        self.episode_nb_of_step += 1
        self.cumulated_reward[episode_step_idx] = cumulated_reward
        self.reward_delta[episode_step_idx] = reward_delta
        self.obs[episode_step_idx] = obs
        self.q_values[episode_step_idx] = q_values
        return None

    def __setitem__(self, episode_step_idx, values):
        (cumulated_reward, reward_delta, obs, q_values) = values
        self.episode_nb_of_step += 1
        self.cumulated_reward[episode_step_idx] = cumulated_reward
        self.reward_delta[episode_step_idx] = reward_delta
        self.obs[episode_step_idx] = obs
        self.q_values[episode_step_idx] = q_values
        return None

    def read_step_data(self, idx):
        return self.cumulated_reward[:idx+1], self.reward_delta[:idx+1], self.obs[idx], self.q_values[idx]

    def __getitem__(self, item):
        return self.cumulated_reward[:item+1], self.reward_delta[:item+1], self.obs[item], self.q_values[item]


def read_and_write_EpisodeData(the_dataclass):
    dataclass_container = populate_EpisodeData(the_dataclass)

    len_dataclass_container = len(dataclass_container)
    max_nb_episode = round(MAX_TOTAL_STEP / MAX_STEP_PER_EPISODE)
    assert len_dataclass_container == max_nb_episode, "{} == {}".format(len_dataclass_container, max_nb_episode)

    # read
    for each_episode in dataclass_container:
        for idx in range(each_episode.episode_nb_of_step):
            cumulated_reward, reward_delta, obs, q_values = each_episode.read_step_data(idx)

            assert cumulated_reward == [*range(idx+1)], "{} == {}".format(cumulated_reward, [*range(idx+1)])
            assert reward_delta == [9] * (idx+1), "{} == {}".format(reward_delta, [9] * (idx+1))
            assert obs[0] == 1
            assert obs[1] == 2
            assert obs[2] == 3
            assert obs[3] == 4
            assert obs[4] == 5
            assert obs[5] == 6
            assert obs[6] == 7
            assert obs[7] == 8
            assert q_values[0] == 1
            assert q_values[1] == 2
            assert q_values[2] == 3
            assert q_values[3] == 4
            assert q_values[4] == 5
            assert q_values[5] == 6


def populate_EpisodeData(the_dataclass):

    dataclass_container = list()
    # write
    dataclass_instance = the_dataclass()
    episode_step_idx = 0
    for total_step in range(1, MAX_TOTAL_STEP + 1):

        dataclass_instance.feed_step_data(episode_step_idx, cumulated_reward=episode_step_idx, reward_delta=9, obs=OBS, q_values=Q_VALUES)
        episode_step_idx += 1

        if total_step % (MAX_STEP_PER_EPISODE) == 0:
            # save the dataclass and create a new one
            dataclass_container.append(dataclass_instance)
            dataclass_instance = the_dataclass()
            episode_step_idx = 0


    return dataclass_container

# - - experiment 4 (method override) - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def overrided_read_and_write_EpisodeData(the_dataclass):
    dataclass_container = overrided_populate_EpisodeData(the_dataclass)

    len_dataclass_container = len(dataclass_container)
    max_nb_episode = round(MAX_TOTAL_STEP / MAX_STEP_PER_EPISODE)
    assert len_dataclass_container == max_nb_episode, "{} == {}".format(len_dataclass_container, max_nb_episode)

    # read
    for each_episode in dataclass_container:
        for idx in range(each_episode.episode_nb_of_step):
            cumulated_reward, reward_delta, obs, q_values = each_episode[idx]

            assert cumulated_reward == [*range(idx + 1)], "{} == {}".format(cumulated_reward, [*range(idx + 1)])
            assert reward_delta == [9] * (idx + 1), "{} == {}".format(reward_delta, [9] * (idx + 1))
            assert obs[0] == 1
            assert obs[1] == 2
            assert obs[2] == 3
            assert obs[3] == 4
            assert obs[4] == 5
            assert obs[5] == 6
            assert obs[6] == 7
            assert obs[7] == 8
            assert q_values[0] == 1
            assert q_values[1] == 2
            assert q_values[2] == 3
            assert q_values[3] == 4
            assert q_values[4] == 5
            assert q_values[5] == 6


def overrided_populate_EpisodeData(the_dataclass):

    dataclass_container = list()
    # write
    dataclass_instance = the_dataclass()
    episode_step_idx = 0
    for total_step in range(1, MAX_TOTAL_STEP + 1):

        # episode_step_idx, reward_delta=9, obs=OBS, q_values=Q_VALUES
        dataclass_instance[episode_step_idx] = (episode_step_idx, 9, OBS, Q_VALUES)
        episode_step_idx += 1

        if total_step % (MAX_STEP_PER_EPISODE) == 0:
            # save the dataclass and create a new one
            dataclass_container.append(dataclass_instance)
            dataclass_instance = the_dataclass()
            episode_step_idx = 0


    return dataclass_container


# --- experiment 1 -----------------------------------------------------------------------------------------------------

def numpy_array_container():

    a = populate_numpy_array()

    # read
    for episode_start in range(0, MAX_TOTAL_STEP, MAX_STEP_PER_EPISODE):
        for step in range(MAX_STEP_PER_EPISODE):
            idx = episode_start + step

            assert np.array_equal(a[0, episode_start:idx], np.arange(1, step + 1)), \
                "{} == {}".format(a[0, episode_start:idx], np.arange(1, step + 1))
            assert np.array_equal(a[1, episode_start:idx], np.full(step, 9)), \
                "{} == {}".format(a[1, episode_start:idx], np.full(step, 9))

            # assert a[0, idx] == 1
            # assert a[1, idx] == 9

            assert a[2, idx] == 1
            assert a[3, idx] == 2
            assert a[4, idx] == 3
            assert a[5, idx] == 4
            assert a[6, idx] == 5
            assert a[7, idx] == 6
            assert a[8, idx] == 7
            assert a[9, idx] == 8
            assert a[10, idx] == 1
            assert a[11, idx] == 2
            assert a[12, idx] == 3
            assert a[13, idx] == 4
            assert a[14, idx] == 5
            assert a[15, idx] == 6


def populate_numpy_array():
    a = np.zeros((16, MAX_TOTAL_STEP), dtype=int)
    episode_nb_of_step = 1
    # write
    for total_step in range(MAX_TOTAL_STEP):


        a[0:2, total_step] = [episode_nb_of_step, 9]
        a[2:10, total_step] = OBS
        a[10:, total_step] = Q_VALUES

        episode_nb_of_step += 1

        # Simulate the added cost of computing logic of EpisodeData
        if (total_step + 1) % MAX_STEP_PER_EPISODE == 0:
            episode_nb_of_step = 1
    return a


# --- experiment 2 -----------------------------------------------------------------------------------------------------

def append_to_list():
    a, b, c, d, e, f, g, h, i, j, q1, q2, q3, q4, q5, q6 = populate_append_to_list()

    # read
    for episode_start in range(0, MAX_TOTAL_STEP, MAX_STEP_PER_EPISODE):
        for step in range(MAX_STEP_PER_EPISODE):
            idx = episode_start + step

            assert a[episode_start:idx] == [*range(1, step + 1)], "{} == {}".format(a[episode_start:idx + 1], [*range(1, step + 1)])
            assert b[episode_start:idx] == [9] * step, "{} == {}".format(b[episode_start:idx], [9] * step)

            # assert a[idx] == 1
            # assert b[idx] == 9

            assert c[idx] == 1
            assert d[idx] == 2
            assert e[idx] == 3
            assert f[idx] == 4
            assert g[idx] == 5
            assert h[idx] == 6
            assert i[idx] == 7
            assert j[idx] == 8
            assert q1[idx] == 1
            assert q2[idx] == 2
            assert q3[idx] == 3
            assert q4[idx] == 4
            assert q5[idx] == 5
            assert q6[idx] == 6


def populate_append_to_list():
    episode_nb_of_step = 1
    a = list()
    b = list()
    c = list()
    d = list()
    e = list()
    f = list()
    g = list()
    h = list()
    i = list()
    j = list()
    q1 = list()
    q2 = list()
    q3 = list()
    q4 = list()
    q5 = list()
    q6 = list()

    # write
    for total_step in range(MAX_TOTAL_STEP):

        a.append(episode_nb_of_step)
        b.append(9)
        c.append(OBS[0])
        d.append(OBS[1])
        e.append(OBS[2])
        f.append(OBS[3])
        g.append(OBS[4])
        h.append(OBS[5])
        i.append(OBS[6])
        j.append(OBS[7])
        q1.append(Q_VALUES[0])
        q2.append(Q_VALUES[1])
        q3.append(Q_VALUES[2])
        q4.append(Q_VALUES[3])
        q5.append(Q_VALUES[4])
        q6.append(Q_VALUES[5])

        episode_nb_of_step += 1

        # Simulate the added cost of computing logic of EpisodeData
        if (total_step + 1) % MAX_STEP_PER_EPISODE == 0:
            episode_nb_of_step = 1
    return a, b, c, d, e, f, g, h, i, j, q1, q2, q3, q4, q5, q6


# --- experiment 3 -----------------------------------------------------------------------------------------------------

def fix_size_list():
    a, b, c, d, e, f, g, h, i, j, q1, q2, q3, q4, q5, q6 = populate_fix_size_list()

    # read
    for episode_start in range(0, MAX_TOTAL_STEP, MAX_STEP_PER_EPISODE):
        for step in range(MAX_STEP_PER_EPISODE):
            idx = episode_start + step

            assert a[episode_start:idx] == [*range(1, step + 1)], "{} == {}".format(a[episode_start:idx + 1], [*range(1, step + 1)])
            assert b[episode_start:idx] == [9] * step, "{} == {}".format(b[episode_start:idx], [9] * step)

            # assert a[idx] == 1
            # assert b[idx] == 9

            assert c[idx] == 1
            assert d[idx] == 2
            assert e[idx] == 3
            assert f[idx] == 4
            assert g[idx] == 5
            assert h[idx] == 6
            assert i[idx] == 7
            assert j[idx] == 8
            assert q1[idx] == 1
            assert q2[idx] == 2
            assert q3[idx] == 3
            assert q4[idx] == 4
            assert q5[idx] == 5
            assert q6[idx] == 6


def populate_fix_size_list():
    episode_nb_of_step = 1
    a = [0] * MAX_TOTAL_STEP
    b = [0] * MAX_TOTAL_STEP
    c = [0] * MAX_TOTAL_STEP
    d = [0] * MAX_TOTAL_STEP
    e = [0] * MAX_TOTAL_STEP
    f = [0] * MAX_TOTAL_STEP
    g = [0] * MAX_TOTAL_STEP
    h = [0] * MAX_TOTAL_STEP
    i = [0] * MAX_TOTAL_STEP
    j = [0] * MAX_TOTAL_STEP
    q1 = [0] * MAX_TOTAL_STEP
    q2 = [0] * MAX_TOTAL_STEP
    q3 = [0] * MAX_TOTAL_STEP
    q4 = [0] * MAX_TOTAL_STEP
    q5 = [0] * MAX_TOTAL_STEP
    q6 = [0] * MAX_TOTAL_STEP
    # write
    for total_step in range(MAX_TOTAL_STEP):

        a[total_step] = episode_nb_of_step
        b[total_step] = 9
        c[total_step] = OBS[0]
        d[total_step] = OBS[1]
        e[total_step] = OBS[2]
        f[total_step] = OBS[3]
        g[total_step] = OBS[4]
        h[total_step] = OBS[5]
        i[total_step] = OBS[6]
        j[total_step] = OBS[7]

        q1[total_step] = Q_VALUES[0]
        q2[total_step] = Q_VALUES[1]
        q3[total_step] = Q_VALUES[2]
        q4[total_step] = Q_VALUES[3]
        q5[total_step] = Q_VALUES[4]
        q6[total_step] = Q_VALUES[5]

        episode_nb_of_step += 1

        # Simulate the added cost of computing logic of EpisodeData
        if (total_step + 1) % MAX_STEP_PER_EPISODE == 0:
            episode_nb_of_step = 1
    return a, b, c, d, e, f, g, h, i, j, q1, q2, q3, q4, q5, q6


# ----------------------------------------------------------------------------------------------------------------------

class ToConsolTracker:
    def __init__(self, message):
        self.i = 1
        self.message = message

    def __call__(self):
        if self.i != 1:
            print("\r>>> Running {} timeit experiment --> {}".format(self.message, self.i), end="")
        else:
            print("\n>>> Running {} timeit experiment --> {}".format(self.message, self.i), end="")

        print(
            "EpisodeDataClass prof of concept and benchmark: {} --> casse {}!".format(self.message, self.i))

        time.sleep(0.3)
        self.i += 1

    def done(self):
        print("\r ")
        print(
            "EpisodeDataClass prof of concept and benchmark: {} --> DONE!".format(self.message))


def benchmark_data_structure(config):
    global MAX_TOTAL_STEP
    global NUMBER_OF_LOOP
    global MAX_STEP_PER_EPISODE

    MAX_TOTAL_STEP = config['MAX_TOTAL_STEP']
    NUMBER_OF_LOOP = config['NUMBER_OF_LOOP']
    MAX_STEP_PER_EPISODE = config['MAX_STEP_PER_EPISODE']

    assert MAX_TOTAL_STEP >= MAX_STEP_PER_EPISODE
    assert (MAX_TOTAL_STEP % MAX_STEP_PER_EPISODE) == 0

    _run_write_exp_message = ToConsolTracker("WRITE")
    _run_write_exp_message()

    write_results = np.zeros(7)
    write_results[0] = timeit.timeit("populate_numpy_array()", number=NUMBER_OF_LOOP, globals=globals())
    _run_write_exp_message()
    write_results[1] = timeit.timeit("populate_append_to_list()", number=NUMBER_OF_LOOP, globals=globals())
    _run_write_exp_message()
    write_results[2] = timeit.timeit("populate_fix_size_list()", number=NUMBER_OF_LOOP, globals=globals())
    _run_write_exp_message()
    write_results[3] = timeit.timeit("populate_EpisodeData(EpisodeData)", number=NUMBER_OF_LOOP, globals=globals())
    _run_write_exp_message()
    write_results[4] = timeit.timeit("populate_EpisodeData(EpisodeDataFixSizeList)", number=NUMBER_OF_LOOP, globals=globals())
    _run_write_exp_message()
    write_results[5] = timeit.timeit("overrided_populate_EpisodeData(EpisodeData)", number=NUMBER_OF_LOOP, globals=globals())
    _run_write_exp_message()
    write_results[6] = timeit.timeit("overrided_populate_EpisodeData(EpisodeDataFixSizeList)", number=NUMBER_OF_LOOP, globals=globals())
    _run_write_exp_message.done()
    write_results = write_results / NUMBER_OF_LOOP

    _run_readwrite_exp_message = ToConsolTracker("READ&WRITE")
    _run_readwrite_exp_message()

    read_and_write_results = np.zeros(7)
    read_and_write_results[0] = timeit.timeit("numpy_array_container()", number=NUMBER_OF_LOOP, globals=globals())
    _run_readwrite_exp_message()
    read_and_write_results[1] = timeit.timeit("append_to_list()", number=NUMBER_OF_LOOP, globals=globals())
    _run_readwrite_exp_message()
    read_and_write_results[2] = timeit.timeit("fix_size_list()", number=NUMBER_OF_LOOP, globals=globals())
    _run_readwrite_exp_message()
    read_and_write_results[3] = timeit.timeit("read_and_write_EpisodeData(EpisodeData)", number=NUMBER_OF_LOOP, globals=globals())
    _run_readwrite_exp_message()
    read_and_write_results[4] = timeit.timeit("read_and_write_EpisodeData(EpisodeDataFixSizeList)", number=NUMBER_OF_LOOP, globals=globals())
    _run_readwrite_exp_message()
    read_and_write_results[5] = timeit.timeit("overrided_read_and_write_EpisodeData(EpisodeData)", number=NUMBER_OF_LOOP, globals=globals())
    _run_readwrite_exp_message()
    read_and_write_results[6] = timeit.timeit("overrided_read_and_write_EpisodeData(EpisodeDataFixSizeList)", number=NUMBER_OF_LOOP, globals=globals())
    _run_readwrite_exp_message.done()
    read_and_write_results = read_and_write_results / NUMBER_OF_LOOP

    read_results = np.zeros(7)
    read_results[0] = read_and_write_results[0] - write_results[0]
    read_results[1] = read_and_write_results[1] - write_results[1]
    read_results[2] = read_and_write_results[2] - write_results[2]
    read_results[3] = read_and_write_results[3] - write_results[3]
    read_results[4] = read_and_write_results[4] - write_results[4]
    read_results[5] = read_and_write_results[5] - write_results[5]
    read_results[6] = read_and_write_results[6] - write_results[6]

    message_write_str = "\n\n--- Timeit WRITE result (with {} items) {}\n\t{} step per episode, over {} timeit loops\n\n" \
                        "\tpopulate_numpy_array: \t\t\t\t\t\t\t\t\t\t\t{:2.5f} usec per loop\n" \
                        "\tpopulate_append_to_list: \t\t\t\t\t\t\t\t\t\t{:2.5f} usec per loop\n" \
                        "\tpopulate_fix_size_list: \t\t\t\t\t\t\t\t\t\t{:2.5f} usec per loop\n" \
                        "\tpopulate_EpisodeData(EpisodeData): \t\t\t\t\t\t\t\t{:2.5f} usec per loop\n" \
                        "\tpopulate_EpisodeData(EpisodeDataFixSizeList): \t\t\t\t\t{:2.5f} usec per loop\n\n" \
                        "\tOverrided (WRITE):\n" \
                        "\toverrided_populate_EpisodeData(EpisodeData): \t\t\t\t\t{:2.5f} usec per loop\n" \
                        "\toverrided_populate_EpisodeData(EpisodeDataFixSizeList): \t\t{:2.5f} usec per loop\n" \
                        "".format(MAX_TOTAL_STEP, "-"*49, MAX_STEP_PER_EPISODE,
                                  NUMBER_OF_LOOP,
                                  write_results[0],
                                  write_results[1],
                                  write_results[2],
                                  write_results[3],
                                  write_results[4],
                                  write_results[5],
                                  write_results[6])

    read_write_with_hole = "\n--- Timeit {} result (with {} items) {}\n\t{} step per episode, over {} timeit loops\n\n" \
                           "\tnumpy_array_container: \t\t\t\t\t\t\t\t\t\t\t{:2.5f} usec per loop\n" \
                           "\tappend_to_list: \t\t\t\t\t\t\t\t\t\t\t\t{:2.5f} usec per loop\n" \
                           "\tfix_size_list: \t\t\t\t\t\t\t\t\t\t\t\t\t{:2.5f} usec per loop\n" \
                           "\tread_and_write_EpisodeData(EpisodeData): \t\t\t\t\t\t{:2.5f} usec per loop\n" \
                           "\tread_and_write_EpisodeData(EpisodeDataFixSizeList): \t\t\t{:2.5f} usec per loop\n\n" \
                           "\tOverrided (READ):\n" \
                           "\toverrided_read_and_write_EpisodeData(EpisodeData): \t\t\t\t{:2.5f} usec per loop\n" \
                           "\toverrided_read_and_write_EpisodeData(EpisodeDataFixSizeList): \t{:2.5f} usec per loop\n"


    message_read_str = read_write_with_hole.format("READ", MAX_TOTAL_STEP, "-" * 50, MAX_STEP_PER_EPISODE,
                                                   NUMBER_OF_LOOP,
                                                   read_results[0],
                                                   read_results[1],
                                                   read_results[2],
                                                   read_results[3],
                                                   read_results[4],
                                                   read_results[5],
                                                   read_results[6])

    message_read_and_write_str = read_write_with_hole.format("READ&WRITE", MAX_TOTAL_STEP, "-" * 45,
                                                             MAX_STEP_PER_EPISODE,
                                                             NUMBER_OF_LOOP,
                                                             read_and_write_results[0],
                                                             read_and_write_results[1],
                                                             read_and_write_results[2],
                                                             read_and_write_results[3],
                                                             read_and_write_results[4],
                                                             read_and_write_results[5],
                                                             read_and_write_results[6])

    print(message_write_str)
    print(message_read_str)
    print(message_read_and_write_str)

    # === Sizeof =======================================================================================================

    _run_size_exp_message = ToConsolTracker("SizeOf")
    _run_size_exp_message()

    numpy_array_size = sys.getsizeof(np.zeros((16, MAX_TOTAL_STEP), dtype=int))

    _run_size_exp_message()
    size_of_lists = sum([sys.getsizeof([0] * MAX_TOTAL_STEP) for _ in range(16)])

    def sizeof_a_dataclass_instances(the_dataclass_instance):
        sizeof_obs = sum([sys.getsizeof(obs) for obs in the_dataclass_instance.obs])
        sizeof_qvalues = sum([sys.getsizeof(q_values) for q_values in the_dataclass_instance.q_values])

        return sum([sys.getsizeof(the_dataclass_instance.reward_delta),
                    sys.getsizeof(the_dataclass_instance.cumulated_reward),
                    sys.getsizeof(the_dataclass_instance.episode_nb_of_step),
                    sizeof_obs,
                    sizeof_qvalues])

    def sizeof_dataclass_instances(the_dataclass, test_nb_of_step_in_dataclass=False):
        dataclass_instance = populate_EpisodeData(the_dataclass)

        dataclass_size = sum([sizeof_a_dataclass_instances(inst) for inst in dataclass_instance])

        if test_nb_of_step_in_dataclass:
            total_nb_of_step_in_dataclass_instances = sum([inst.episode_nb_of_step for inst in dataclass_instance])
            assert MAX_TOTAL_STEP == total_nb_of_step_in_dataclass_instances, \
                "received {} == {}".format(MAX_TOTAL_STEP, total_nb_of_step_in_dataclass_instances)

        return dataclass_size

    _run_size_exp_message()
    MyDataClass_size = sizeof_dataclass_instances(EpisodeData, test_nb_of_step_in_dataclass=True)
    _run_size_exp_message()
    MyDataClassFullSizeList_size = sizeof_dataclass_instances(EpisodeDataFixSizeList)
    _run_size_exp_message.done()

    message_sizeof_str = "".join(["\n--- Sizeof (with {} items) {}\n".format(MAX_TOTAL_STEP, "-"*62),
                                  "\t{}\t\t\t\t\t\t\t{:2.5f} Mb\n".format("numpy array", numpy_array_size*1e-6),
                                  "\t{}\t\t\t\t\t\t\t{:2.5f} Mb\n".format("python list", size_of_lists * 1e-6),
                                  "\t{}\t\t\t\t\t\t\t{:2.5f} Mb\n".format("EpisodeData", MyDataClass_size*1e-6),
                                  "\t{}\t\t\t\t{:2.5f} Mb\n\n".format("EpisodeDataFixSizeList",
                                                                      MyDataClassFullSizeList_size*1e-6)])

    print(message_sizeof_str)

    file_name = "data_structure_benchmark-{}step-{}loop-{}max.txt".format(MAX_TOTAL_STEP,
                                                                          NUMBER_OF_LOOP,
                                                                          MAX_STEP_PER_EPISODE)
    with open(file_name, "w") as f:
        f.writelines(message_write_str)
        f.writelines(message_read_str)
        f.writelines(message_read_and_write_str)
        f.writelines(message_sizeof_str)

    _run_size_exp_message.notification.push_notification(
                "".join(["EpisodeDataClass prof of concept and benchmark >>> ALL DONE!\n",
                         message_write_str, message_read_str, message_read_and_write_str, message_sizeof_str]))

    return None


if __name__ == '__main__':

    debug_config = {
        'MAX_TOTAL_STEP': 5000,
        'NUMBER_OF_LOOP': 5,
        'MAX_STEP_PER_EPISODE': 10,
    }

    experiment_1_config = {
        'MAX_TOTAL_STEP': 1000000,
        'NUMBER_OF_LOOP': 20,
        'MAX_STEP_PER_EPISODE': 250,
    }

    experiment_2_config = {
        'MAX_TOTAL_STEP': 1000000,
        'NUMBER_OF_LOOP': 20,
        'MAX_STEP_PER_EPISODE': 500,
    }

    experiment_3_config = {
        'MAX_TOTAL_STEP': 4000000,
        'NUMBER_OF_LOOP': 20,
        'MAX_STEP_PER_EPISODE': 250,
    }

    # benchmark_data_structure(debug_config)
    benchmark_data_structure(experiment_1_config)
    benchmark_data_structure(experiment_2_config)
    benchmark_data_structure(experiment_3_config)
