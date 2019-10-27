# coding=utf-8
from typing import Tuple
from collections import namedtuple
import numpy as np
import pytest
from blocAndTools.temporal_difference_computation import computhe_the_Advantage, compute_TD_target, get_t_and_tPrime_array_view_for_element_wise_op

key = ['arrayOne', 'mock_V_estimate', 'array_tPrime', 'array_t', 'expected_target', 'expected_advantage']
SetUpFixture = namedtuple('SetUpFixture', key)

@pytest.fixture
def setUpDecr() -> SetUpFixture:
    mock_array = SetUpFixture(
        arrayOne=[1 for _ in range(5)],
        mock_V_estimate=[5, 4, 3, 2, 1],
        array_t=np.array([5, 4, 3, 2, 1]),
        array_tPrime=np.array([4, 3, 2, 1, 0]),
        expected_target=np.array([5, 4, 3, 2, 1]),
        expected_advantage=np.zeros(5)
    )
    return mock_array

def test_namedtupleSetup_PASS(setUpDecr):
    assert np.equal(setUpDecr.array_t, np.array([5, 4, 3, 2, 1])).all()

def test_get_t_and_tPrime_array_view_for_element_wise_op(setUpDecr):

    view_t, view_tPrime = get_t_and_tPrime_array_view_for_element_wise_op(setUpDecr.mock_V_estimate)

    assert np.equal(view_t, setUpDecr.array_t).all(), "{} != {}".format(view_t, setUpDecr.array_t)
    assert np.equal(view_tPrime, setUpDecr.array_tPrime).all(), "{} != {}".format(view_tPrime, setUpDecr.array_tPrime)


def test_compute_TD_target_COMPUTE_PASS(setUpDecr):

    target = compute_TD_target(setUpDecr.arrayOne, setUpDecr.mock_V_estimate, 1)

    assert np.equal(target, setUpDecr.expected_target).all(), "{} != {}".format(target, setUpDecr.expected_target)


def test_compute_TD_target_LEN_PASS(setUpDecr):

    target = compute_TD_target(setUpDecr.arrayOne, setUpDecr.mock_V_estimate, 1)

    assert len(setUpDecr.arrayOne) == len(target)


def test_computhe_the_Advantage_COMPUTE_PASS(setUpDecr):

    Advantage = computhe_the_Advantage(setUpDecr.arrayOne, setUpDecr.mock_V_estimate)
    assert np.equal(setUpDecr.expected_advantage, Advantage).all(), "{} != {}".format(setUpDecr.expected_advantage, Advantage)

def test_computhe_the_Advantage_LEN_PASS(setUpDecr):

    Advantage = computhe_the_Advantage(setUpDecr.arrayOne, setUpDecr.mock_V_estimate)
    assert len(setUpDecr.arrayOne) == len(Advantage)
