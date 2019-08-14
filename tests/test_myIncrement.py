# !/usr/bin/env python

import pytest
from thePkg import scratch_pad


def test_my_increment_pass():
    assert scratch_pad.myIncrement(4) == 5


def test_my_increment_fail():
    with pytest.raises(AssertionError):
        assert scratch_pad.myIncrement(4) == 5

