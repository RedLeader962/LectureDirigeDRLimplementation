# !/usr/bin/env python


from thePkg import scratch_pad


def test_myIncrement_pass():
    assert scratch_pad.myIncrement(4) == 5

def test_myIncrement_fail():
    assert scratch_pad.myIncrement(4) == 4