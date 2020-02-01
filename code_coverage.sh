#!/bin/bash

#set -eux
set -ux

# Run coverage package
cd DRLimplementation/
coverage run -m pytest

# Produce report and save to a XML file
coverage report
coverage xml -i

# push report to codecov
cd ..
codecov --token=c443acdd-4ce5-4208-9c0e-017fe30342d7