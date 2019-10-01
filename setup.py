# coding=utf-8

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LectureDirigeDRLimplementation",
    version="0.0.1",
    author="Luc Coupal",
    author_email="luc.coupal.1@ulval.ca",
    description="Directed rearing on Deep Reinforcement Learning",
    url="https://github.com/RedLeader962/LectureDirigeDRLimplementation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)