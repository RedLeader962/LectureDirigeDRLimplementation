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
    install_requires=[
        'cloudpickle==1.2.1',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'ipython',
        'joblib',
        'matplotlib==3.1.0',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'tensorflow>=1.14.0,<2.0',
        'tqdm'
    ],
    python_requires='>=3.7',
)