# coding=utf-8

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LectureDirigeDRLimplementation",
    version="0.0.1",
    author="Luc Coupal",
    author_email="luc.coupal.1@ulval.ca",
    description="Directed reading on Deep Reinforcement Learning",
    url="https://github.com/RedLeader962/LectureDirigeDRLimplementation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
            'gym[atari,box2d,classic_control]>=0.14.0',
            'ipython',
            'joblib',
            'matplotlib>=3.1.0',
            'numpy>=1.16.4',
            'pandas',
            'pytest',
            'psutil',
            'scipy',
            'seaborn>=0.9.0',
            'tensorflow>=1.14.0,<2.0',
        ],
    python_requires='>=3.7',
)