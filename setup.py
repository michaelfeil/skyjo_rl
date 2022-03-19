from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

import os
import sys

def _get_version():
    with open('rlskyjo/__init__.py') as fp:
        for line in fp:
            if line.startswith('__version__') and '=' in line:
                version = line[line.find('=')+1:].strip(' \'"\n')
                if version:
                    return version
        raise ValueError('`__version__` not defined in `rlskyjo/__init__.py`')

setup(
    name="rlskyjo",
    packages=find_packages(),
    version=_get_version(),
    description="Multi-Agent Reinforcement Learning Environment"
    " for the card game SkyJo, compatible with PettingZoo and RLLIB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michael Feil",
    license="MIT",
    url="https://github.com/michaelfeil/skyjo_rl",
    project_urls={
        "Bug Tracker": "https://github.com/michaelfeil/skyjo_rl/issues",
    },
    install_requires=[
        "gym",
        "numba",
        "numpy",
        "PettingZoo",
        "ray[tune]",
        "ray[rllib]",
        "setuptools",
        "torch",
    ],
    extras_require={'numba': ['numba']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
