from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")))
import rlskyjo

setup(
    name="rlskyjo",
    packages=find_packages(),
    version=rlskyjo.__version__,
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
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
)
