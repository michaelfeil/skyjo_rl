skyjo_rl
==============================

Multi-Agent Reinforcement Learning Environment for the card game SkyJo, compatible with PettingZoo and RLLIB

[![codecov](https://codecov.io/gh/michaelfeil/skyjo_rl/branch/master/graph/badge.svg?token=56TSLUCER8)](https://codecov.io/gh/michaelfeil/skyjo_rl)![CI pytest](https://github.com/michaelfeil/skyjo_rl/actions/workflows/test_release.yml/badge.svg)

[Read the docs](https://michaelfeil.github.io/skyjo_rl/)

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

Project Organization
------------
[Github Repository](https://github.com/michaelfeil/skyjo_rl)

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project.
    │
    ├── docs                    <- Docs HTMLs, see Sphinx [docs](https:/michaelfeil.github.io/skyjo_rl)
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks               <- Jupyter notebooks. 
    ├── requirements.txt                        <- requirements for the rlskyjo
    ├── requirements_dev.txt                    <- requirements for developers
    ├── rlskyjo                                    
    │   ├── environment
    │   │   ├── skyjo_env.py
    │   │   └── vanilla_env_example.py
    │   ├── game
    │   │   ├── sample_game.py
    │   │   └── skyjo.py
    │   ├── models
    │   │   ├── action_mask_model.py
    │   │   ├── random_admissible_policy.py
    │   │   └── train_model_simple_rllib.py
    │   └── utils.py
    ├── setup.py                                <- makes project pip installable (pip install -e .) so skyjo_rl can be imported
    ├── test_environment.py
    ├── tests                                   <- Unittests
    └── tox.ini                                 <- tox file with settings for running tox; see tox.readthedocs.io

--------
## PYPI Install
```
conda create --name skyjo python=3.8 pip
conda activate skyjo
pip install rlskyjo
```
--------
## Developer Install
```
git clone https://github.com/michaelfeil/skyjo_rl.git
conda create --name skyjo python=3.8 pip
conda activate skyjo
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
pre-commit install
coverage run -m --source=./rlskyjo pytest tests
```

## Tutorials
[Vanilla SkyJo PettingZoo Env example](https://github.com/michaelfeil/skyjo_rl/blob/master/rlskyjo/environment/vanilla_env_example.py)

[SkyJo game example](https://github.com/michaelfeil/skyjo_rl/blob/master/rlskyjo/game/sample_game.py)

[Train PPO MultiAgent with SkyJo PettingZoo Env, Pytorch and RLLib](https://github.com/michaelfeil/skyjo_rl/blob/master/rlskyjo/models/train_model_simple_rllib.py)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/michaelfeil/skyjo_rl.svg?style=for-the-badge
[contributors-url]: https://github.com/michaelfeil/skyjo_rl/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/michaelfeil/skyjo_rl.svg?style=for-the-badge
[forks-url]: https://github.com/michaelfeil/skyjo_rl/network/members
[stars-shield]: https://img.shields.io/github/stars/michaelfeil/skyjo_rl.svg?style=for-the-badge
[stars-url]: https://github.com/michaelfeil/skyjo_rl/stargazers
[issues-shield]: https://img.shields.io/github/issues/michaelfeil/skyjo_rl.svg?style=for-the-badge
[issues-url]: https://github.com/michaelfeil/skyjo_rl/issues
[license-shield]: https://img.shields.io/github/license/michaelfeil/skyjo_rl.svg?style=for-the-badge
[license-url]: https://github.com/michaelfeil/skyjo_rl/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/michael-feil
