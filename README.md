skyjo_rl
==============================

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

A RL repo for playing the card game 'SkyBo'

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so rlskyjo can be imported
    ├── rlskyjo           <- Source code for use in this project.
    │   ├── __init__.py    <- Makes rlskyjo a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Install
```
conda create --name  skybo python=3.9
conda activate skybo
pip install -r requirements.txt
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/michaelfeil/rlskyjo.svg?style=for-the-badge
[contributors-url]: https://github.com/michaelfeil/rlskyjo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/michaelfeil/rlskyjo.svg?style=for-the-badge
[forks-url]: https://github.com/michaelfeil/rlskyjo/network/members
[stars-shield]: https://img.shields.io/github/stars/michaelfeil/rlskyjo.svg?style=for-the-badge
[stars-url]: https://github.com/michaelfeil/rlskyjo/stargazers
[issues-shield]: https://img.shields.io/github/issues/michaelfeil/rlskyjo.svg?style=for-the-badge
[issues-url]: https://github.com/michaelfeil/rlskyjo/issues
[license-shield]: https://img.shields.io/github/license/michaelfeil/rlskyjo.svg?style=for-the-badge
[license-url]: https://github.com/michaelfeil/rlskyjo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/michael-feil