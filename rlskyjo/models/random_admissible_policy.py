import numpy as np


def random_admissible_policy(observation: np.array, action_mask: np.array) -> int:
    """for demonstration.
    picks randomly an admissible action from the action mask"""
    return np.random.choice(
        np.arange(len(action_mask)), p=action_mask / np.sum(action_mask)
    )
