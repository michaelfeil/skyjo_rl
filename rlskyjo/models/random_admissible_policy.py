from typing import Union

import numpy as np


def policy_ra(
    observation: np.array,
    action_mask: np.array,
    rng: Union[None, np.random.Generator] = None,
) -> int:
    """for demonstration.
    picks randomly an admissible action from the action mask

    Args:
        observation (np.array): [description]
        action_mask (np.array): [description]
        rng (Union[None, np.random.Generator], optional): [description]. Defaults to None.

    Returns:
        int: [description]
    """     
    if rng is None:
        module = np.random
    else:
        module = rng
    return module.choice(
        np.arange(len(action_mask)), p=action_mask / np.sum(action_mask)
    )
