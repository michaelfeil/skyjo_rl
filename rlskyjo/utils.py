from pathlib import Path


def get_project_root() -> Path:
    """
    return Path to the project directory, top folder of rlskyjo
    """
    return Path(__file__).parent.parent.resolve()


print(get_project_root())
