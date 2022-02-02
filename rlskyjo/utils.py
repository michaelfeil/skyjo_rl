from pathlib import Path
from typing import Union
import glob
import os
import re

def get_project_root() -> Path:  
    """return Path to the project directory, top folder of rlskyjo

    Returns:
        Path: Path to the project directory
    """     
    return Path(__file__).parent.parent.resolve()

def find_file_in_subdir(parent_dir: Union[Path, str], file_str: Union[Path, str], regex_match: str = None) -> Union[str, None]:
    files = glob.glob(
        os.path.join(parent_dir, "**", file_str), recursive=True
    )
    if regex_match is not None:
        p = re.compile(regex_match)
        files = [ s for s in files if p.match(s) ]
    return sorted(files)[-1] if len(files) else None