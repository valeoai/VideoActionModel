import os


def expand_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))
