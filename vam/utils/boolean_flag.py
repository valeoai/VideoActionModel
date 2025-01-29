def boolean_flag(arg: str | bool) -> bool:
    """Add a boolean flag to argparse parser."""
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("true", "1", "yes", "y"):
        return True
    elif arg.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise ValueError(f"Expected 'true'/'false' or '1'/'0', but got '{arg}'")
