
def find_non_serializable(obj, path="root"):
    import json
    """
    Recursively check obj for any values that are not JSON serializable.
    Prints the path to the problematic item.
    """
    try:
        json.dumps(obj)
        return None  # All good
    except TypeError:
        # Not serializable at this level, try deeper if possible
        if isinstance(obj, dict):
            for k, v in obj.items():
                err_path = find_non_serializable(v, f"{path}['{k}']")
                if err_path:
                    return err_path
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                err_path = find_non_serializable(v, f"{path}[{i}]")
                if err_path:
                    return err_path
        else:
            # Not dict/list and not serializable here
            return path
    return None