from typing import Callable

import torch
from torch.utils.data import DataLoader


def spread_dict(input: dict, groups: list | None = None) -> dict:
    """Spread a dict of `{key: {group: value}}` to a dict of `{group: {key: value}}`. Top-level
    non-dict `key: value` will be spreaded to all groups. Missing keys for a group will be ignored.

    Args:
    + `input`: The dict to spread.
    + `groups`: First-level keys to spread items to. Defaults to `None` to find all possible groups.

    Returns: A dict spreaded by `groups`.

    ---

    Example:
    ```
    >>> spread_dict({
    >>>     "batch_size": 128,
    >>>     "shuffle": {"train": True, "val": False, "test": False},
    >>>     "augment": {"train": "aug0", "val": "aug1"}
    >>> })
    {
        "train": {"batch_size": 128, "shuffle": True, "augment": "aug0"},
        "val": {"batch_size": 128, "shuffle": False, "augment": "aug1"},
        "test": {"batch_size": 128, "shuffle": False},
    }
    ```
    """
    if groups is None:
        # Find all possible groups
        groups = []
        for v in input.values():
            if isinstance(v, dict):
                groups.extend(v.keys())

    # Spread values to separate groups
    spreaded = {g: {} for g in groups}
    for g in groups:
        for k, v in input.items():
            if isinstance(v, dict):
                if g in v.keys():
                    spreaded[g][k] = v.get(g)
            else:
                spreaded[g][k] = v
    return spreaded


def flatten_dictlist(d: dict | list, sep: str = ".", parent_key: str = "") -> dict:
    """Recursively flattens a nested dictionary and/or list. Each key in the returned dict is the
    path to the value, joined by `sep`.

    Args:
    + `d`: Input dict/list.
    + `sep`: Separator used to join keys. Defaults to `'.'`.
    + `parent_key`: Key prefix for recursive calls. Defaults to `''`.

    Returns: A flattened dict.

    ---
    Examples:
    ```
    >>> flatten_dictlist({"a": {"b": 1}})
    {"a.b": 1}

    >>> flatten_dictlist(nested_dict = {
    >>>     'a': {  'b': {  'c':    1,
    >>>                     'd': [  10, 20, {'x':   99}]},
    >>>             'e':    2},
    >>>     'f':    3,
    >>> })
    {'a.b.c': 1, 'a.b.d.0': 10, 'a.b.d.1': 20, 'a.b.d.2.x': 99, 'a.e': 2, 'f': 3}
    ```
    """
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, (dict, list)):
                items.update(flatten_dictlist(v, sep=sep, parent_key=new_key))
            else:
                items[new_key] = v
    elif isinstance(d, list):
        for idx, item in enumerate(d):
            item_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            if isinstance(item, (dict, list)):
                items.update(flatten_dictlist(item, sep=sep, parent_key=item_key))
            else:
                items[item_key] = item
    else:
        # If d is a primitive value, just return it as a single-item dict
        items[parent_key if parent_key else ""] = d
    return items


def traverse_dictlist(input: dict | list, keys: list[str | int], strict: bool = False):
    """Traverses a nested dict/list using a list of keys (string or number).

    Args:
    + input: The dict or list to traverse.
    + keys: A list of keys representing the traversal path.

    Returns: The value at the specified path or `None` if the path does not exist.

    ---
    Examples:
    ```
    >>> a = {'b': {'c': {'d': {'e': 0}}}}
    >>> traverse_dictlist(input=a, keys=['b', 'c', 'd', 'e'])
    0
    >>> traverse_dictlist(input=a, keys=['b', 'f'])
    None
    ```
    """
    current = input
    for k in keys:
        if isinstance(current, (dict)) and k in current.keys():
            current = current[k]
        elif isinstance(current, list) and isinstance(k, int) and 0 <= k < len(current):
            current = current[k]
        else:
            if strict:
                raise ValueError(f"Could not find key `{k}` for input {input}")
            else:
                return None  # Return None if the path does not exist
    return current
