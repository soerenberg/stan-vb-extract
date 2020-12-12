import re
import collections
from typing import Dict, Tuple

import numpy as np


def parse_bracketed_param_name(param_name: str) -> Tuple[str, Tuple[int, ...]]:
    """Parses parameter names with array brackets.

    Args:
        param_name: parameter name as found in stan vb fit objects.

    Returns:
        str: purged parameter name
        Tuple[int]: indices found in brackets (if any).

    Raises:
        ValueError: if param_name does not a valid Stan parameter name.

    Example:
        parse_bracketed_param_name("alpha[2,3]")
        yields
            "alpha", (2,3)

        parse_bracketed_param_name("beta")
        yields
            "beta", tuple()
    """
    pattern = re.compile(
        r"""(?P<name>[a-zA-Z0-9_]+)(\[(?P<indices>(\d+)(,\s*\d+)*)\])?$""")
    match = pattern.match(param_name)
    if match is None:
        raise ValueError(f"Name {param_name} is no valid parameter name.")
    name = match.group("name")
    indices = tuple() if match.group("indices") is None else match.group(
        "indices").split(",")
    return name, tuple(int(x) for x in indices)
