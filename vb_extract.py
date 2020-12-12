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


def extract_vb_samples(stan_vb_results: collections.OrderedDict) -> Dict:
    """Extract samples from a stan fit object true to size.

    Args:
        stan_vb_results: as returned from pystan.StanModel.vb(...)

    Returns:
        dict: contains samples in combined by size
    """
    num_samples = len(stan_vb_results["sampler_params"]
                      [0]) if stan_vb_results["sampler_params"] else 0
    parameter_shapes = collections.defaultdict(list)
    parameter_indices = {}
    for param_index, raw_name in enumerate(
            stan_vb_results["sampler_param_names"]):
        name, indices = parse_bracketed_param_name(raw_name)
        shape = [int(x) for x in indices]
        indices = tuple(int(x) - 1 for x in indices)
        parameter_shapes[name].append(shape)
        parameter_indices[param_index] = (name, indices)

    samples = {
        name: np.empty((num_samples,) + tuple(np.array(shape).max(axis=0)))
        for name, shape in parameter_shapes.items()
    }

    for param_index, (name, indices) in parameter_indices.items():
        samples[name][(...,) +
                      indices] = stan_vb_results["sampler_params"][param_index]

    return samples
