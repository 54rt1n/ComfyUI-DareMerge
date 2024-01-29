# ddare/model.py

import re
from typing import List, Generator

def layer_in_mask(layer: str, keys : List[str]) -> bool:
    """
    Checks if a layer is in the mask.

    Args:
        layer (str): The layer name.
        mask (ModelMask): The mask.

    Returns:
        bool: True if the layer is in the mask.
    """
    for k in layers_for_mask(layer, keys):
        return True
    return False

def layers_for_mask(layer: str, keys : List[str]) -> Generator[str, None, None]:
    """
    Gets the layers for a mask.

    Args:
        layer (str): The layer name.
        keys (List[str]): The keys to search.

    Returns:
        Generator[str, None, None]: A generator containing the layers.
    """
    # If layer doesn't end with a dot, add one
    if not layer.endswith("."):
        layer += "."
    
    # Match our wildcard
    if re.search(r"\*", layer):
        # We need to escape the layer name, and then replace the wildcard with a regex
        wclayer = re.escape(layer)
        wclayer = re.sub(r"\\\*", r"(.*)", wclayer)
        wclayer = re.compile(wclayer)
    else:
        wclayer = None
    
    for k in keys:
        #prefix = len("diffusion_model.")
        prefix = 0
        key = k[prefix:]
        if key.startswith(layer) or key.endswith(layer) or key == layer:
            yield k
        elif wclayer is not None:
            match = wclayer.match(key)
            if match:
                yield k

def collect_layers(layers: str, keys : List[str]) -> List[str]:
    """
    Collects the layer names from the input string.

    Args:
        layers (str): The layer names.

    Returns:
        Tuple[str]: A tuple containing the layer names.
    """
    # We should split by newline and comma, and remove whitespace and empty strings
    clean = re.sub(r"\s+", "", layers)
    layers = re.split(r"[\n,]", clean)
    # if we have any braces, we need to collect the numbers inside and expand them
    # we do this by matching for braces, and then pulling out the comma separated values inside with regex
    # TODO recurse this to handle multiple braces in one key
    
    bracket = re.compile(r"\{(.*?)\}")
    results = []
    for layer in layers:
        match = bracket.match(layer)
        if match:
            matchval = match.group(1)
            branches = re.sub(r"\s+", "", matchval).split(",")
            for branch in branches:
                new_branch = re.sub(layer, matchval, branch)
                if layer_in_mask(new_branch, keys):
                    results.append(new_branch)
                else:
                    print("Branch not found, skipping", new_branch)
        else:
            if layer_in_mask(layer, keys):
                results.append(layer)
            else:
                print("Layer not found, skipping", layer)
    return results
