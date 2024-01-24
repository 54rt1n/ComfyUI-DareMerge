# merge/util.py
from comfy.model_patcher import ModelPatcher
import contextlib
import torch
from typing import Optional, Dict

def patcher(model: ModelPatcher, key : str) -> Optional[torch.Tensor]:
    # This is slow, but seems to work
    model_sd = model.model_state_dict()
    if key not in model_sd:
        print("could not patch. key doesn't exist in model:", key)
        return None

    weight : torch.Tensor = model_sd[key]

    temp_weight = weight.to(torch.float32, copy=True)
    out_weight = model.calculate_weight(model.patches[key], temp_weight, key).to(weight.dtype)
    return out_weight

@contextlib.contextmanager
def cuda_memory_profiler(display : str = True):
    """
    A context manager for profiling CUDA memory usage in PyTorch.
    """
    if display is False:
        yield
        return
    
    if not torch.cuda.is_available():
        print("CUDA is not available, skipping memory profiling")
        yield
        return
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_memory = torch.cuda.memory_allocated()

    try:
        yield
    finally:
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Memory allocated at start: {start_memory / (1024 ** 2):.2f} MB")
        print(f"Memory allocated at end: {end_memory / (1024 ** 2):.2f} MB")
        print(f"Net memory change: {(end_memory - start_memory) / (1024 ** 2):.2f} MB")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_patched_state(model : ModelPatcher) -> Dict[str, torch.Tensor]:
    """Uses a Comfy ModelPatcher to get the patched state dict of a model.

    Args:
        model (ModelPatcher): The model to get the patched state dict from.
        
    Returns:
        Dict[str, torch.Tensor]: The patched state dict.
    """
    if len(model.patches) > 0:
        print("Model has patches, applying them")
        model.patch_model(None, True)
        model_sd = model.model_state_dict()
        model.unpatch_model()
    else:
        model_sd = model.model_state_dict()
    
    return model_sd
