# ddare/reporting.py
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
import torch
from typing import Tuple

PLOT_SCALING = ['mean', 'log', 'max', 'min', 'std', 'none']

def get_figsize(numel : int) -> Tuple[int, int]:
    if numel <= 320:
        width = 40
    elif numel <= 1280:
        width = 64
    elif numel <= 2560:
        width = 64
    elif numel <= 102400:
        width = 320
    elif numel <= 245760:
        width = 768
    elif numel <= 409600:
        width = 2560
    elif numel <= 819200:
        width = 640
    elif numel <= 921600:
        width = 320 * 3
    elif numel <= 1638400:
        width = 1280
    else:
        width = 1280 * 3

    height = numel // width

    return (width, height)

def plot_model_layer(layer: torch.Tensor, layer_name: str, scaling : str = 'mean', show_legend: bool = True, **kwargs) -> torch.Tensor:
    # Count the number of layers to determine subplot grid size
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    dshape = layer.shape
    data = layer.cpu().numpy().flatten()
    numel = np.prod(dshape)
    width, height = get_figsize(numel)
    data = data.reshape((height, width))

    # Scaling
    if scaling == 'mean':
        data = data / np.mean(data)
    elif scaling == 'log':
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = np.max(data[~np.isinf(data)])
        data = np.sign(data) * np.log(np.abs(data)+1e-9)
    elif scaling == 'max':
        data = data / np.max(data)
    elif scaling == 'min':
        data = data / np.min(data)
    elif scaling == 'std':
        data = data / np.std(data)
    elif scaling == 'none':
        pass
    else:
        raise ValueError(f'Unknown scaling: {scaling}')

    # Plotting
    cax = ax.matshow(data, cmap='viridis')
    if show_legend:
        fig.colorbar(cax, ax=ax)
    ax.set_title(f'Layer: {layer_name} - {dshape}')
    
    plt.tight_layout()
    
    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Create and return a PIL image
    img = Image.open(buf)
    return torch.from_numpy(np.array(img)).float() / 255.0
