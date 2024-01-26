# ddare/lora.py
from collections import defaultdict
import comfy
import json
import os
import safetensors.torch
import torch
from typing import Dict, Any, Optional, List

from .util import dumb_json

# From comfy.utils.load_torch_file
def load_torch_file(ckpt: str, safe_load : bool = True, device : torch.device = None) -> Dict[str, torch.Tensor]:
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=comfy.checkpoint_pickle)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def read_header(filename : str) -> Optional[Dict[str, Any]]:
    st = os.stat(filename)
    if st.st_size < 8:
        print(f"File {filename} is less than 8 bytes long")
        return None

    f = open(filename, "rb")
    b8 = f.read(8)
    if len(b8) != 8:
        print(f"read only {len(b8)} bytes at start of file")
        return None
    
    headerlen = int.from_bytes(b8, 'little', signed=False)

    if 8 + headerlen > st.st_size:
        print(f"header size {headerlen} is too big for file {filename} of size {st.st_size}")
        return None

    hdrbuf = f.read(headerlen)
    if len(hdrbuf) != headerlen:
        print(f"header size is {headerlen}, but read {len(hdrbuf)} bytes")
        return None

    header = json.loads(hdrbuf)
    if header.get('__metadata__', None) is None:
        print(f"File {filename} does not contain __metadata__")
        return None
    
    return header['__metadata__']

def get_weight_type(name: str) -> str:
    if name.endswith("lora_up.weight"):
        return 'up'
    if name.endswith("lora_down.weight"):
        return 'down'
    if name.endswith(".alpha"):
        return 'alpha'
    return 'other'

def get_model_type(name: str) -> str:
    if name.startswith("lora_unet_"):
        return "unet"
    if name.startswith("lora_te_"):
        return "clip"
    if name.startswith("lora_te1_") or name.startswith("lora_te2_"):
        return "multite"
    return "other"

class DoctorLora:
    """
    A wrapper around a LORA file that provides some metadata and a dictionary-like interface to the LORA tensors.
    """
    def __init__(self, metadata : Optional[Dict[str, Any]], lora : Dict[str, torch.Tensor]):
        self._metadata = metadata
        self.lora = lora
        
    @classmethod
    def load(cls, filename : str) -> Optional['DoctorLora']:
        """
        Factory to load a LORA file and return a DoctorLora object.  If the file is not a LORA file, return None.
        
        Args:
            filename: The name of the file to load.
            
        Returns:
            A DoctorLora object or None.
        """
        filesize = os.path.getsize(filename)
        metadata = read_header(filename)
        if metadata is None:
            metadata = {}
        metadata['dm_filename'] = filename
        metadata['dm_filesize'] = filesize
        lora = load_torch_file(filename, safe_load=True)
        signature = cls.extract_signature(lora)
        if len(signature) > 0:
            metadata['dm_signature'] = signature
        return cls(metadata, lora)

    @classmethod
    def extract_signature(cls, lora : Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Extract a signature from a LORA file.  The signature is a dictionary of the shapes of the tensors in the LORA file and their m lengths.
        
        Args:
            lora: The LORA file to extract the signature from.
        
        Returns:
            A dictionary of the shapes of the tensors in the LORA file and their m lengths. 
        """
        lora_shapes = defaultdict(lambda: defaultdict(set))
        size = 0
        dtype = set()
        for k in lora.keys():
            model_type = get_model_type(k)
            weight_type = get_weight_type(k)
            sz = lora[k].shape
            if len(sz) > 0:
                sz = sz[0]
            else:
                sz = 0
            lora_shapes[model_type][weight_type].add(sz)
            size += lora[k].numel()
            dtype.add(lora[k].dtype)

        clean = json.loads(json.dumps(lora_shapes, default=dumb_json))
        return clean

    def __getitem__(self, key : str) -> torch.Tensor:
        return self.lora[key]
    
    def keys(self):
        return self.lora.keys()

    @property
    def keycount(self) -> int:
        return len(self.lora.keys())

    @property
    def filesize(self) -> int:
        return self._metadata['dm_filesize']

    @property
    def filename(self) -> str:
        return self._metadata['dm_filename']
    
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata
    
    @property
    def signature(self) -> Optional[Dict[str, Any]]:
        return self._metadata.get('dm_signature', None)

    @property
    def parameters(self) -> int:
        return sum([x.numel() for x in self.lora.values()])

    @property
    def tags(self) -> List[str]:
        tags = self._metadata.get("ss_tag_frequency", None)
        if tags is None:
            return []

        try:
            tags = json.loads(tags)
        except Exception as e:
            print(f"Could not load tags: {e}")
            return []
        if not isinstance(tags, dict):
            return []
        result = {}
        for _, vi in tags.items():
            if not isinstance(vi, dict):
                continue
            for k, v in vi.items():
                if k not in result:
                    result[k] = 0
                result[k] += v
        result = [k for k, _ in sorted(result.items(), key=lambda x: x[1], reverse=True)]
        return result
        
