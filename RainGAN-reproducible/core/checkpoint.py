import os
from pathlib import Path

import torch


class CheckpointIO:
    def __init__(self, fname_template, device="cpu", **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.device = torch.device(device)
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = Path(self.fname_template.format(step))
        print(f"Saving checkpoint into {fname}...")
        outdict = {name: module.state_dict() for name, module in self.module_dict.items()}
        temporary = fname.with_suffix(fname.suffix + ".tmp")
        torch.save(outdict, temporary)
        os.replace(temporary, fname)

    def load(self, step):
        fname = Path(self.fname_template.format(step))
        if not fname.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {fname}")
        print(f"Loading checkpoint from {fname}...")
        module_dict = torch.load(fname, map_location=self.device, weights_only=True)
        for name, module in self.module_dict.items():
            if name not in module_dict:
                raise KeyError(f"Checkpoint {fname} does not contain component {name!r}")
            module.load_state_dict(module_dict[name])
