import torch
from pathlib import Path
import pickle
import numpy as np


class PickleLoader(torch.utils.data.Dataset):

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.files = sorted(source_dir.glob('*.pkl'))

    def __len__(self):
        return len(self.files)

    def _load_pickle(self, idx):
        with open(self.files[idx], 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, idx: int):
        assert idx < len(self), f'Index {idx} out of range.'
        pickle_data = self._load_pickle(idx)
        input = pickle_data['input']
        target = pickle_data['target']

        # Check shapes are the same
        assert input.ndim == 3, \
            f"Input shape {input.shape} does not have 3 dimensions"
        assert target.ndim == 2, \
            f"Target shape {target.shape} does not have 2 dimensions"

        assert input.shape[:2] == target.shape, \
            f"Input shape {input.shape[:2]} does not match target shape {target.shape[:2]}"
        assert input.shape[2] == 3, \
            f"Input shape {input.shape} does not have 3 channels"


        assert input.dtype == np.float32, \
            f"Input dtype {input.dtype} is not float32"
        assert target.dtype == np.uint8, \
            f"Target dtype {target.dtype} is not uint8"

        # Convert to torch-ready datatypes
        target = target.astype(np.int64)

        # Move channel axis to the beginning for the input (NxNx3 -> 3xNxN)
        input = np.moveaxis(input, 2, 0)

        return {'input': input, 'target': target}
