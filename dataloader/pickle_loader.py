import torch
from pathlib import Path
import pickle


class PickleLoader(torch.utils.data.Dataset):

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.files = sorted(source_dir.glob('*.pkl'))

    def __len__(self):
        return len(self.files)

    def _load_pickle(self, idx):
        with open(self.files[idx], 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, idx):
        assert idx < len(self), f'Index {idx} out of range.'
        pickle_data = self._load_pickle(idx)
        ground_truth_map = pickle_data['map_patch']
        laser_scan = pickle_data['laser_scan']
        return {'ground_truth_map': ground_truth_map, 'laser_scan': laser_scan}
