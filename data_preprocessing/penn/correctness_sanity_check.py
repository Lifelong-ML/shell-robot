import sys
# Add . to path so we can import from shared_structs
sys.path.append('data_preprocessing/')

import argparse
import numpy as np
import pickle
import copy
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from demo_ros.robot_messaging.serializable_map import SerializableMap
from demo_ros.robot_messaging.serializable_pose import SerializablePose
from demo_ros.robot_messaging.serializable_laser_scan import SerializableLaserScan

from shared_structs.data_structs import SE2, LaserScan

from typing import Tuple, Union, List

# Get path to data from command line
parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=Path, help='Data path.')
args = parser.parse_args()

ROBOT_FRAME_SIZE = [160, 160]


def load_pickle(path: Path):
    path = Path(path)
    assert path.exists(), f'Path {path} does not exist.'
    with open(path, 'rb') as f:
        return pickle.load(f, encoding="latin1")


class SequenceDir():

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        map_file = self.data_path / 'map.pkl'
        # Extract the `.pkl` files
        self.sequence_files = sorted(self.data_path.glob('*.pkl'))

        if map_file in self.sequence_files:
            # Remove the map file from the data files
            self.sequence_files.remove(map_file)

        self.sequence: List[Tuple[SerializableLaserScan, SerializablePose]] = [
            load_pickle(f) for f in self.sequence_files
        ]

        # Convert to standard types
        self.sequence = [(LaserScan.from_serializable(scan),
                          SE2.from_serializable(pose))
                         for scan, pose in self.sequence]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(
            self,
            index: int) -> Tuple[SerializableLaserScan, SerializablePose]:
        return self.sequence[index]

    def __iter__(self):
        return iter(self.sequence)

    def visualize_pose_positions_on_map(self):

        for scan, pose in self.sequence:
            plt.plot(pose.x, pose.y, 'x', color='red')
            points = scan.to_global(pose)
            plt.plot(points[:, 0], points[:, 1], '.', color='blue')
        plt.show()


sequence_dir = SequenceDir(args.data_path)
sequence_dir.visualize_pose_positions_on_map()
