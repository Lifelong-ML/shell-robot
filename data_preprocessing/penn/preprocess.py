import sys
# Add . to path so we can import from shared_structs
sys.path.append('data_preprocessing/')

import argparse
import numpy as np
import pickle
import copy
import cv2
import matplotlib.pyplot as plt
import yaml

from pathlib import Path
from demo_ros.robot_messaging.serializable_pose import SerializablePose
from demo_ros.robot_messaging.serializable_laser_scan import SerializableLaserScan

from typing import Tuple, Union, List

from shared_structs.data_structs import SE2, LaserScan, MapWrapper, ROBOT_FRAME_SIZE

from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('raw_input_path', type=Path, help='Data path.')
parser.add_argument('output_path', type=Path, help='Output path.')
args = parser.parse_args()

input_path = args.raw_input_path
output_path = args.output_path

assert input_path.exists(), f'Input path {input_path} does not exist.'
output_path.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    path = Path(path)
    assert path.exists(), f'Path {path} does not exist.'
    with open(path, 'rb') as f:
        return pickle.load(f, encoding="latin1")


def save_pickle(path: Path, obj):
    path = Path(path)
    print(f'Saving {path}...')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class SequenceDir():

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)

        # Extract the `.pkl` files
        self.sequence_files = sorted(e for e in self.data_path.glob('*.pkl')
                                     if e.name != 'map.pkl')

        self.sequence: List[Tuple[SerializableLaserScan, SerializablePose]] = [
            load_pickle(f) for f in self.sequence_files
        ]

        # Convert to standard types
        self.sequence = [(LaserScan.from_serializable(scan),
                          SE2.from_serializable(pose))
                         for scan, pose in self.sequence]

        self.map = self._load_map()

    def _load_map(self) -> MapWrapper:
        """
        Load the map from the map file.
        """
        map_img_path = self.data_path / 'map.pgm'
        map_metadata_path = self.data_path / 'map.yaml'
        map_img = cv2.imread(str(map_img_path), cv2.IMREAD_GRAYSCALE)
        # load yaml metadata
        with open(map_metadata_path, 'r') as f:
            map_metadata = yaml.load(f, Loader=yaml.FullLoader)
        # Walls are 2
        map_img[map_img == 0] = 2
        # Free space is 1
        map_img[map_img == 254] = 1
        # Unknown space is 0
        map_img[map_img == 205] = 0
        map_img = map_img.astype(np.int8)

        cells = map_img

        resolution = map_metadata['resolution']
        origin = -np.array(map_metadata['origin'][:2])

        return MapWrapper(cells=cells, resolution=resolution, origin=origin)

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index) -> Tuple[LaserScan, SE2]:
        laser_scan, pose = self.sequence[index]
        ego_grid_occ = laser_scan.ego_occupancy(self.map.resolution)
        ground_truth_label = self.map.extract_region(pose)
        return ego_grid_occ, ground_truth_label

    def _visualize_input_outputs(self):

        for idx in range(len(self)):
            ego_grid_occ, ground_truth_label = self[idx]

            fig, ax = plt.subplots(2, 1, figsize=(5, 10))
            ax[0].imshow(ego_grid_occ, norm=None)
            fig.colorbar(ax[1].imshow(np.flip(ground_truth_label, axis=1)))
            plt.show()

    def _manually_align_pose_points(self):
        for idx in range(len(self)):

            def on_key_release(event, state_dict):
                # Handle key release event
                print('You released', event.key)
                if event.key == 'shift':
                    state_dict['shift'] = False

            def on_key_press(event, state_dict):
                # Handle key press event
                print('You pressed', event.key)

                shift_scalar = 0.2 if state_dict['shift'] else 1

                if event.key == 'right':
                    global_pose.x += 0.1 * shift_scalar
                elif event.key == 'left':
                    global_pose.x -= 0.1 * shift_scalar
                elif event.key == 'up':
                    global_pose.y -= 0.1 * shift_scalar
                elif event.key == 'down':
                    global_pose.y += 0.1 * shift_scalar
                elif event.key == 'a' or event.key == 'A':
                    global_pose.theta += 0.1 * shift_scalar
                elif event.key == 'd' or event.key == 'D':
                    global_pose.theta -= 0.1 * shift_scalar
                elif event.key == 'shift':
                    state_dict['shift'] = True

                dots = state_dict['dots']
                center_mark = state_dict['center_mark']

                dots.remove()
                center_mark.remove()
                dots, center_mark = draw_scan_pose(global_scan, global_pose)

                state_dict['dots'] = dots
                state_dict['center_mark'] = center_mark

                event.canvas.draw()

            def draw_scan_pose(global_scan, global_pose):
                global_scan_points = global_scan.to_global(global_pose)
                global_pose_point = global_pose.center()

                print("global pose:", global_pose)

                map_scan_points = self.map._world_to_map(global_scan_points)
                map_pose_point = self.map._world_to_map(global_pose_point)

                dots = plt.scatter(map_scan_points[:, 0],
                                   map_scan_points[:, 1],
                                   color='blue',
                                   marker='.',
                                   s=1)
                center_mark = plt.scatter(map_pose_point[0],
                                          map_pose_point[1],
                                          color='red',
                                          marker='x',
                                          s=1)
                return dots, center_mark

            plt.gca().set_aspect('equal', adjustable='box')
            plt.imshow(self.map.cells)
            plt.title(f'Pose points for {len(self)} scans')
            print("Map origin", self.map.origin)

            global_scan, global_pose = self[idx]

            dots, center_mark = draw_scan_pose(global_scan, global_pose)
            state_dict = {
                'dots': dots,
                'center_mark': center_mark,
                'shift': False
            }

            # Register the key press callback function with the figure
            plt.gcf().canvas.mpl_connect(
                'key_press_event', partial(on_key_press,
                                           state_dict=state_dict))
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                partial(on_key_release, state_dict=state_dict))
            plt.show()


# Load data
data_dir = SequenceDir(input_path)

for idx in range(len(data_dir)):
    input, target = data_dir[idx]
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
    
    # Plot the input and target using matplotlib
    # fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    # ax[0].imshow(input)
    # fig.colorbar(ax[1].imshow(target))
    # plt.show()


    save_pickle(output_path / f"input_target_{idx:06d}.pkl", {
        'input': input,
        'target': target
    })

print("Done")