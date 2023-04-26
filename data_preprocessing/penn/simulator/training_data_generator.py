import sys
# Add . to path so we can import from shared_structs
sys.path.append('data_preprocessing/')

from shared_structs.data_structs import SE2, LaserScan, MapWrapper

import argparse
import random
import pickle
from pathlib import Path
from typing import Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

# Define the argument parser
parser = argparse.ArgumentParser(
    description='Generate random poses and robot frame laser scans.')
parser.add_argument('map_file',
                    type=Path,
                    help='the path to the raster map file')
parser.add_argument('output_path', type=Path, help='Output path.')
parser.add_argument('--num_samples',
                    type=int,
                    default=50,
                    help='Number of laser samples to take from the map.')
parser.add_argument('--visualize',
                    action='store_true',
                    help='Visualize the generated laser scans.')
args = parser.parse_args()

assert args.map_file.exists(), f'Path {args.map_file} does not exist.'
args.output_path.mkdir(parents=True, exist_ok=True)

UNKNOWN_COLOR = 127
FREE_SPACE_COLOR = 255
WALL_COLOR = 0


def save_pickle(path: Path, obj):
    path = Path(path)
    print(f'Saving {path}...')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# Load the map file
map_data = plt.imread(str(args.map_file.resolve()))
map_data = (map_data * 255).astype(np.uint8)
# cells is a 2D array of 0, 1, or 2
# 0 unknown
# 1 is free
# 2 is occupied
map_data = np.where(map_data == FREE_SPACE_COLOR, 1, map_data)
map_data = np.where(map_data == WALL_COLOR, 2, map_data)
map_data = np.where(map_data == UNKNOWN_COLOR, 0, map_data)

# Convert the map data to a MapWrapper
map = MapWrapper(map_data, origin=[0, 0], resolution=0.05)


def random_map_frame_positions(map: MapWrapper) -> Tuple[int, int, float]:
    """
    Returns a random map frame SE2 pose in free space
    """
    obstacle_inflated_cells = map.cells.copy()
    # Convolve the map with a blur kernel
    obstacle_inflated_cells = cv2.GaussianBlur(
        obstacle_inflated_cells.astype(np.float32), (45, 45), 3)

    # # Plot the blurred map
    # plt.imshow(obstacle_inflated_cells.T, cmap='gray')
    # plt.colorbar()
    # plt.show()

    # Get the indices of the free space
    free_space_indices = np.where(obstacle_inflated_cells == 1)
    # Choose a random index
    random_index = random.randint(0, len(free_space_indices[0]) - 1)
    # Get the x, y coordinates of the random free space cell
    x, y = free_space_indices[0][random_index], free_space_indices[1][
        random_index]
    # Get the theta of the random pose
    theta = random.uniform(-np.pi, np.pi)
    return (x, y, theta)


# Generate random poses
positions = [random_map_frame_positions(map) for _ in range(args.num_samples)]


def make_laser_scan_pose(position: Tuple[int, int, float],
                         visualize: bool = False) -> Tuple[LaserScan, SE2]:
    x, y, theta = position
    map_position = np.array([x, y])

    laser_scan = map.make_laser_scan(x,
                                     y,
                                     theta,
                                     max_range=3.5,
                                     fov_degrees=90,
                                     num_scans=10,
                                     visualize_steps=False)
    global_position = map._map_to_world(map_position)
    global_se2 = SE2(global_position[0], global_position[1], theta)

    if visualize:
        plt.title("Map Frame Laser Scan")
        plt.imshow(map.cells.T, cmap='gray')
        plt.scatter(x, y, color='r')
        # Draw arrow for the pose
        plt.arrow(x, y, 30 * np.cos(theta), 30 * np.sin(theta), color='r')
        laser_points_global = laser_scan.to_global(global_se2)
        laser_points_map = map._world_to_map(laser_points_global)
        plt.scatter(laser_points_map[:, 0],
                    laser_points_map[:, 1],
                    color='r',
                    s=3)
        plt.show()

    return laser_scan, global_se2


def plot_poses(scan_pose_lst: List[Tuple[LaserScan, SE2]], map: MapWrapper):

    plt.title("Map Frame Laser Scan")
    plt.imshow(map.cells.T, cmap='gray')
    for scan, pose in scan_pose_lst:
        pose_center_global = pose.center()
        pose_center_map = map._world_to_map(pose_center_global)
        plt.scatter(pose_center_map[0], pose_center_map[1], color='b')
        # Draw arrow for the pose
        plt.arrow(pose_center_map[0],
                  pose_center_map[1],
                  30 * np.cos(pose.theta),
                  30 * np.sin(pose.theta),
                  color='b')
        laser_points_global = scan.to_global(pose)
        laser_points_map = map._world_to_map(laser_points_global)
        plt.scatter(laser_points_map[:, 0],
                    laser_points_map[:, 1],
                    color='r',
                    s=3)
    plt.show()


print(f'Generating {args.num_samples} laser scans...')
print("Before loop", time.time())
scan_pose = [make_laser_scan_pose(position) for position in positions]
print("After loop", time.time())
for idx, (laser_scan, pose) in enumerate(scan_pose):
    input = laser_scan.ego_occupancy(map.resolution)
    target = map.extract_region(pose)

    assert input.shape[:
                       2] == target.shape, f"Input shape {input.shape} does not match target shape {target.shape}"

    save_pickle(args.output_path / f"input_target_{idx:06d}.pkl", {
        'input': input,
        'target': target
    })

if args.visualize:
    plot_poses(scan_pose, map)
