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


def cut_label(
        ground_truth_occ_map,  #: npt.NDArray[np.uint8],
        robot_position,  # Union[List[int], npt.NDArray[np.int0]],
        robot_angle: int,
        robot_frame_size: List[int] = ROBOT_FRAME_SIZE,
        plot_images: bool = False):  #-> npt.NDArray[np.int0]:
    """Cuts out the label from the ground truth map and thresholds it.

    Args:
        ground_truth_occ_map (npt.NDArray[np.uint8]): Ground truth map
            in map frame.
        robot_position (Union[List[int], npt.NDArray[np.int0]]): 
            Array indices of coordinates.
        robot_angle (int): Robot angle in degrees.
        robot_frame_size (List[int]): Size of frame to cut out on each side
            (half distance of the rectangle).
        plot_images (bool, optional): Whether or not to visualize result.
            Used for testing. Defaults to False.

    Returns:
        npt.NDArray[np.int0]: Ground truth label map in robot frame.
    """
    if plot_images:
        ground_truth_occ_map = ground_truth_occ_map.copy()
        rect = ((int(robot_position[0]), int(robot_position[1])),
                (robot_frame_size[0], robot_frame_size[1]), robot_angle)
        box = cv2.boxPoints(rect).astype(np.int0)
        cv2.drawContours(ground_truth_occ_map, [box], 0, (255, 0, 0), 3)

    label_map = np.zeros_like(ground_truth_occ_map)
    label_map[ground_truth_occ_map == 0] = 2
    label_map[ground_truth_occ_map == 100] = 1
    label_map = label_map.astype(np.uint8)

    padding_size = int(np.sqrt(2) * max(robot_frame_size))
    label_map = np.pad(label_map, [(padding_size, ), (padding_size, )],
                       mode='constant')

    M = cv2.getRotationMatrix2D((int(robot_position[0]) + padding_size,
                                 int(robot_position[1]) + padding_size),
                                robot_angle + 90, 1)
    img_rot = cv2.warpAffine(label_map,
                             M, (label_map.shape[0], label_map.shape[1]),
                             flags=cv2.INTER_LINEAR)

    cut_img = img_rot[robot_position[1] - int(robot_frame_size[1] / 2) +
                      padding_size:robot_position[1] +
                      int(robot_frame_size[1] / 2) + padding_size,
                      robot_position[0] - int(robot_frame_size[0] / 2) +
                      padding_size:robot_position[0] +
                      int(robot_frame_size[0] / 2) + padding_size]

    if plot_images:
        _, axs = plt.subplots(3, 1, figsize=(4, 10))
        axs[0].imshow(ground_truth_occ_map)
        axs[0].set_title('Ground truth map')
        axs[0].set_aspect('equal')
        axs[1].imshow(img_rot)
        axs[1].set_aspect('equal')
        axs[1].set_title('Rotated map')
        axs[2].imshow(cut_img)
        axs[2].set_title('Cut out map')
        axs[2].set_aspect('equal')
        axs[2].scatter(robot_frame_size[0] // 2 - 1,
                       robot_frame_size[1] // 2 - 1,
                       c='r')
        plt.show()

    return cut_img


def ray_march_free_space(cell_size: float, laser_scan: SerializableLaserScan):
    # Do processing here.
    min_angle = laser_scan.min_angle
    angle_inc = laser_scan.angle_increment
    ranges = laser_scan.ranges

    #To sample free points
    step_size = cell_size / 2  # free points are sampled at half the cell_size along the range from lidar

    local_3d = []

    for i in range(len(ranges)):
        if not (ranges[i] < 0 or ranges[i] == float('inf')
                or ranges[i] == float('-inf')):
            angle = min_angle + i * angle_inc

            #Getting coords of occupied points
            local_3d.append(
                [ranges[i] * np.sin(angle), 0, ranges[i] * np.cos(angle),
                 1])  #[y, 0, x] -> [x, y, z] in original code
            #Getting coords of free points (points upto the obstacle are all free)
            num_of_samples = int(ranges[i] / step_size)
            for j in range(num_of_samples):
                local_3d.append([(j + 1) * step_size * np.sin(angle), 0,
                                 (j + 1) * step_size * np.cos(angle),
                                 2])  #[y, 0, x] -> [x, y, z] in original code
    return np.array(local_3d)


def estimate_occupancy_from_depth(local_3d, grid_dim, cell_size):

    local_3d_step = local_3d[:, :3]
    occ_lbl = local_3d[:, 3].reshape(-1, 1).astype(np.int64)

    #Uncomment
    # Keep points for which z < 3m (to ensure reliable projection)
    # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
    z = local_3d_step[:, 2]  #-local3D_step[:,2]
    # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
    y = local_3d_step[:, 1]
    #local3D_step = local3D_step[(z < 3) & (z > 0.5) & (y < 1), :]
    #occ_lbl = occ_lbl[(z < 3) & (z > 0.5) & (y < 1), :]

    # print("\nlocal3D_step.shape (only occ & free): " + str(local3D_step.shape))
    # print("occ_lbl.shape (only occ & free): " + str(occ_lbl.shape))

    map_coords = discretize_coords(x=-local_3d_step[:, 0],
                                   z=-local_3d_step[:, 2],
                                   grid_dim=grid_dim,
                                   cell_size=cell_size)

    ## Replicate label pooling
    grid = np.empty((3, grid_dim[0], grid_dim[1]))
    grid[:] = 1 / 3

    # If the robot does not project any values on the grid, then return the empty grid
    if map_coords.shape[0] == 0:
        return np.expand_dims(grid, axis=0)

    concatenated = np.concatenate([map_coords, occ_lbl], axis=-1)
    unique_values, counts = np.unique(concatenated, axis=0, return_counts=True)
    grid[unique_values[:, 2], unique_values[:, 1],
         unique_values[:, 0]] = counts + 1e-5

    return grid / grid.sum(axis=0)


def discretize_coords(x, z, grid_dim, cell_size, translation=0):
    # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
    # If translation=0, assumes the agent is at the center
    # If we want the agent to be positioned lower then use positive translation. When getting the gt_crop, we need negative translation
    map_coords = np.zeros((len(x), 2))
    xb = np.floor(x[:] / cell_size) + (grid_dim[0] - 1) / 2.0
    zb = np.floor(z[:] / cell_size) + (grid_dim[1] - 1) / 2.0 + translation
    xb = xb.astype(np.int32)
    zb = zb.astype(np.int32)
    map_coords[:, 0] = xb
    map_coords[:, 1] = zb
    # keep bin coords within dimensions
    map_coords[map_coords > grid_dim[0] - 1] = grid_dim[0] - 1
    map_coords[map_coords < 0] = 0
    return map_coords.astype(np.int64)


class SequenceDir():

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.map_file = self.data_path / 'map.pkl'
        # Extract the `.pkl` files
        self.sequence_files = sorted(self.data_path.glob('*.pkl'))
        # Remove the map file from the data files
        self.sequence_files.remove(self.map_file)

        self.sequence: List[Tuple[SerializableLaserScan, SerializablePose]] = [
            load_pickle(f) for f in self.sequence_files
        ]
        self.map: SerializableMap = load_pickle(self.map_file)

    def __len__(self):
        return len(self.sequence)

    def _extract_map_label(self, pose: SerializablePose):
        """
        Extract the map label from the pose.
        """
        x_coord = int(pose.x / self.map.resolution)
        y_coord = int(pose.y / self.map.resolution)
        theta = int(np.rad2deg(pose.theta))

        ground_truth_label = cut_label(ground_truth_occ_map=self.map.grid,
                                       robot_position=np.array(
                                           [x_coord, y_coord]),
                                       robot_angle=theta,
                                       plot_images=True)
        return ground_truth_label

    def _process_scan(self, laser_scan: SerializableLaserScan):
        """
        Process the laser scan to get the local3D.
        """
        ray_marched_array = ray_march_free_space(cell_size=self.map.resolution,
                                                 laser_scan=laser_scan)
        ego_grid_occ = estimate_occupancy_from_depth(ray_marched_array,
                                                     ROBOT_FRAME_SIZE,
                                                     self.map.resolution)
        return ego_grid_occ

    def __getitem__(self,
                    index) -> Tuple[SerializableLaserScan, SerializablePose]:
        laser_scan, pose = self.sequence[index]
        return laser_scan, pose
    
    def _visualize_pose_points(self):
        plt.imshow(self.map.grid)
        for idx in range(len(self)):
            global_scan, global_pose = self[idx]
            center_x, center_y = global_pose.global_center(self.map)
            plt.scatter(center_x, center_y, color='red', marker='x', s=100)
        plt.show()

    def _visualize_map(self):

        for idx in range(len(self)):
            global_scan, global_pose = self[idx]
            gt_occupancy = self._extract_map_label(global_pose)
            ego_occupancy = self._process_scan(global_scan)

            points = global_scan.to_global(global_pose, self.map)

            # Visualize the gt occupancy and ego occupancy
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(gt_occupancy)
            plt.scatter(ROBOT_FRAME_SIZE[0] / 2,
                        ROBOT_FRAME_SIZE[1] / 2,
                        color='red',
                        marker='x',
                        s=100)
            plt.scatter(points[:, 0],
                        points[:, 1],
                        color='blue',
                        marker='x',
                        s=1)
            plt.title("Ground Truth Occupancy")
            plt.subplot(1, 2, 2)
            plt.imshow(ego_occupancy.transpose(1, 2, 0))
            plt.title("Ego Occupancy")
            plt.show()


# Load data
data_dir = SequenceDir(args.data_path)
# Visualize the map
data_dir._visualize_pose_points()
