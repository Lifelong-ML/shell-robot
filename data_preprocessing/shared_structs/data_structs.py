import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from PIL import Image
import copy
import math
import cv2

ROBOT_FRAME_SIZE = [160, 160]


class SE2():

    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    def center(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def theta_deg(self) -> float:
        return np.rad2deg(self.theta)

    @staticmethod
    def from_serializable(serializable_pose):
        x = serializable_pose.x
        y = serializable_pose.y
        theta = serializable_pose.theta
        return SE2(x, y, theta)

    def __repr__(self) -> str:
        return f'({self.x}, {self.y}, {self.theta})'


class LaserScan():

    def __init__(self, ranges: np.ndarray, angles: np.ndarray):
        self.ranges = ranges
        self.angles = angles

    def to_points(self) -> np.ndarray:
        points = np.zeros((len(self.ranges), 2))
        points[:, 0] = self.ranges * np.cos(self.angles)
        points[:, 1] = self.ranges * np.sin(self.angles)
        in_max_range = self.ranges < 2
        in_min_range = self.ranges > 0.1
        points = points[np.logical_and(in_max_range, in_min_range)]

        return -points

    def to_global(self, pose: SE2) -> np.ndarray:
        points = self.to_points()

        # Rotate the points
        def make_rotation_matrix(theta):
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        rotation_matrix = make_rotation_matrix(-pose.theta + np.pi)
        points = np.matmul(points, rotation_matrix)
        # Translate the points
        points += pose.center()
        return points

    @staticmethod
    def from_serializable(serializable_scan):
        ranges = np.array(serializable_scan.ranges)
        angles = serializable_scan.angles
        return LaserScan(ranges, angles)

    def ego_occupancy(self, map_grid_size: float, grid_dim=ROBOT_FRAME_SIZE):

        cell_size = map_grid_size

        def ray_march_free_space(cell_size: float, laser_scan: LaserScan):
            # Do processing here.
            ranges = laser_scan.ranges

            #To sample free points
            step_size = cell_size / 2  # free points are sampled at half the cell_size along the range from lidar

            local_3d = []

            for i in range(len(ranges)):
                if not (ranges[i] < 0 or ranges[i] == float('inf')
                        or ranges[i] == float('-inf')):
                    angle = laser_scan.angles[i]

                    #Getting coords of occupied points
                    local_3d.append([
                        ranges[i] * np.sin(angle), 0,
                        ranges[i] * np.cos(angle), 1
                    ])  #[y, 0, x] -> [x, y, z] in original code
                    #Getting coords of free points (points upto the obstacle are all free)
                    num_of_samples = int(ranges[i] / step_size)
                    for j in range(num_of_samples):
                        local_3d.append([
                            (j + 1) * step_size * np.sin(angle), 0,
                            (j + 1) * step_size * np.cos(angle), 2
                        ])  #[y, 0, x] -> [x, y, z] in original code
            return np.array(local_3d)

        local_3d = ray_march_free_space(cell_size, self)

        local_3d_step = local_3d[:, :3]
        occ_lbl = local_3d[:, 3].reshape(-1, 1).astype(np.int64)

        #Uncomment
        # Keep points for which z < 3m (to ensure reliable projection)
        # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
        # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
        #local3D_step = local3D_step[(z < 3) & (z > 0.5) & (y < 1), :]
        #occ_lbl = occ_lbl[(z < 3) & (z > 0.5) & (y < 1), :]

        def discretize_coords(x, z, grid_dim, cell_size, translation=0):
            # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
            # If translation=0, assumes the agent is at the center
            # If we want the agent to be positioned lower then use positive translation. When getting the gt_crop, we need negative translation
            map_coords = np.zeros((len(x), 2))
            xb = np.floor(x[:] / cell_size) + (grid_dim[0] - 1) / 2.0
            zb = np.floor(
                z[:] / cell_size) + (grid_dim[1] - 1) / 2.0 + translation
            xb = xb.astype(np.int32)
            zb = zb.astype(np.int32)
            map_coords[:, 0] = xb
            map_coords[:, 1] = zb
            # keep bin coords within dimensions
            map_coords[map_coords > grid_dim[0] - 1] = grid_dim[0] - 1
            map_coords[map_coords < 0] = 0
            return map_coords.astype(np.int64)

        map_coords = discretize_coords(x=-local_3d_step[:, 0],
                                       z=-local_3d_step[:, 2],
                                       grid_dim=grid_dim,
                                       cell_size=cell_size)

        ## Replicate label pooling
        grid = np.ones((3, grid_dim[0], grid_dim[1])) * (1 / 3)

        # If the robot does not project any values on the grid, then return the empty grid
        if map_coords.shape[0] == 0:
            return np.expand_dims(grid, axis=0)

        concatenated = np.concatenate([map_coords, occ_lbl], axis=-1)
        unique_values, counts = np.unique(concatenated,
                                          axis=0,
                                          return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1],
             unique_values[:, 0]] = counts + 1e-5
        
        res_grid = grid / grid.sum(axis=0)
        res_grid = res_grid.transpose(1, 2, 0)

        return res_grid.astype(np.float32)


class MapWrapper():

    def __init__(self, cells: np.ndarray, origin: np.ndarray,
                 resolution: float):
        # cells is a 2D array of 0, 1, or 2
        # 0 unknown
        # 1 is free
        # 2 is occupied
        self.cells = cells
        self.origin = origin
        self.resolution = resolution

        assert self.cells.ndim == 2, "Cells must be 2D array"
        assert (set(np.unique(self.cells)) - set([0, 1, 2])) == set(), \
            "Cells must be 0, 1, or 2"

    def _world_to_map(self, world: np.ndarray) -> np.ndarray:
        if type(world) == SE2:
            center = world.center() - self.origin
            scaled_center = center / self.resolution
            return SE2(*np.floor(scaled_center).astype(int), world.theta)
        world = np.array(world)
        if world.ndim == 1:
            unscaled_center = world - self.origin
        elif world.ndim == 2:
            # world is N x 2 array
            unscaled_center = world - np.tile(self.origin, (world.shape[0], 1))
        else:
            raise ValueError("World must be 1D or 2D array")
        scaled_center = unscaled_center / self.resolution
        return np.floor(scaled_center).astype(int)

    def _map_to_world(self, map: np.ndarray) -> np.ndarray:
        return map * self.resolution + self.origin

    def place_pose_laser(self, pose: SE2, scan: LaserScan) -> 'MapWrapper':
        global_center = pose.center()
        map_center = self._world_to_map(global_center)

        global_points = scan.to_global(pose)
        map_points = self._world_to_map(global_points)

        modified_cells = self.cells.copy()
        modified_cells[map_center[0], map_center[1]] = 10
        modified_cells[map_points[:, 0], map_points[:, 1]] = 5
        return MapWrapper(modified_cells, self.origin, self.resolution)

    def extract_region(self,
                       global_pose: SE2,
                       robot_frame_size: List[int] = ROBOT_FRAME_SIZE,
                       plot_images: bool = False):
        """Cuts out the label from the ground truth map and returns the proper channels.
        """
        ground_truth_occ_map = self.cells
        robot_position = self._world_to_map(global_pose.center())
        robot_angle = int(global_pose.theta_deg())
        if plot_images:
            ground_truth_occ_map = ground_truth_occ_map.copy()
            rect = ((int(robot_position[0]), int(robot_position[1])),
                    (robot_frame_size[0], robot_frame_size[1]), robot_angle)
            box = cv2.boxPoints(rect).astype(np.int0)
            cv2.drawContours(ground_truth_occ_map, [box], 0, (255, 0, 0), 3)
        # Unknown space is 0, free space is 2, occupied space is 1.
        label_map = np.zeros_like(ground_truth_occ_map)
        label_map[ground_truth_occ_map == 1] = 2
        label_map[ground_truth_occ_map == 2] = 1
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
