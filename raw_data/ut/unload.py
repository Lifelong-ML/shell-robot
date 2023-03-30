import rosbag
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from PIL import Image
import copy
import math
# from tf.transformations import euler_from_quaternion

parser = argparse.ArgumentParser(description='Unload a bag file.')
parser.add_argument('bagfile', type=Path, help='The bag file to unload.')
args = parser.parse_args()

assert args.bagfile.exists(), f'Bag file {args.bagfile} does not exist.'

bag = rosbag.Bag(args.bagfile)


def euler_from_quaternion(x, y, z, w):
    """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class SE2():

    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta

    def center(self) -> np.ndarray:
        return np.array([self.x, self.y])


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


class MapWrapper():

    def __init__(self, cells: np.ndarray, origin: np.ndarray,
                 resolution: float):
        self.cells = cells
        self.origin = origin
        self.resolution = resolution

    def _world_to_map(self, world: np.ndarray) -> np.ndarray:
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

    def extract_region(self, pose: SE2) -> np.ndarray:
        img = Image.fromarray(self.cells)

        pose_center = pose.center()
        map_center = self._world_to_map(pose_center)

        img = img.rotate(-pose.theta * 180 / np.pi,
                         expand=True,
                         center=tuple(map_center))
        img = img.crop((map_center[0] - 100, map_center[1] - 100,
                        map_center[0] + 100, map_center[1] + 100))
        return np.array(img)


def process_map_msg(msg) -> MapWrapper:
    width = msg.info.width
    height = msg.info.height
    origin_x = msg.info.origin.position.x
    origin_y = msg.info.origin.position.y
    resolution = msg.info.resolution
    data = np.array(msg.data).reshape((height, width)).T

    data[data == 100] = 2
    data[data == 0] = 1
    data[data == -1] = 0
    data = data.astype(np.uint8)

    origin = np.array([origin_x, origin_y])
    return MapWrapper(data, origin, resolution)


def process_odom_msg(msg) -> SE2:
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    _, _, theta = euler_from_quaternion(msg.pose.pose.orientation.x,
                                        msg.pose.pose.orientation.y,
                                        msg.pose.pose.orientation.z,
                                        msg.pose.pose.orientation.w)
    return SE2(x, y, theta)


def process_scan_msg(msg) -> LaserScan:
    ranges = np.array(msg.ranges)
    angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment,
                       msg.angle_increment)

    assert len(ranges) == len(
        angles), f"Ranges {len(ranges)} != angles {len(angles)}"

    # ranges = np.ones_like(ranges)
    # ranges = np.ones((50,)) * 0.11
    # angles = np.linspace(-np.pi/10, np.pi/10, 50)
    return LaserScan(np.flip(ranges), angles)


def load_map(bag: rosbag.Bag) -> Optional[MapWrapper]:
    map = None
    for _, msg, _ in bag.read_messages(topics=['/map']):
        map = process_map_msg(msg)
    return map


def load_laser_pose_pair(bag: rosbag.Bag) -> List[Tuple[SE2, LaserScan]]:
    odom = None
    scan_odom_list = []
    for t, msg, _ in bag.read_messages(topics=['/odom', '/scan_filtered']):
        if t == '/odom':
            odom = process_odom_msg(msg)
        elif t == '/scan_filtered':
            scan = process_scan_msg(msg)
            if odom is not None:
                scan_odom_list.append((scan, odom))
    return scan_odom_list


map = load_map(bag)
print("Loaded map")

scan_odom_list = load_laser_pose_pair(bag)
print(f"Loaded {len(scan_odom_list)} laser scan and odometry pairs")

bag.close()

accumulated_map = copy.deepcopy(map)

for idx, (scan, odom) in enumerate(scan_odom_list):
    # if 30 < idx < 200:
    accumulated_map = accumulated_map.place_pose_laser(odom, scan)

plt.imshow(accumulated_map.cells)
plt.colorbar()
plt.show()