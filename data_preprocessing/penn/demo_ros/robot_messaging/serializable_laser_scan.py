import math
import numpy as np


class SerializableLaserScan:

    def __init__(self, ranges, min_angle, angle_increment):
        assert type(ranges) == tuple or type(
            ranges
        ) == list, 'Ranges must be a tuple or list; got {} {}'.format(
            type(ranges), ranges)
        for e in ranges:
            assert type(
                e
            ) == float, 'Ranges must be a tuple of floats; got {} {}'.format(
                type(e), e)
        assert type(min_angle) == float, 'Min angle must be a float'
        assert type(
            angle_increment) == float, 'Angle increment must be a float'

        self.ranges = tuple(
            [float(-1) if math.isnan(e) else e for e in ranges])
        self.min_angle = min_angle
        self.angle_increment = angle_increment

    @property
    def angles(self):
        return (
            np.arange(0, len(self.ranges)) * self.angle_increment +
            self.min_angle)

    def to_points(self) -> np.ndarray:
        ranges = np.array(self.ranges)
        points = np.zeros((len(self.ranges), 2))
        points[:, 0] = ranges * np.cos(self.angles)
        points[:, 1] = ranges * np.sin(self.angles)
        in_max_range = ranges < 3
        in_min_range = ranges > 0.1
        points = points[np.logical_and(in_max_range, in_min_range)]

        return -points

    def to_global(self, pose, map) -> np.ndarray:
        points = self.to_points()

        # Rotate the points
        def make_rotation_matrix(theta):
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        rotation_matrix = make_rotation_matrix(-pose.theta + np.pi)
        points = np.matmul(points, rotation_matrix)
        # Translate the points
        points += pose.center()
        points /= map.resolution
        return points
