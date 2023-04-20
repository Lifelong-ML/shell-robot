import numpy as np


class SerializablePose:

    def __init__(self, x, y, theta):
        assert type(x) == float, 'X must be a float'
        assert type(y) == float, 'Y must be a float'
        assert type(theta) == float, 'Theta must be a float'
        self.x = x
        self.y = y
        self.theta = theta

    def center(self):
        return np.array([self.x, self.y])

    def global_center(self, map):
        return self.center() / map.resolution
