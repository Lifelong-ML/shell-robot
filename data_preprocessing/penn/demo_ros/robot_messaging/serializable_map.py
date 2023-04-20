import numpy as np

class SerializableMap:

    def __init__(self, grid, resolution):
        assert type(grid) == np.ndarray, 'Map must be a numpy array'
        assert grid.dtype == np.int8, 'Map must be an int8 array'
        assert grid.ndim == 2, 'Map must be a 2D array'
        assert type(resolution) == float, 'Resolution must be of type float'
        self.grid = grid 
        self.resolution = resolution