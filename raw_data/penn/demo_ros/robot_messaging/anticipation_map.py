import numpy as np


class AnticipationMap:

    def _numpy_arr_to_list(self, arr):
        return arr.tolist()

    def __init__(self, value_arr):
        if type(value_arr) == np.ndarray:
            value_arr = self._numpy_arr_to_list(value_arr)
        assert type(value_arr
                    ) == list, 'Values must be a list of lists, not {}'.format(
                        type(value_arr))
        assert len(set(
            len(l)
            for l in value_arr)) == 1, 'Lists must all be the same length'
        for l in value_arr:
            assert type(
                l) == list, 'Values must be a list of floats, not {}'.format(
                    type(l))
            for e in l:
                assert type(
                    e
                ) == float, 'Values must be a list of lists of floats, not {}'.format(
                    type(e))
        self.value_arr = value_arr

    def to_numpy(self):
        return np.array(self.value_arr)