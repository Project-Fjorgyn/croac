import unittest
import numpy as np

from .. import (
    Pond
)

class MockFrog(object):
    def __init__(self, zeros):
        self.zeros = zeros

    def listen(self, sample_rate):
        return np.concatenate([
            np.array([sample_rate]), np.zeros(self.zeros)
        ])

class TestPondListenToAntenna(unittest.TestCase):
    def test_base(self):
        pond = Pond(
            [MockFrog(0), MockFrog(1)],
            [np.array([1, 1, 2]), np.array([3, 1, 1])],
            [np.array([1, 1, 1])]
        )
        sample_rate = 1000
        expected = np.array([
            0, 0, 0, # time delay for the closer of the two
            1000, 0, 0, # time delay complete for the further of the two
            1000 / 4, 0
        ])
        actual = pond.listen_to_antenna(pond.antennas[0], sample_rate)

        assert (expected == actual).all()
