import unittest
import numpy as np

from .. import (
    DelayArray,
    SPEED_OF_SOUND
)

class TestDelayArray(unittest.TestCase):
    def test_compute_delays(self):
        antenna_x = np.array([1, 2])
        antenna_y = np.array([3, 2])
        theta = np.array([0, np.pi/4])
        phi = np.array([np.pi/4, np.pi])
        sample_rate = 1000

        darray = DelayArray(antenna_x, antenna_y, theta, phi, sample_rate)
        darray._compute_delays()

        expected_delays = np.array(
            [
                [
                    [
                        -1 * np.sin(0) * np.cos(np.pi/4) - 3 * np.sin(0) * np.sin(np.pi/4), # antenna 1
                        -2 * np.sin(0) * np.cos(np.pi/4) - 2 * np.sin(0) * np.sin(np.pi/4), # antenna 2
                    ], # phi 1
                    [
                        -1 * np.sin(0) * np.cos(np.pi) - 3 * np.sin(0) * np.sin(np.pi), # antenna 1
                        -2 * np.sin(0) * np.cos(np.pi) - 2 * np.sin(0) * np.sin(np.pi), # antenna 2
                    ] # phi 2
                ], # theta 1
                [
                    [
                        -1 * np.sin(np.pi/4) * np.cos(np.pi/4) - 3 * np.sin(np.pi/4) * np.sin(np.pi/4), # antenna 1
                        -2 * np.sin(np.pi/4) * np.cos(np.pi/4) - 2 * np.sin(np.pi/4) * np.sin(np.pi/4), # antenna 2
                    ], # phi 1
                    [
                        -1 * np.sin(np.pi/4) * np.cos(np.pi) - 3 * np.sin(np.pi/4) * np.sin(np.pi), # antenna 1
                        -2 * np.sin(np.pi/4) * np.cos(np.pi) - 2 * np.sin(np.pi/4) * np.sin(np.pi), # antenna 2
                    ] # phi 2
                ] # theta 2
            ]
        )
        expected_delays *= sample_rate/SPEED_OF_SOUND
        
        assert (expected_delays == darray.delays).all()
        assert darray.delay_bounds == (-6, 5)
