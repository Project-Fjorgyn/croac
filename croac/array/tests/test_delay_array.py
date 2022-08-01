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

    def test_sweep(self):
        antenna_x = np.array([1, 2])
        antenna_y = np.array([3, 2])
        theta = np.array([0, np.pi/4])
        phi = np.array([np.pi/4, np.pi])
        sample_rate = 1000

        darray = DelayArray(antenna_x, antenna_y, theta, phi, sample_rate)

        signals = np.array(
            [
                list(range(13)),
                list(reversed(range(13)))
            ]
        )

        darray.ingest_signals(signals)
        darray.sweep()

        expected_result = np.array(
            [
                [
                    [6 + 6, 7 + 5], # phi 1,
                    [6 + 6, 7 + 5], # phi 2
                ], # theta 1
                [
                    [0.1691 + 11.831, 1.1691 + 10.831], # phi 1,
                    [8.0612 + 1.877, 9.0612 + 0.877], # phi 2
                ] # theta 2
            ]
        )
        assert (abs(darray.result - expected_result) < 10 ** -3).all()

    def test_decompose(self):
        antenna_x = np.array([1, 2])
        antenna_y = np.array([3, 2])
        theta = np.array([0, np.pi/4])
        phi = np.array([np.pi/4, np.pi])
        sample_rate = 1000

        darray = DelayArray(antenna_x, antenna_y, theta, phi, sample_rate)

        t = np.arange(0, 1, 1/1000) 
        darray.result = np.array(
            [
                [
                    np.sin(2 * np.pi * (10 * t))
                ],
                [
                    np.sin(2 * np.pi * (20 * t))
                ]
            ]
        )

        darray.decompose()

        expected = np.zeros((2, 1, 500))
        expected[0,0,10] = 0.5
        expected[1,0,20] = 0.5
        assert (abs(expected - darray.decomposition) < 10 ** -10).all()
