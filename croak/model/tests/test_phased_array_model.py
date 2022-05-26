import unittest
import numpy as np

from .. import (
    PhasedArrayModel
)

class TestPhasedArrayModel(unittest.TestCase):
    def test_scan_angles_2D(self):
        model = PhasedArrayModel(
            5, 5, 1, 1, 1, D=2, theta_res=0.5, phi_res=0.5
        )
        theta = [0., 0.5, 1., 1.5]
        phi = [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.]
        theta, phi = np.meshgrid(theta, phi)
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        assert u.shape == (13, 4)
        assert v.shape == (13, 4)
        assert (u == model.u).all()
        assert (v == model.v).all()

    def test_scan_angles_1D(self):
        model = PhasedArrayModel(
            5, 5, 1, 1, 1, D=1, theta_res=0.5, phi_res=0.5
        )
        theta = [-np.pi/2, -np.pi/2 + 0.5, -np.pi/2 + 1., -np.pi/2 + 1.5, -np.pi/2 + 2., -np.pi/2 + 2.5, -np.pi/2 + 3.]
        phi = [0.]
        theta, phi = np.meshgrid(theta, phi)
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        assert u.shape == (1, 7)
        assert v.shape == (1, 7)
        assert (u == model.u).all()
        assert (v == model.v).all()
