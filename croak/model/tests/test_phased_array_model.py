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
        flat_shape = u.shape[0] * u.shape[1]
        assert model.base_u.shape == (13*4,)
        assert model.base_v.shape == (13*4,)
        assert (u.reshape(flat_shape) == model.base_u).all()
        assert (v.reshape(flat_shape) == model.base_v).all()

    def test_scan_angles_1D(self):
        model = PhasedArrayModel(
            5, 5, 1, 1, 1, D=1, theta_res=0.5, phi_res=0.5
        )
        theta = [-np.pi/2, -np.pi/2 + 0.5, -np.pi/2 + 1., -np.pi/2 + 1.5, -np.pi/2 + 2., -np.pi/2 + 2.5, -np.pi/2 + 3.]
        phi = [0.]
        theta, phi = np.meshgrid(theta, phi)
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        assert model.base_u.shape == (7,)
        assert model.base_v.shape == (7,)
        assert (u == model.base_u).all()
        assert (v == model.base_v).all()
        assert (model.base_v == 0).all()

    def test_copy_uv_over_array_and_sources(self):
        model = PhasedArrayModel(
            M=5, N=10, P=2, d_x=1, d_y=1, D=2, theta_res=0.5, phi_res=0.5
        )
        assert model.u.shape == (52, 2, 5)
        assert model.v.shape == (52, 2, 10)
        assert set(model.m.reshape(52 * 2 * 5)) == set(range(5))
        assert set(model.n.reshape(52 * 2 * 10)) == set(range(10))

    def test_copy_source_positions_over_array_source(self):
        model = PhasedArrayModel(
            M=5, N=10, P=2, d_x=1, d_y=1, D=2, theta_res=0.5, phi_res=0.5
        )
        model.set_source_info(
            np.array([np.pi/4, 0]),
            np.array([np.pi/2, 0]),
            np.array([1, 2]),
            np.array([3, 4])
        )
        assert (np.array([
            np.sin(np.pi/4)*np.cos(np.pi/2),
            np.sin(0)*np.cos(0)
        ]) == model.source_u).all()
        assert (np.array([
            np.sin(np.pi/4)*np.sin(np.pi/2),
            np.sin(0)*np.sin(0)
        ]) == model.source_v).all()
        assert (model.a == np.array([1, 2])).all()
        assert (model.psi == np.array([3, 4])).all()

        assert model.su.shape == (52, 2, 5)
        assert model.sv.shape == (52, 2, 10)
        
        assert (
            set(model.su[:,0,:].reshape(52*5)) 
            == set([np.sin(np.pi/4)*np.cos(np.pi/2)])
        )
        assert (
            set(model.su[:,1,:].reshape(52*5)) 
            == set([np.sin(0)*np.cos(0)])
        )

        assert (
            set(model.sv[:,0,:].reshape(52*10)) 
            == set([np.sin(np.pi/4)*np.sin(np.pi/2)])
        )
        assert (
            set(model.sv[:,1,:].reshape(52*10)) 
            == set([np.sin(0)*np.sin(0)])
        )
