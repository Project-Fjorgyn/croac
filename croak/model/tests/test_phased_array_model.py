import unittest
import numpy as np

from .. import (
    PhasedArrayModel
)

class TestPhasedArrayModel(unittest.TestCase):
    def test_scan_angles_2D(self):
        model = PhasedArrayModel(
            1, 5, 5, 1, 1, 1, D=2, theta_res=0.5, phi_res=0.5
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
            1, 5, 5, 1, 1, 1, D=1, theta_res=0.5, phi_res=0.5
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
            omega=1, M=5, N=10, P=2, d_x=1, d_y=1, D=2, theta_res=0.5, phi_res=0.5
        )
        assert model.u.shape == (52, 2, 5)
        assert model.v.shape == (52, 2, 10)
        assert set(model.m.reshape(52 * 2 * 5)) == set(range(1, 6))
        assert set(model.n.reshape(52 * 2 * 10)) == set(range(1, 11))

    def test_copy_source_positions_over_array_source(self):
        model = PhasedArrayModel(
            omega=1, M=5, N=10, P=2, d_x=1, d_y=1, D=2, theta_res=0.5, phi_res=0.5
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

    def test_compute_e(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=np.pi/4, phi_res=np.pi
        )
        model.set_source_info(
            np.array([np.pi/4, 0]),
            np.array([np.pi/2, 0]),
            np.array([1, 2]),
            np.array([3, 4])
        )
        e_x, e_y = model._compute_e()
        assert e_x.shape == (4, 2, 2)
        assert e_y.shape == (4, 2, 3)

        dif = e_x[0,0,1] - np.exp(
            1j * np.pi * 2 * 1 * (np.sin(np.pi/4)*np.cos(np.pi/2) - model.u[0,0,1])
        )
        assert dif * np.conjugate(dif) < 10 ** -16

        dif = e_x[0,1,1] - np.exp(
            1j * np.pi * 2 * 1 * (np.sin(0)*np.cos(0) - model.u[0,1,1])
        )
        assert dif * np.conjugate(dif) < 10 ** -16

        dif = e_x[2,1,1] - np.exp(
            1j * np.pi * 2 * 1 * (np.sin(0)*np.cos(0) - model.u[2,1,1])
        )
        assert dif * np.conjugate(dif) < 10 ** -16

        dif = e_y[0,0,1] - np.exp(
            1j * np.pi * 2 * 2 * (np.sin(np.pi/4)*np.sin(np.pi/2) - model.v[0,0,1])
        )
        assert dif * np.conjugate(dif) < 10 ** -16

        dif = e_y[0,0,2] - np.exp(
            1j * np.pi * 2 * 3 * (np.sin(np.pi/4)*np.sin(np.pi/2) - model.v[0,0,2])
        )
        assert dif * np.conjugate(dif) < 10 ** -16