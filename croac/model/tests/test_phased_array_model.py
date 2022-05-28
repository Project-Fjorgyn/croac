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
        assert (model.phases == np.array([np.exp(3j), np.exp(4j)])).all()

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

    def test_compute_I(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=np.pi/4, phi_res=np.pi
        )
        model.set_source_info(
            np.array([np.pi/4, 0]),
            np.array([np.pi/2, 0]),
            np.array([1, 2]),
            np.array([3, 4])
        )
        I = model.compute_I()
        assert I.shape == (4,)
        assert (model.I == I).all()

    def test_compute_P(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=np.pi/4, phi_res=np.pi
        )
        model.set_source_info(
            np.array([np.pi/4, 0]),
            np.array([np.pi/2, 0]),
            np.array([1, 2]),
            np.array([3, 4])
        )
        P = model.compute_P()
        assert P.shape == (4,)
        assert (model.P == P).all()
        assert (P == model.I * np.conjugate(model.I)).all()

    def test_ingest_new_scan_positions(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=0.5, phi_res=0.5
        )
        model._ingest_new_scan_positions(np.array([[0, 0], [np.pi/4, np.pi/2]]))
        assert (model.theta == np.array([0, np.pi/4])).all()
        assert (model.phi == np.array([0, np.pi/2])).all()
        assert (model.base_u == np.sin([0, np.pi/4]) * np.cos([0, np.pi/2])).all()
        assert (model.base_v == np.sin([0, np.pi/4]) * np.sin([0, np.pi/2])).all()
        assert model.u.shape == (2, 2, 2)
        assert model.v.shape == (2, 2, 3)
        assert model.su.shape == (2, 2, 2) 
        assert model.sv.shape == (2, 2, 3)

    def test_compute_E(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=0.5, phi_res=0.5
        )
        model.P = np.array([1, 2, 3])
        model.O = np.array([3, 2, 1])
        E = model.compute_E()
        assert E == model.E
        assert E == np.sqrt(8/3)

    def test_compute_gradient_I_u(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=0.5, phi_res=0.5
        )
        model.k = 2
        model.d_x = 3
        *_, model.m = np.meshgrid(range(1), range(2), range(1, 3), indexing='ij')
        model.e_x = np.array([[[1, 2],
                                [3, 4]]])
        model.sum_e_y = np.array([[1, 2]])
        model.a = np.array([3,4])
        model.phases = np.array([1,2])
        expected = np.array([[
            3 * 1 * 1 * (1j * 2 * 3 * 1 * 1 + 1j * 2 * 3 * 2 * 2),
            4 * 2 * 2 * (1j * 2 * 3 * 1 * 3 + 1j * 2 * 3 * 2 * 4)
        ]])
        assert (expected == model.compute_gradient_I_u()).all()

    def test_compute_gradient_I_v(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=0.5, phi_res=0.5
        )
        model.k = 2
        model.d_y = 3
        *_, model.n = np.meshgrid(range(1), range(2), range(1, 3), indexing='ij')
        model.e_y = np.array([[[1, 2],
                                [3, 4]]])
        model.sum_e_x = np.array([[1, 2]])
        model.a = np.array([3,4])
        model.phases = np.array([1,2])
        expected = np.array([[
            3 * 1 * 1 * (1j * 2 * 3 * 1 * 1 + 1j * 2 * 3 * 2 * 2),
            4 * 2 * 2 * (1j * 2 * 3 * 1 * 3 + 1j * 2 * 3 * 2 * 4)
        ]])
        assert (expected == model.compute_gradient_I_v()).all()

    def test_compute_gradient_I_a(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=0.5, phi_res=0.5
        )
        model.I_p = np.array([1, 2])
        model.a = np.array([3, 4])
        expected = np.array([1/3, 2/4])
        assert (expected == model.compute_gradient_I_a()).all()

    def test_compute_gradient_I_phases(self):
        model = PhasedArrayModel(
            omega=2, M=2, N=3, P=2, d_x=1, d_y=2, D=2, theta_res=0.5, phi_res=0.5
        )
        model.I_p = np.array([1, 2])
        expected = np.array([1j, 2j])
        assert (expected == model.compute_gradient_I_phases()).all()
