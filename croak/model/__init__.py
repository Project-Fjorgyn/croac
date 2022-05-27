import numpy as np

class PhasedArrayModel(object):
    def __init__(self, omega, M, N, d_x, d_y, P, D=2, theta_res=np.pi/1000, phi_res=np.pi/1000):
        """
        Inputs:
            omega (float) - the wavelength
            M (int) - size of the array along the x-axis
            N (int) - size of the array along the y-axis
            d_x (float) - spacing of elements along the x-axis
            d_y (float) - spacing of elements along the y-axis
            P (int) - number of point sources
            D (1 or 2) - number of dimensions of our phased array
            theta_res (float) - resolution of theta scan
            phi_res (float) - resolution of phi scan
        """
        self.omega = omega
        self.k = 2 * np.pi / omega
        self.M = M 
        self.N = N if D == 2 else 1
        self.d_x = d_x
        self.d_y = d_y
        self.P = P
        self.D = D
        self.theta_res = theta_res
        self.phi_res = phi_res
        # array info
        self.base_u, self.base_v, self.theta, self.phi = \
         self._initialize_scan_angles()
        self.u, self.v, self.m, self.n = self._copy_uv_over_array_and_sources()
        # source info
        self.set_source_info(np.zeros(P), np.zeros(P), np.ones(P), np.zeros(P))

    def _initialize_scan_angles(self):
        """
        This method will initialize a grid of u,v pairs
        that will represent all of the scan angles that
        we'll be looking over.
        """
        # we start by creating our individual axes
        if self.D == 2:
            theta = np.arange(0, np.pi/2, self.theta_res)
            phi = np.arange(0, 2*np.pi, self.phi_res)
        else:
            theta = np.arange(-np.pi/2, np.pi/2, self.theta_res)
            phi = np.array([0])

        # and then build our mesh grid
        theta, phi = np.meshgrid(theta, phi)
        # which we then convert to u, v
        u = np.sin(theta)*np.cos(phi)
        v = np.sin(theta)*np.sin(phi)
        flat_shape = u.shape[0] * u.shape[1]
        return (
            u.reshape(flat_shape), 
            v.reshape(flat_shape),
            theta.reshape(flat_shape),
            phi.reshape(flat_shape)
        )

    def _copy_uv_over_array_and_sources(self):
        """
        This method takes our u and v and projects them
        over the array so we can vectorize the
        computations in our model. It also gives us back
        the m and n in a vectorized fashion. 
        """
        # note that because the problem is separable we can 
        # grid out u and v separably 
        u, _, m = np.meshgrid(self.base_u, range(self.P), range(1, self.M + 1), indexing='ij')
        v, _, n = np.meshgrid(self.base_v, range(self.P), range(1, self.N + 1), indexing='ij')
        return u, v, m, n

    def set_source_info(self, theta, phi, a, psi):
        for arr in (theta, phi, a, psi):
            assert arr.shape == (self.P,)

        self.source_u = np.sin(theta)*np.cos(phi)
        self.source_v = np.sin(theta)*np.sin(phi)
        self.a = a 
        self.phases = np.exp(1j*psi)

        self.su, self.sv = self._copy_source_positions_over_array_sources()
        
    def _copy_source_positions_over_array_sources(self):
        """
        This method projects our amplitudes, phases,
        and source positions (in uv space) over the grid
        so that they are ready for vectorized computation.
        """
        _, u, _ = np.meshgrid(self.base_u, self.source_u, range(self.M), indexing='ij')
        _, v, _ = np.meshgrid(self.base_v, self.source_v, range(self.N), indexing='ij')
        return u, v

    def _compute_e(self):
        e_x = np.exp(1j*self.k*self.m*self.d_x*(self.su-self.u))
        e_y = np.exp(1j*self.k*self.n*self.d_y*(self.sv-self.v))
        return e_x, e_y

    def compute_I(self):
        self.e_x, self.e_y = self._compute_e()
        # sum over the array for each of the sources
        sum_over_array = np.sum(self.e_x, axis=2) * np.sum(self.e_y, axis=2)
        # get the contributions per source
        source_contributions = sum_over_array * self.a * self.phases
        # sum over the sources
        self.I = np.sum(source_contributions, axis=1)
        return self.I

    def compute_P(self):
        I = self.compute_I()
        self.P = np.real(I * np.conjugate(I))
        return self.P


    