import numpy as np

class PhasedArrayModel(object):
    def __init__(
        self, omega, M, N, d_x, d_y, S, D=2, 
        theta_res=np.pi/1000, phi_res=np.pi/1000,
        theta_gate=0.05, phi_gate=0.05, a_gate=0.1,
        psi_gate=0, iterations=1000, attempts=10,
        err_thresh=10**-1
    ):
        """
        Inputs:
            omega (float) - the wavelength
            M (int) - size of the array along the x-axis
            N (int) - size of the array along the y-axis
            d_x (float) - spacing of elements along the x-axis
            d_y (float) - spacing of elements along the y-axis
            S (int) - number of point sources
            D (1 or 2) - number of dimensions of our phased array
            theta_res (float) - resolution of theta scan
            phi_res (float) - resolution of phi scan
            theta_gate (float) - maximum step for theta
            phi_gate (float) - maximum step for phi
            a_gate (float) - maximum step for a
            psi_gate (float) - maximum step for psi
            iterations (int) - number of iterations to fit each guess
            attempts (int) - max number of guesses
            err_thresh (float) - error to consider equivalent to zero
        """
        self.omega = omega
        self.k = 2 * np.pi / omega
        self.M = M 
        self.N = N if D == 2 else 1
        self.d_x = d_x
        self.d_y = d_y
        self.S = S
        self.D = D
        self.theta_res = theta_res
        self.phi_res = phi_res
        # array info
        self._initialize_scan_angles()
        self._copy_uv_over_array_and_sources()
        # source info
        self.set_source_info(np.zeros(S), np.zeros(S), np.ones(S), np.zeros(S))
        self.set_gates(theta_gate, phi_gate if self.D == 2 else 0, a_gate, psi_gate)
        self.iterations = iterations
        self.attempts = attempts
        self.err_thresh = err_thresh

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
        self.base_u = u.reshape(flat_shape)
        self.base_v = v.reshape(flat_shape)
        self.theta = theta.reshape(flat_shape)
        self.phi = phi.reshape(flat_shape)
        return (
            self.base_u,
            self.base_v,
            self.theta,
            self.phi
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
        self.u, _, self.m = np.meshgrid(self.base_u, range(self.S), range(1, self.M + 1), indexing='ij')
        self.v, _, self.n = np.meshgrid(self.base_v, range(self.S), range(1, self.N + 1), indexing='ij')
        return self.u, self.v, self.m, self.n

    def set_source_info(self, theta, phi, a, psi):
        for arr in (theta, phi, a, psi):
            assert arr.shape == (self.S,)

        self.source_theta = theta
        self.source_phi = phi
        self.source_psi = psi
        self.source_u = np.sin(theta)*np.cos(phi)
        self.source_v = np.sin(theta)*np.sin(phi)
        self.a = a 
        self.phases = np.exp(1j*psi)

        self._copy_source_positions_over_array_sources()
        
    def _copy_source_positions_over_array_sources(self):
        """
        This method projects our amplitudes, phases,
        and source positions (in uv space) over the grid
        so that they are ready for vectorized computation.
        """
        _, self.su, _ = np.meshgrid(self.base_u, self.source_u, range(self.M), indexing='ij')
        _, self.sv, _ = np.meshgrid(self.base_v, self.source_v, range(self.N), indexing='ij')
        return self.su, self.sv

    def _compute_e(self):
        self.e_x = np.exp(1j*self.k*self.m*self.d_x*(self.su-self.u))
        self.e_y = np.exp(1j*self.k*self.n*self.d_y*(self.sv-self.v))
        return self.e_x, self.e_y

    def _compute_I(self):
        # sum over the array for each of the sources
        self.sum_e_x = np.sum(self.e_x, axis=2)
        self.sum_e_y = np.sum(self.e_y, axis=2)
        sum_over_array = self.sum_e_x * self.sum_e_y
        # get the contributions per source
        self.I_p = sum_over_array * self.a * self.phases
        # sum over the sources
        self.I = np.sum(self.I_p, axis=1)
        return self.I

    def _compute_P(self):
        self.P = np.real(self.I * np.conjugate(self.I))
        return self.P

    def compute_P(self):
        self._compute_e()
        self._compute_I()
        return self._compute_P()

    def _ingest_new_scan_positions(self, X):
        self.theta = X[:,0]
        self.phi = X[:,1]
        self.base_u = np.sin(self.theta)*np.cos(self.phi)
        self.base_v = np.sin(self.theta)*np.sin(self.phi)
        self._copy_uv_over_array_and_sources()
        self.set_source_info(self.source_theta, self.source_phi, self.a, self.phases)

    def _compute_E(self):
        self.Errors = self.P - self.O
        self.E = np.sqrt(np.sum(self.Errors ** 2)/self.O.shape[0])
        return self.E

    def compute_E(self):
        self.compute_P()
        return self._compute_E()

    def _compute_gradient_I_u(self):
        summand = 1j*self.k*self.d_x*self.m*self.e_x
        V = self.a * self.phases * self.sum_e_y
        self.grad_I_u = V * np.sum(summand, axis=2)
        return self.grad_I_u

    def _compute_gradient_I_v(self):
        summand = 1j*self.k*self.d_y*self.n*self.e_y
        U = self.a * self.phases * self.sum_e_x
        self.grad_I_v = U * np.sum(summand, axis=2)
        return self.grad_I_v

    def _compute_gradient_I_a(self):
        self.grad_I_a = self.I_p / self.a
        return self.I_p / self.a

    def _compute_gradient_I_phases(self):
        self.grad_I_phases = self.I_p * 1j
        return self.grad_I_phases

    def _compute_gradient_I_theta(self):
        # we have to transpose to broadcast positions correctly
        self.grad_I_theta = (
            self.grad_I_u.T * np.cos(self.theta) * np.cos(self.phi)
            + self.grad_I_v.T * np.cos(self.theta) * np.sin(self.phi)
        ).T
        return self.grad_I_theta

    def _compute_gradient_I_phi(self):
        # we have to transpose to broadcast positions correctly
        self.grad_I_phi = (
            self.grad_I_v.T * np.sin(self.theta) * np.cos(self.phi)
            - self.grad_I_u.T * np.sin(self.theta) * np.sin(self.phi)
        ).T 
        return self.grad_I_phi

    def _compute_gradient_I(self):
        # note that at this stage we have to
        # transpose the matrix so that the last
        # axis is our scan positions axis
        self.grad_I = np.concatenate(
            (
                self.grad_I_theta,
                self.grad_I_phi,
                self.grad_I_a,
                self.grad_I_phases
            ), axis=1
        ).T
        return self.grad_I

    def _compute_gradient_P(self):
        self.grad_P = np.real(
            self.grad_I * np.conjugate(self.I)
            + np.conjugate(self.grad_I) * self.I
        )
        return self.grad_P

    def _compute_gradient(self):
        self.grad = (
            1/(self.E * self.O.shape[0])
            * np.sum(self.grad_P * self.Errors, axis=1)
        )
        return self.grad

    def compute_gradient(self):
        self.compute_P()
        self._compute_E()
        self._compute_gradient_I_u()
        self._compute_gradient_I_v()
        self._compute_gradient_I_a()
        self._compute_gradient_I_phases()
        self._compute_gradient_I_theta()
        self._compute_gradient_I_phi()
        self._compute_gradient_I()
        self._compute_gradient_P()
        return self._compute_gradient()

    def set_target(self, X, y):
        self._ingest_new_scan_positions(X)
        self.O = y

    def set_gates(self, theta, phi, a, psi, step_down=2., min_step_size=0.001):
        # the gates are effectively a maximum
        # learning step for each dimension
        # we do this because the order of magnitude
        # of our error does not correspond to the
        # same for the ranges of our various params
        self.gates = np.concatenate((
            theta * np.ones(self.S),
            phi * np.ones(self.S),
            a * np.ones(self.S),
            psi * np.ones(self.S)
        ))
        self.step_down = step_down
        self.min_step_size = min_step_size

    def step(self):
        last_E = self.compute_E()
        grad = self.compute_gradient()
        divisor = np.max(np.abs(grad[self.gates != 0]) / self.gates[self.gates != 0])
        # for the ones we've frozen, set the step to
        # zero
        steps = grad / divisor * (self.gates != 0)
        self.set_source_info(
            self.source_theta - steps[0:self.S], 
            self.source_phi - steps[self.S:2*self.S], 
            self.a - steps[2*self.S:3*self.S], 
            self.source_psi - steps[3*self.S:4*self.S]
        )
        new_E = self.compute_E()
        # if we followed the gradient and got
        # worse error, we're moving too fast
        if new_E > last_E:
            self.gates = np.maximum(
                self.gates / self.step_down,
                self.min_step_size * np.ones(4*self.S)
            )

    def _make_guess(self):
        # we guess the positions on the basis of where
        # the peaks in our observed distribution are
        thetas = np.array([
            np.random.choice(
                self.theta, p=self.O/np.sum(self.O)
            )
            for _ in range(self.S)
        ])
        if self.D == 2:
            phis= np.array([
                np.random.choice(
                    self.phi, p=self.O/np.sum(self.O)
                )
                for _ in range(self.S)
            ])
        else:
            phis = np.zeros(self.S)
        return thetas, phis, np.ones(self.S), np.zeros(self.S)

    def fit(self, X, y):
        self.set_target(X, y)
        best_error = float('inf')
        best_solution = None
        for _ in range(self.attempts):
            guess = self._make_guess()
            self.set_source_info(*guess)
            for _ in range(self.iterations):
                self.step()
            self.compute_E()
            if best_error > self.E:
                best_error = self.E
                best_solution = (
                    self.source_theta,
                    self.source_phi,
                    self.a,
                    self.source_psi
                )
            if best_error <= self.err_thresh:
                break
        self.set_source_info(*best_solution)
        self.compute_E()
