import numpy as np

class PhasedArrayModel(object):
    def __init__(self, M, N, d_x, d_y, P, D=2, theta_res=np.pi/1000, phi_res=np.pi/1000):
        """
        Inputs:
            M (int) - size of the array along the x-axis
            N (int) - size of the array along the y-axis
            d_x (float) - spacing of elements along the x-axis
            d_y (float) - spacing of elements along the y-axis
            P (int) - number of point sources
            D (1 or 2) - number of dimensions of our phased array
            theta_res (float) - resolution of theta scan
            phi_res (float) - resolution of phi scan
        """
        self.M = M 
        self.N = N 
        self.d_x = d_x
        self.d_y = d_y
        self.P = P
        self.D = D
        self.theta_res = theta_res
        self.phi_res = phi_res
        self.u, self.v = self._initialize_scan_angles()


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
        return u, v
        
    

    