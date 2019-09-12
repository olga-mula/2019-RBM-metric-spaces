# Computing the H^{-1} distance

import numpy as np
import scipy

def norm_Hminus1(u, x):
    """
        Computes H^{-1} norm of u by computing the H^1_0 norm of phi, solution of
            - phi''(x) = u(x) for x in [xmin, xmax]
            phi = 0 for at xmin, xmax
    """

    u_spline = scipy.interpolate.UnivariateSpline(x, u, k=3)

    xmin = x[0]
    xmax = x[-1]
    x = np.linspace(xmin, xmax, num=1e5, endpoint=True)
    dx = x[1] - x[0]

    rhs = u_spline(x)
    rhs[0] = 0.0
    rhs[-1]= 0.0

    # Stiffness matrix of -Laplacian: we give it in banded form
    # Principal diagonal, upper and lower diagonals
    d = 2/(dx*dx) * np.ones(len(x)); d[0] = 1.; d[-1]=1.
    ud = -1/(dx*dx) * np.ones(len(x)); ud[0] = 0.
    ld = -1/(dx*dx) * np.ones(len(x)); ld[-1] = 0.
    K = np.matrix([ud, d, ld])
    # computing the potential by solving Poisson
    phi = scipy.linalg.solve_banded((1,1), K, rhs)

    # computing the gradient of phi by finite differences
    d_phi = np.zeros(len(x))
    d_phi[1:-1] = (np.diff(phi)[:-1] + np.diff(phi)[1:])/(2*dx)
    d_phi[0] = (phi[1] - phi[0])/dx
    d_phi[-1] = (phi[-1] - phi[-2])/dx

    # distance is the H^1 seminorm of phi:

    norm_Hminus1_squared = np.sum(dx*d_phi**2)
    # for iComp in range(sizeOfVec):
    #     distance += dx* (dxPhi[iComp]*dxPhi[iComp])
    distance = np.sqrt(norm_Hminus1_squared)

    return distance