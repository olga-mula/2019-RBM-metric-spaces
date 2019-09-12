import numpy as np
from scipy.integrate import ode

from viscousburgers1D import Grid1d

class SimulationCamassaHolm():
	def __init__(self, grid):
		self.grid = grid
		self.p = None # Mass of one peakon
		self.q = None # Position of one peakon
		self.alpha = None

	def init_cond(self, q, p, alpha):
		""" initial conditions -- two peakons """
		self.p = [p, 1.0 - p]
		self.q = [q, -5.]
		self.alpha = alpha

	def u(self, z):
		def K(x):
			return np.exp(-abs(x))

		x = self.grid.x
		P = z[:int(len(z)/2)]
		Q = z[int(len(z)/2):]
		U = np.zeros(len(x))

		for (p, q) in list(zip(P, Q)):
			U = U + 0.5 * p * K((x-q)/self.alpha)

		return U

	def evolve(self, dt, T):
		""" We propagate solution from 0 to T
			with explicit RK method of order 8.
			For this, we propagate first the Lagrangian coordinates
			by solving a system of the form
				dz/dt = f(t,z,...)
			with the DOPRI method.
			Then, we use the Lagrangian coordinates to assemble the solution.
		"""
		def f(t, z):

			def K(x):
				return np.exp(-abs(x))

			def dHdp(p, q, P, Q):
				return 0.5*np.sum([pi*K((q-qi)/self.alpha) for (pi, qi) in list(zip(P,Q))])

			def dHdq(p, q, P, Q):
				return -0.5/self.alpha*p*np.sum([pi*K((q-qi)/self.alpha)*np.sign(q-qi) for (pi, qi) in list(zip(P,Q))])

			P = z[:int(len(z)/2)]
			Q = z[int(len(z)/2):]

			list_P = list()
			list_Q = list()

			for (p, q) in list(zip(P, Q)):
				list_P.append( -dHdq(p, q, P, Q) )
				list_Q.append( dHdp(p, q, P, Q) )

			return list_P + list_Q

		ode_solver = ode(f)
		ode_solver.set_integrator('dop853')
		t0 = 0.
		z0 = self.p + self.q
		ode_solver.set_initial_value(z0, t0)

		t = 0.

		while t < T and ode_solver.successful():
			if (t + dt > T):
				dt = T - t
			t = t + dt
			ode_solver.integrate(t)

		z = ode_solver.y

		# mass = np.sum(self.u(z))*self.grid.dx
		# print(mass)

		# print(t)

		return self.u(z)








