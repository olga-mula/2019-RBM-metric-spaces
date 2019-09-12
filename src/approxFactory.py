import numpy as np

class ApproxFactory():
	@staticmethod
	def g_inv(threshold, x, fx):
		"""
			Generalized inverse:
			$$
				f^{-1}(t) = inf{ x in Omega : f(x)>t }
			$$
			Here threshold <=> t.
		"""
		if np.isscalar(threshold):
			x1 = ApproxFactory.g_inv_scalar(threshold, x, fx)
			return x1
		else:
			X = list()
			for t in threshold:
				x1 = ApproxFactory.g_inv_scalar(t, x, fx)
				X.append(x1)
			return np.asarray(X)

	@staticmethod
	def g_inv_scalar(threshold, x, fx):
		if np.all(fx <= threshold):
			return x[len(x)-1]
		else:
			idx1 = np.argmax(fx>=threshold)
			idx0 = idx1 - 1
			if idx1==0:
				idx0 = idx1

			x0 = x[idx0]
			x1 = x[idx1]
			Fx0 = fx[idx0]
			Fx1 = fx[idx1]

			if Fx1 == Fx0:
				return x1
			else:
				return x0 + (threshold - Fx0)/(Fx1 - Fx0) * (x1-x0)

	@staticmethod
	def fd_deriv(dx, fx):
		"""
			Generalized inverse:
			$$
				f^{-1}(t) = inf{ x in Omega : f(x)>t }
			$$
			Here threshold <=> t.
		"""
		nx = len(fx)
		diff_f = np.zeros(nx)
		diff_f[1:-1] = (np.diff(fx)[:-1] + np.diff(fx)[1:])/(2*dx)
		
		diff_f[0] = (fx[1] - fx[0])/dx
		diff_f[-1] = (fx[-1] - fx[-2])/dx

		return diff_f

