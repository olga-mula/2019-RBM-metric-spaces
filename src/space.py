import numpy as np
import scipy.linalg
import scipy.interpolate
import time
import sys
sys.path.append('contributions/')
from cvxopt import solvers, matrix

from snapshots import ProbaMeasure, Burgers, KdV, ViscousBurgers
from approxFactory import ApproxFactory
from HminusOne import norm_Hminus1
# from multiprocessing import Pool
# from itertools import repeat


class W2space():
	"""
		L2-Wasserstein space (P_2(Omega), W_2 ) of probability measures
	"""
	def __init__(self):
		pass

	def dist(self, mu1, mu2, nQuad=200):
		"""
			L2 Wasserstein distance between two probability densities.
			We assume densities have same image support.
			We use a quadrature with uniform points.
		"""
		total_mass = mu1.m
		icdf1 = mu1.icdf
		icdf2 = mu2.icdf
		return np.sqrt(total_mass*np.mean((icdf1-icdf2)**2))

	# def barycenter(self, measures):
	# 	"""
	# 		Computation of the barycenter of a set of measures given in the
	# 		form of a list.
	# 	"""
	# 	print('Begin computation of barycenter')
	# 	start = time.time()
	# 	N = len(measures)
	# 	D = np.zeros((N,N))
	# 	for i, b in enumerate(measures):
	# 		D[:, i] = [self.dist(mu, b) for mu in measures]
		 
	# 	imax = np.argmax([sum(x) for x in zip(*D**2)]) # Remark: the summation is for columns of D**2
	# 	end = time.time()
	# 	print('End computation of barycenter. Took '+str(end-start)+' secs. Measure index '+str(imax))
	# 	return imax, measures[imax]

	def barycenter(self, measures):
		"""
			Computation of the barycenter of a set of measures given in the
			form of a list.
		"""
		print('Begin computation of barycenter')
		start = time.time()

		N = len(measures)
		x = measures[0].x
		r = measures[0].r
		Nr = r.shape[0]
		D = np.zeros((N,Nr))

		for i, mu in enumerate(measures):
			D[i,:] = mu.icdf

		barycenter_icdf = np.mean(D, axis=0)

		end = time.time()
		print('End computation of barycenter. Took '+str(end-start)+' secs.')

		return ProbaMeasure(icdf=barycenter_icdf, m=measures[0].m)

class LogPCAspace():
	"""
		Reduced basis for the log snapshots.
		An n dimensional space of L2([0,1]).
	"""
	def __init__(self, rbConstructionStrategy, n):
		"""
			dictionary: set of snapshots
			barycenter: of the dictionary
			U: The j-th PCA vector is given by
				$$
					sum_j U[:,j].T snapshot[j]
				$$
			G: Grammian of the log snapshots.
			$$
				G_{i,j} = < log s[i], log s[j] >_{L2([0,mass])}
			$$
			err: Error of PCA
			dim: Dimension of the reduced space
			grammian: Grammian of the reduced basis

		"""

		self.dictionary, self.barycenter, self.U, self.G, self.err \
			= rbConstructionStrategy.generateBasis(n)
		self.dim      = n
		self.grammian = np.dot(np.dot(self.U.T, self.G), self.U)
		L = np.linalg.cholesky(self.grammian)
		self.L_inv = scipy.linalg.lapack.dtrtri(L)[0]
		self.M = self._computeM()
		self.P, self.C = self._prepare_local_interpolation()

	def _computeM(self):
		total_mass = self.dictionary[0].m
		nr = len(self.dictionary[0].r)
		N = len(self.dictionary)
		M = np.zeros((nr, N))
		for j, snapshot in enumerate(self.dictionary):
			M[:, j] = snapshot.log(self.barycenter)
		return M

	def _compute_coefs_projection(self, density):
		# rhs: ( <log density, basis[i]>_{L2([0,1])} )_{i=1,..,dim}
		nr = len(density.icdf)
		rhs= np.dot(np.dot(self.M, self.U).T, density.log(self.barycenter))/nr

		# We solve system G x = rhs
		# Option 1: seems more accurate but slower
		# c = np.linalg.solve(self.grammian, rhs)
		# Option 2: seems a bit less accurate but faster
		c = np.dot(self.L_inv.T, np.dot(self.L_inv, rhs))

		return c

	def project(self, density):
		c = self._compute_coefs_projection(density)
		return np.dot(np.dot(self.M, self.U),c)

	def _prepare_local_interpolation(self):
		n_snapshots = len(self.dictionary)
		n_params    = len(self.dictionary[0].param)

		P = np.zeros((n_snapshots, n_params)) # Matrix of snapshot parameters
		C = np.zeros((n_snapshots, self.dim)) # Matrix of coefs projection

		for i, snapshot in enumerate(self.dictionary):
			P[i, :] = snapshot.param
			C[i, ]  = self._compute_coefs_projection(snapshot)

		return P, C

	def _build_interpolator(self, param, k=10):
		rbfi_coef = list()

		# Find k nearest-neighbors
		dE = (self.P-param)/np.max(self.P, axis=0)
		E = np.linalg.norm(dE, axis=1)
		idx = np.argpartition(E, k)[:k]
		P_neighbors = self.P[idx, :]
		C_neighbors = self.C[idx, :]

		for i in range(self.dim):
			rbfi_coef.append(self._build_rbf(P_neighbors, C_neighbors[:,i]))

		return rbfi_coef

	def _build_rbf(self, P, d, function='multiquadric', smooth=0.):
		# This is ugly but it is due to the form of the input of interpolate.Rbf...
		if P.shape[1] == 3: # 3 parameters
			return scipy.interpolate.Rbf(P[:,0], P[:,1], P[:,2], d, function=function, smooth=smooth)
		elif P.shape[1] == 2: # 2 parameters
			return scipy.interpolate.Rbf(P[:,0], P[:,1], d, function=function, smooth=smooth)
		elif P.shape[1] == 1: # 1 parameter
			return scipy.interpolate.Rbf(P[:,0], d, function=function, smooth=smooth)
		else:
			raise Exception('Number of parameters not supported for interpolation.')

	def _compute_coefs_interpolation(self, density):
		c = np.zeros(self.dim)
		param = density.param

		# Interpolator over neighboring coefficients
		rbfi_coef = self._build_interpolator(param)

		if len(param) == 3:
			for i in range(self.dim):
				c[i] = rbfi_coef[i](param[0], param[1], param[2])
		elif len(param) == 2:
			for i in range(self.dim):
				c[i] = rbfi_coef[i](param[0], param[1])
		elif len(param) == 1:
			for i in range(self.dim):
				c[i] = rbfi_coef[i](param[0])
		else:
			raise Exception('Number of parameters not supported for interpolation.')

		return c

	def interpolate(self, density):
		c = self._compute_coefs_interpolation(density)
		return np.dot(np.dot(self.M, self.U),c)

	def exp(self, density, type_approx = 'project', s_icdf=0., s_cdf=0., k=3):
		"""
		Exp map of v in LogSpace with respect to probability density mu
		and evaluated at point x.

		We first compute numerically the cdf of the exp map and then do finite differences to evaluate the derivative.

		The result is very sensitive to:
			- the implementation of the generalized inverse
			- the generalized inverse might sometimes not be monotonically increasing. This might cause spikes in the reconstruction. One attempt has been to pre-process the values of the the generalized inverse with isotonic regression.
			- the type of spline and smoothing factors.

		Notes:
			- Monotonic splines: scipy.interpolate.PchipInterpolator(x, y)
			- Isotonic regression

		As by-products, we give in the output the cdf of the map and the approx errors in W2, L2 and L1.

		Remark: Good values for s:
			- KdV: s_cdf = np.ceil(len(x)/5) ; s_icdf = 30./nr
			- ViscousBurgers : s_cdf = 0. ; s_icdf = 1./nr
		"""

		total_mass = density.m
		r  = density.r
		x  = density.x
		nr = len(r)
		nx = len(x)

		# icdf
		function_approx_log = getattr(self, type_approx)
		approx_log = function_approx_log(density)
		icdf_exp_map = self.barycenter.icdf+approx_log

		# cdf
		if density.icdf_smoothing():
			# Smoothing icdf with cubic spline to compute cdf.

			# Check if icdf_exp_map is monotonic
			d_icdf_exp_map = np.diff(icdf_exp_map)
			icdf_exp_map_smoothed = np.array(icdf_exp_map, copy= True)
			spline = scipy.interpolate.UnivariateSpline(r[1:-1], icdf_exp_map_smoothed[1:-1], s=s_icdf, k=3)
			icdf_exp_map_smoothed[1:-1] = spline(r[1:-1])
			cdf_x_exp_map  = ApproxFactory.g_inv(x, r, icdf_exp_map_smoothed)
			# Smoothing cdf with cubic spline
			cdf_x_exp_map_spline = scipy.interpolate.UnivariateSpline(x, cdf_x_exp_map, s=s_cdf, k=k)

			# Exp map
			exp_map = cdf_x_exp_map_spline.derivative()(x)
		else:
			icdf_exp_map_smoothed = icdf_exp_map
			dx = x[1] - x[0]
			cdf_x_exp_map  = ApproxFactory.g_inv(x, r, icdf_exp_map)
			# cdf_xh_exp_map = ApproxFactory.g_inv(x+dx, r, icdf_exp_map)
			# exp_map = (cdf_xh_exp_map - cdf_x_exp_map) / dx

			exp_map = ApproxFactory.fd_deriv(dx,cdf_x_exp_map)

			# dx = density.grid.dx
			# exp_map = np.zeros(len(cdf_x_exp_map))
			# exp_map[1:-1] = (np.diff(cdf_x_exp_map)[:-1] + np.diff(cdf_x_exp_map)[1:])/(2*dx)
			# exp_map[0] = (cdf_x_exp_map[1] - cdf_x_exp_map[0])/dx
			# exp_map[-1] = (cdf_x_exp_map[-1] - cdf_x_exp_map[-2])/dx

		# Error in W2, L2 and L1
		errW2 = np.sqrt(np.mean((icdf_exp_map-density.icdf)**2))
		xmin = density.xmin
		xmax = density.xmax
		errL2 = np.sqrt((xmax-xmin)*np.mean((exp_map-density.fun)**2))
		errL1 = (xmax-xmin)*np.mean(np.abs(exp_map-density.fun))
		errHminus1 = norm_Hminus1(exp_map-density.fun, density.x)

		return exp_map, cdf_x_exp_map, icdf_exp_map, icdf_exp_map_smoothed, errW2, errL2, errL1, errHminus1

class PCAspace():
	"""
		Classical PCA basis for the snapshots.
		An n dimensional space of L2([xmin,xmax]).
	"""
	def __init__(self, rbConstructionStrategy, n):
		"""
			dictionary: set of snapshots
			U: The j-th PCA vector is given by
				$$
					sum_j U[:,j].T snapshot[j]
				$$
			G: Grammian of the log snapshots.
			$$
				G_{i,j} = < log s[i], log s[j] >_{L2([0,1])}
			$$
			err: Error of PCA
			dim: Dimension of the reduced space
			grammian: Grammian of the reduced basis

		"""
		
		self.dictionary, self.U, self.G, self.err \
			= rbConstructionStrategy.generateBasis(n)
		self.dim      = n
		self.grammian = np.dot(np.dot(self.U.T, self.G), self.U)
		L = np.linalg.cholesky(self.grammian)
		self.L_inv = scipy.linalg.lapack.dtrtri(L)[0]
		self.M = self._computeM()

	def _computeM(self):
		nQuad = len(self.dictionary[0].x)
		N = len(self.dictionary)
		M = np.zeros((nQuad, N)) # Matrix of snapshots evaluated at quadrature points
		for j, snapshot in enumerate(self.dictionary):
			M[:, j] = snapshot.fun
		return M

	def project(self, density):
		# rhs: ( <density, basis[i]>_{L2([0,1])} )_{i=1,..,dim}
		xQuad = density.x
		nQuad = len(xQuad)
		rhs= np.dot(np.dot(self.M, self.U).T, density.fun)/nQuad

		# We solve system G x = rhs
		# Option 1: seems more accurate but slower
		# c = np.linalg.solve(self.grammian, rhs)
		# Option 2: seems a bit less accurate but faster
		c = np.dot(self.L_inv.T, np.dot(self.L_inv, rhs))

		# We evaluate projection at x
		density_proj_xQuad = np.dot(np.dot(self.M, self.U),c)

		# Computation of the error.
		errL2 = np.sqrt(np.mean((density_proj_xQuad-density.fun)**2) )
		errHminus1 = norm_Hminus1(density_proj_xQuad-density.fun, density.x)

		return density_proj_xQuad, errL2, errHminus1

class Barycenterspace():
	"""
		Approximation with barycenter.
	"""
	def __init__(self, rbConstructionStrategy, n):
		self.dictionary, self.U, self.indices, self.err, self.errav \
			= rbConstructionStrategy.generateBasis(n)
		self.dim      = n
		self.Qn = 2*matrix(np.dot(self.U,self.U.T))  

		self.Gn = matrix(-1.0*np.identity(n))
		self.hn = matrix(np.zeros(n))

		self.An = matrix(np.ones((1,n)))
		self.bn = matrix(np.ones(1))

		self.P, self.C = self._prepare_local_interpolation()

	def project(self, density):
		# TODO Virginie:
		# Given a density (e.g. a snapshot), compute the approximation with
		# the barycenter
		r  = density.r
		x  = density.x
		nr = len(r)
		nx = len(x)
		dx = x[1] - x[0]


		M = density.icdf
		pn = matrix(-2*np.dot(self.U,M.T))

		solvers.options['show_progress'] = False
		solvers.options['maxiters'] = 10000
		solvers.options['abstol'] = 1e-14
		solvers.options['reltol'] = 1e-14
		solvers.options['feastol'] = 1e-14
		sol= solvers.qp(self.Qn, pn, self.Gn, self.hn, self.An, self.bn)
		coeffs = np.array(sol['x'])

		icdf_recons = np.dot(coeffs[:,0].T, self.U)

		## TODO: test monotonicity
		cdf_x  = ApproxFactory.g_inv(x, r, icdf_recons)

		exp_recons = ApproxFactory.fd_deriv(dx,cdf_x)

		errW2 = np.sqrt(np.mean((icdf_recons-density.icdf)**2))
		errL2 = np.sqrt(np.mean((exp_recons-density.fun)**2) )
		errHminus1 = norm_Hminus1(exp_recons-density.fun, density.x)
		
		return exp_recons, cdf_x, icdf_recons, errL2, errW2, errHminus1, coeffs

	def _prepare_local_interpolation(self):
		n_snapshots = len(self.dictionary)
		n_params    = len(self.dictionary[0].param)

		P = np.zeros((n_snapshots, n_params)) # Matrix of snapshot parameters
		C = np.zeros((n_snapshots, self.dim)) # Matrix of coefs projection

		for i, snapshot in enumerate(self.dictionary):
			P[i, :] = snapshot.param
			C[i, :]  = self._compute_coefs_projection(snapshot)

		return P, C

	def _compute_coefs_projection(self, density):

		exp_recons, cdf_x, icdf_recons, errL2, errW2, errHminus1, coeffs = self.project(density)

		return coeffs[:,0]

	def _build_barycentric_coord(self,P, param):

		mp = P.shape[0]

		param0 = np.array(param)

		Q = 2*matrix(np.dot(P,P.T))  

		G = matrix(-1.0*np.identity(mp))
		h = matrix(np.zeros(mp))

		A = matrix(np.ones((1,mp)))
		b = matrix(np.ones(1))

		p = matrix(-2*np.dot(P,param0))

		solvers.options['show_progress'] = False
		solvers.options['maxiters'] = 10000
		solvers.options['abstol'] = 1e-14
		solvers.options['reltol'] = 1e-14
		solvers.options['feastol'] = 1e-14
		sol= solvers.qp(Q, p, G, h, A, b)
		coeffs = np.array(sol['x'])

		return coeffs

	def interpolate(self, density):

		param = density.param
		# Find k nearest-neighbors
		# We first find the value of k:
		k = 2
		if len(param) == 3:
			k = 8
		elif len(param) == 2:
			k = 4
		elif len(param) == 1:
			k = 2
		else:
			raise Exception('Number of parameters not supported for interpolation.')


		dE = (self.P-param)/np.max(self.P, axis=0)
		E = np.linalg.norm(dE, axis=1)
		idx = np.argpartition(E, k)[:k]
	
		P_neighbors = np.array(self.P[idx, :])
		C_neighbors = np.array(self.C[idx, :])

		coeffs = self._build_barycentric_coord(P_neighbors, param)

		newC = np.dot(coeffs.T,C_neighbors)

		temp = np.dot(newC, self.U)
		icdf_recons = (temp.T)[:,0]

		r  = density.r
		x  = density.x
		nr = len(r)
		nx = len(x)
		dx = x[1] - x[0]
		# TODO: test monotonicity
		cdf_x  = ApproxFactory.g_inv(x, r, icdf_recons)

		exp_recons = ApproxFactory.fd_deriv(dx,cdf_x)

		errW2 = np.sqrt(np.mean((icdf_recons-density.icdf)**2))
		errL2 = np.sqrt(np.mean((exp_recons-density.fun)**2) )
		errHminus1 = norm_Hminus1(exp_recons-density.fun, density.x)
		
		return exp_recons, cdf_x, icdf_recons, errL2, errW2, errHminus1, newC

	def reconstruction(self, density, type_approx = 'project'):

		if (type_approx == 'project'):
			return self.project(density)

		elif (type_approx == 'interpolate'):
			return self.interpolate(density)




	
