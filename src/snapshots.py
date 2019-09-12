import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('contributions/')

from approxFactory import ApproxFactory
from viscousburgers1D import SimulationViscousBurgers, Grid1d
from CH import SimulationCamassaHolm

class ProbaMeasure():
	"""
		Probability measure in [xmin, xmax]
	"""
	def __init__(self, fun=None, cdf=None, icdf=None, m=1.):
		self.fun  = fun
		self.cdf  = cdf
		self.icdf = icdf
		self.m    = m

	def eval(self, x):
		pass

	def cdf(self, x):
		"""
		Explicit formula of cdf
		"""
		pass

	def icdf(self, r):
		"""
		Explicit formula of icdf
		"""
		pass


class Burgers(ProbaMeasure):
	"""
		Snapshot: Solution at time t of the parametrized
		1d Burger's equation for x in [0,5] with periodic BC
			u_t + 0.5*(u^2)_x = 0
			u(0,x) = y if ...
					 y/2 if ...
					 0 if .....
			u(t,xmin) = u(t,xmax)  (periodic BC)

	"""
	def __init__(self, param, nx=5000, ng=2):
		# Domain
		(self.xmin, self.xmax) = self.xRange()
		# Params snapshot
		self.param = param
		self.t     = param[0]
		self.y     = param[1]
		# Mass
		self.m = 1.
		# Spatial grid
		self.grid = Grid1d(nx, ng=ng, vars=["u"], xmin=self.xmin, xmax=self.xmax)
		self.x = self.grid.x
		# Discrete solution, cdf, icdf and mass
		self.fun  = self.eval(self.x)
		self.cdf  = self.cdf(self.x)
		self.r = np.linspace(0., self.m, num=nx+2*ng)
		self.icdf = self.icdf(self.r)

	@staticmethod
	def problemType():
		return 'Burgers'

	@staticmethod
	def xRange():
		return (-1., 4.)

	@staticmethod
	def paramRange():
		tmin = 0.; tmax = 5.
		ymin = 0.5; ymax = 3.0
		return [(tmin, tmax), (ymin, ymax)]

	@staticmethod
	def nSamplesPerParameter():
		return [5, 5]

	@staticmethod
	def typeSamplingPerParameter():
		return ['linear', 'linear']

	@staticmethod
	def icdf_smoothing():
		return False

	@staticmethod
	def get_info():
		paramRange = ",".join("(%s,%s)" % tup for tup in Burgers.paramRange())
		paramRange = '[' + paramRange + ']'

		total_samples = str(np.prod(Burgers.nSamplesPerParameter()))

		nSamplesPerParameter = ",".join(str(n) for n in Burgers.nSamplesPerParameter())
		nSamplesPerParameter = '[' + nSamplesPerParameter + ']'

		typeSamplingPerParameter = ",".join(s for s in Burgers.typeSamplingPerParameter())
		typeSamplingPerParameter = '[' + typeSamplingPerParameter + ']'

		s = 'Problem type: ' + Burgers.problemType() + '\n'
		s = s + 'Parameter domain: (t, y) in ' + paramRange + '\n'
		s = s + 'Offline phase with ' + total_samples + ' snapshots.' +'\n'
		s = s + 'Cartesian product grid: ' + nSamplesPerParameter + ' snapshots per parameter. Distribution: ' + typeSamplingPerParameter + '\n'
		s = s + 'Smoothing for computation of exp: ' + str(Burgers.icdf_smoothing()) + '\n'
		return s

	def eval(self, x):
		"""
		Explicit formula of u(t,x,y)
		"""
		val = None

		if (self.t==0):
			m1 = (x>=0.) & (x<=1./self.y)
			val = self.y * m1

		elif self.t <= 2./self.y**2:
			m1 = (x>=-1) & (x<0)
			m2 = (x>=0) & (x<self.t * self.y)
			m3 = (x>=self.t * self.y) & (x<= 1./self.y + self.t * self.y/2.)
			m4 = (x>1./self.y + self.t * self.y/2.) & (x<=4)
			val = x/self.t*m2 + self.y*m3

		else:
			m1 = (x>=-1) & (x<0)
			m2 = (x>=0) & (x<=np.sqrt(2.*self.t))
			m3 = (x>np.sqrt(2.*self.t)) & (x<=4)
			val = x/self.t *m2

		return val


	def cdf(self, x):
		"""
		Explicit formula of cdf.
		$$
			cdf : Omega to [0,1]
		$$
		"""
		
		val = None

		if (self.t==0):
			m1 = (x>=0.) & (x<=1./self.y)
			val = self.y * x * m1

		elif self.t <= 2./self.y**2:
			m1 = (x>=-1) & (x<0)
			m2 = (x>=0) & (x<self.t * self.y)
			m3 = (x>=self.t * self.y) & (x<= 1./self.y + self.t * self.y/2.)
			m4 = (x>1./self.y + self.t * self.y/2.) & (x<=4)
			val = 0.*m1 + x**2/(2.*self.t)*m2 + (self.y*x-self.t*self.y**2/2)*m3 + m4

		else:
			m1 = (x>=-1) & (x<0)
			m2 = (x>=0) & (x<=np.sqrt(2.*self.t))
			m3 = (x>np.sqrt(2.*self.t)) & (x<=4)
			val = x**2/(2*self.t)*m2 + m3

		return val

	def icdf(self, r):
		"""
		Explicit formula of icdf
		$$
			icdf : [0,1] to Omega
		$$
		"""
		
		val = None

		if (self.t==0):
			m1 = (r==0.)
			m2 = (r>0.) & (r <=1.)
			val = -1. * m1 + r / self.y * m2

		elif self.t <= 2./self.y**2:
			m1 = (r==0.)
			m2 = (r>=0.) & (r<=self.y**2 * self.t /2.)
			m3 = (r>self.y**2 * self.t /2.) & (r <=1)
			val = -1. * m1 + np.sqrt(2. * self.t * r) * m2 + 1./self.y * (r + self.y**2 * self.t /2.) *m3

		else:
			m1 = (r==0.)
			m2 = (r>=0.) & (r <=1.)
			val = -1. * m1 + np.sqrt(2. * self.t * r) * m2

		return val

	def log(self, mu):
		"""
			Log map with respect to probability density mu
		"""
		return self.icdf - mu.icdf


	def exp(self, x, mu):
		"""
			Exp map with respect to probability density mu
		"""
		pass

	def plot(self, save=False, num=200, opt = 'density'):
		if opt == 'density':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.eval(x))
		elif opt == 'cdf':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.cdf(x))
		elif opt == 'icdf':
			r = np.linspace(0., self.m, num=num, endpoint=True)
			plt.plot(r, self.icdf(r))
		elif opt == 'test-cdf-icdf':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.icdf(self.cdf(x)))
		plt.show()
		return


class KdV(ProbaMeasure):
	"""
		Snapshot: Solution at time t of the parametrized
		1d KdV's equation

	"""
	def __init__(self, param, nx=5000, ng=0):
		# Domain
		(self.xmin, self.xmax) = self.xRange()
		# Params snapshot
		self.param = param
		self.t     = param[0]
		# self.c     = np.array([param[1], 1.5], copy=True)
		self.c     = np.array([2., 1.5], copy=True)
		#  self.k     = np.array([30.-param[2], param[2]], copy=True)
		self.k     = np.array([30.-param[1], param[1]], copy=True)
		# Important variables derived from previous
		self.m    = 4.*(self.k[0]+self.k[1])
		# Spatial grid
		self.grid = Grid1d(nx, ng=ng, vars=["u"], xmin=self.xmin, xmax=self.xmax)
		self.x = self.grid.x
		# Discrete solution, cdf, icdf and mass
		self.fun  = self.eval(self.x)
		self.cdf  = self.cdf(self.x)
		self.r = np.linspace(0., self.m, num=nx+2*ng)
		self.icdf = self.icdf(self.r)

	@staticmethod
	def problemType():
		return 'KdV'

	@staticmethod
	def xRange():
		return (-0.5, 2.0)

	@staticmethod
	def paramRange():
		tmin = 0.; tmax = 2.5 * 1.e-3
		# c1min = 2.; c1max = 2.
		k2min = 16;   k2max = 22
		# return [(tmin, tmax), (c1min, c1max), (k2min, k2max)]
		return [(tmin, tmax), (k2min, k2max)]

	@staticmethod
	def nSamplesPerParameter():
		return [100, 80]

	@staticmethod
	def typeSamplingPerParameter():
		return ['linear', 'linear']

	@staticmethod
	def icdf_smoothing():
		return True

	@staticmethod
	def get_info():
		paramRange = ",".join("(%s,%s)" % tup for tup in KdV.paramRange())
		paramRange = '[' + paramRange + ']'

		total_samples = str(np.prod(KdV.nSamplesPerParameter()))

		nSamplesPerParameter = ",".join(str(n) for n in KdV.nSamplesPerParameter())
		nSamplesPerParameter = '[' + nSamplesPerParameter + ']'

		typeSamplingPerParameter = ",".join(s for s in KdV.typeSamplingPerParameter())
		typeSamplingPerParameter = '[' + typeSamplingPerParameter + ']'

		s = 'Problem type: ' + KdV.problemType() + '\n'
		s = s + 'Parameter domain: (t, k2) in ' + paramRange + '\n'
		s = s + 'Offline phase with ' + total_samples + ' snapshots.' +'\n'
		s = s + 'Cartesian product grid: ' + nSamplesPerParameter + ' snapshots per parameter. Distribution: ' + typeSamplingPerParameter + '\n'
		s = s + 'Smoothing for computation of exp: ' + str(KdV.icdf_smoothing()) + '\n'
		return s

	def A(self, x):
		A = np.zeros((2,2))
		t = self.t
		for m in [0, 1]:
			for n in [0, 1]:
				cm = self.c[m]
				cn = self.c[n]
				km = self.k[m]
				kn = self.k[n]
				A[m,n] = cm * cn / (km + kn) * np.exp((km+kn)*x - (km**3 + kn**3)*t)
		return A

	def dA(self, x):
		A = np.zeros((2,2))
		t = self.t
		for m in [0, 1]:
			for n in [0, 1]:
				cm = self.c[m]
				cn = self.c[n]
				km = self.k[m]
				kn = self.k[n]
				A[m,n] = cm * cn * np.exp((km+kn)*x - (km**3 + kn**3)*t)
		return A

	def d2A(self, x):
		A = np.zeros((2,2))
		t = self.t
		for m in [0, 1]:
			for n in [0, 1]:
				cm = self.c[m]
				cn = self.c[n]
				km = self.k[m]
				kn = self.k[n]
				A[m,n] = cm * cn * (km+kn) * np.exp((km+kn)*x - (km**3 + kn**3)*t)
		return A

	def eval(self, x):
		"""
		Explicit formula of u(t,x,y)
		"""
		if np.isscalar(x):
			return self._eval_scalar(x)
		else:
			val = list()
			for xi in x:
				val.append(self._eval_scalar(xi))
			return np.asarray(val)

	def _eval_scalar(self, x):
		A   = self.A(x)
		dA  = self.dA(x)
		d2A = self.d2A(x)

		f   = (1.+A[0,0]) * (1.+A[1,1]) - A[0,1]**2

		df  = dA[0,0]*(1.+A[1,1]) + (1.+A[0,0])*dA[1,1] - 2.*A[0,1]*dA[0,1]

		d2f = d2A[0,0]*(1.+A[1,1]) + 2.*dA[0,0]*dA[1,1] + (1.+A[0,0])*d2A[1,1] - 2.*dA[0,1]**2 - 2.*A[0,1]*d2A[0,1]

		return 2.*(d2f/f - df**2/f**2)

	def cdf(self, x):
		"""
		Explicit formula of cdf.
		$$
			cdf : Omega to [0,1]
		$$
		"""
		if np.isscalar(x):
			return self._cdf_scalar(x)
		else:
			val = list()
			for xi in x:
				val.append(self._cdf_scalar(xi))
			return np.asarray(val)

	def _cdf_scalar(self, x):
		A   = self.A(x)
		dA  = self.dA(x)

		f   = (1.+A[0,0]) * (1.+A[1,1]) - A[0,1]**2
		df  = dA[0,0]*(1.+A[1,1]) + (1.+A[0,0])*dA[1,1] - 2.*A[0,1]*dA[0,1]

		return 2. * df / f

	def icdf(self, r):
		"""
		Explicit formula of icdf
		$$
			icdf : r in [0, self.masse] to Omega
		$$
		"""
		return ApproxFactory.g_inv(self.r, self.x, self.cdf)

	def log(self, mu):
		"""
			Log map with respect to probability density mu
		"""
		return self.icdf - mu.icdf

	def exp(self, x, mu):
		"""
			Exp map with respect to probability density mu
		"""
		pass

	def plot(self, save=False, num=200, opt = 'density'):
		if opt == 'density':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.eval(x))
		elif opt == 'cdf':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.cdf(x))
		elif opt == 'icdf':
			r = np.linspace(0., self.m, num=num, endpoint=True)
			plt.plot(r, self.icdf(r))
		elif opt == 'test-cdf-icdf':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.icdf(self.cdf(x)))
		plt.show()
		return

class ViscousBurgers(ProbaMeasure):
	"""
		Snapshot: Solution at time t of the parametrized
		1d KdV's equation

	"""
	def __init__(self, param, nx=5000, ng=2):
		# Domain
		(self.xmin, self.xmax) = self.xRange()
		# Params snapshot
		self.param = param
		self.t     = param[0]
		self.y     = param[1]
		self.nu    = param[2]
		# Spatial grid
		self.grid = Grid1d(nx, ng=ng, vars=["u"], xmin=self.xmin, xmax=self.xmax)
		self.x = self.grid.x
		# Discrete solution, cdf, icdf and mass
		self.fun  = self._solve(self.grid)
		self.m    = 1. # np.sum(self.fun) * self.grid.dx
		self.cdf  = self._cdf()
		self.r = np.linspace(0., self.m, num=nx+2*ng)
		self.icdf = self._icdf(self.r)

	@staticmethod
	def problemType():
		return 'ViscousBurgers'

	@staticmethod
	def xRange():
		return (-3., 5.)

	@staticmethod
	def paramRange():
		tmin = 0.; tmax = 5.
		ymin = 0.5; ymax = 3.0
		numin = 5.e-5; numax = 0.1
		return [(tmin, tmax), (ymin, ymax), (numin, numax)]

	@staticmethod
	def nSamplesPerParameter():
		return [20, 20, 20]

	@staticmethod
	def typeSamplingPerParameter():
		return ['linear', 'linear', 'log']

	@staticmethod
	def icdf_smoothing():
		return True

	@staticmethod
	def get_info():
		paramRange = ",".join("(%s,%s)" % tup for tup in ViscousBurgers.paramRange())
		paramRange = '[' + paramRange + ']'

		total_samples = str(np.prod(ViscousBurgers.nSamplesPerParameter()))

		nSamplesPerParameter = ",".join(str(n) for n in ViscousBurgers.nSamplesPerParameter())
		nSamplesPerParameter = '[' + nSamplesPerParameter + ']'

		typeSamplingPerParameter = ",".join(s for s in ViscousBurgers.typeSamplingPerParameter())
		typeSamplingPerParameter = '[' + typeSamplingPerParameter + ']'

		s = 'Problem type: ' + ViscousBurgers.problemType() + '\n'
		s = s + 'Parameter domain: (t, y, nu) in ' + paramRange + '\n'
		s = s + 'Offline phase with ' + total_samples + ' snapshots.' +'\n'
		s = s + 'Cartesian product grid: ' + nSamplesPerParameter + ' snapshots per parameter. Distribution: ' + typeSamplingPerParameter + '\n'
		s = s + 'Smoothing for computation of exp: ' + str(ViscousBurgers.icdf_smoothing()) + '\n'
		return s

	def _solve(self, grid):
		cfl = 0.1
		s = SimulationViscousBurgers(grid)
		s.init_cond(self.y)
		s.evolve(self.nu, cfl, self.t, dovis=0)
		return s.grid.data["u"]

	def eval(self, x):
		"""
		Evaluation of u(t,x,q,a)
		"""
		pass

	def _eval_scalar(self, x):
		pass


	def _cdf(self):
		"""
		cdf evaluated numerically on self.grid
		$$
			cdf : Omega to [0,1]
		$$
		"""
		return np.cumsum(self.fun) * self.grid.dx

	def _icdf(self, r):
		"""
		Explicit formula of icdf
		$$
			icdf : r in [0, self.masse] to Omega
		$$
		"""
		return ApproxFactory.g_inv(self.r, self.grid.x, self.cdf)

	def log(self, mu):
		"""
			Log map with respect to probability density mu
		"""
		return self.icdf - mu.icdf

	def exp(self, x, mu):
		"""
			Exp map with respect to probability density mu
		"""
		pass

	def plot(self, save=False, num=200, opt = 'density'):
		if opt == 'density':
			plt.plot(self.grid.x, self.fun)
			plt.xlim(self.xmin, self.xmax)
		elif opt == 'cdf':
			plt.plot(self.grid.x, self.cdf)
			plt.xlim(self.xmin, self.xmax)
		elif opt == 'icdf':
			plt.plot(self.r, self.icdf)
		elif opt == 'test-cdf-icdf':
			print('Plot option test-cdf-icdf not implemented')
		plt.legend(frameon=False)
		plt.tight_layout()
		plt.show()
		return


class CamassaHolm(ProbaMeasure):
	"""
		Snapshot: Solution at time t of the parametrized
		1d Camassa-Holm's equation

	"""
	def __init__(self, param, nx=5000, ng=2):
		# Domain
		(self.xmin, self.xmax) = self.xRange()
		# Params snapshot
		self.param = param
		self.t = param[0]
		self.q = param[1] # Position of one peakon
		self.p = param[2] # Mass of one peakon
		# Spatial grid
		self.grid = Grid1d(nx, ng=ng, vars=["u"], xmin=self.xmin, xmax=self.xmax)
		self.x = self.grid.x
		# Discrete solution, cdf, icdf and mass
		self.alpha = 1.
		self.m    = 1.
		self.fun  = self._solve(self.grid)
		self.cdf  = self._cdf()
		self.r = np.linspace(0., self.m, num=nx+2*ng)
		self.icdf = self._icdf(self.r)

	@staticmethod
	def problemType():
		return 'CamassaHolm'

	@staticmethod
	def xRange():
		return (-10., 20.)

	@staticmethod
	def paramRange():
		tmin = 0.; tmax = 40.
		qmin = -2.; qmax = 2.
		pmin = 0.2; pmax = 0.2
		return [(tmin, tmax), (qmin, qmax), (pmin, pmax)]

	@staticmethod
	def nSamplesPerParameter():
		return [100, 100, 1]

	@staticmethod
	def typeSamplingPerParameter():
		return ['linear', 'linear', 'linear']

	@staticmethod
	def icdf_smoothing():
		return False

	@staticmethod
	def get_info():
		paramRange = ",".join("(%s,%s)" % tup for tup in CamassaHolm.paramRange())
		paramRange = '[' + paramRange + ']'

		total_samples = str(np.prod(CamassaHolm.nSamplesPerParameter()))

		nSamplesPerParameter = ",".join(str(n) for n in CamassaHolm.nSamplesPerParameter())
		nSamplesPerParameter = '[' + nSamplesPerParameter + ']'

		typeSamplingPerParameter = ",".join(s for s in CamassaHolm.typeSamplingPerParameter())
		typeSamplingPerParameter = '[' + typeSamplingPerParameter + ']'

		s = 'Problem type: ' + CamassaHolm.problemType() + '\n'
		s = s + 'Parameter domain: (t, q, a) in ' + paramRange + '\n'
		s = s + 'Offline phase with ' + total_samples + ' snapshots.' +'\n'
		s = s + 'Cartesian product grid: ' + nSamplesPerParameter + ' snapshots per parameter. Distribution: ' + typeSamplingPerParameter + '\n'
		s = s + 'Smoothing for computation of exp: ' + str(CamassaHolm.icdf_smoothing()) + '\n'
		return s

	def _solve(self, grid):
		deltat = 0.1
		s = SimulationCamassaHolm(grid)
		s.init_cond(self.q,self.p, self.alpha)
		return s.evolve(deltat, self.t)

	def eval(self, x):
		"""
		Evaluation of u(t,x,y)
		"""
		pass

	def _eval_scalar(self, x):
		pass


	def _cdf(self):
		"""
		cdf evaluated numerically on self.grid
		$$
			cdf : Omega to [0,1]
		$$
		"""
		return np.cumsum(self.fun) * self.grid.dx

	def _icdf(self, r):
		"""
		Explicit formula of icdf
		$$
			icdf : r in [0, self.masse] to Omega
		$$
		"""
		return ApproxFactory.g_inv(self.r, self.grid.x, self.cdf)

	def log(self, mu):
		"""
			Log map with respect to probability density mu
		"""
		return self.icdf - mu.icdf

	def exp(self, x, mu):
		"""
			Exp map with respect to probability density mu
		"""
		pass

	def plot(self, save=False, num=200, opt = 'density'):
		if opt == 'density':
			plt.plot(self.grid.x, self.fun)
			plt.xlim(self.xmin, self.xmax)
		elif opt == 'cdf':
			plt.plot(self.grid.x, self.cdf)
			plt.xlim(self.xmin, self.xmax)
		elif opt == 'icdf':
			plt.plot(self.r, self.icdf)
		elif opt == 'test-cdf-icdf':
			print('Plot option test-cdf-icdf not implemented')
		plt.legend(frameon=False)
		plt.tight_layout()
		plt.show()
		return



class NavierStokes(ProbaMeasure):
	"""
		Snapshot: Solution at time t of the parametrized
		Navier Stokes equation

	"""
	def __init__(self, param, nx=500, ng=0):
		# Domain
		(self.xmin, self.xmax) = self.xRange()
		# Params snapshot
		self.param = param
		self.t     = param[0]
		self.y0    = param[1] # velocity amplitude
		self.y1    = param[2] # pulsation plate
		self.y2    = param[3] # fluid's viscosity
		# Important variables derived from previous
		self.kappa = np.sqrt( self.y1 / (2.*self.y2) )
		self.m     = 1.
		# Spatial grid
		self.grid = Grid1d(nx, ng=ng, vars=["u"], xmin=self.xmin, xmax=self.xmax)
		self.x = self.grid.x
		# Discrete solution, cdf, icdf and mass
		self.fun  = self.eval(self.x)
		self.cdf  = self.cdf(self.x)
		self.r = np.linspace(0., self.m, num=nx+2*ng)
		self.icdf = self.icdf(self.r)

	@staticmethod
	def problemType():
		return 'NavierStokes'

	@staticmethod
	def xRange():
		return (0., 10.0)

	@staticmethod
	def paramRange():
		tmin = 0.; tmax = 2.5 * 1.e-3
		y0min = 1.0; y0max = 5.
		y1min = 1.0; y1max = 5.
		y2min = 1.0; y2max = 5.

		return [(tmin, tmax), (y0min, y0max), (y1min, y1max), (y2min, y2max)]

	@staticmethod
	def nSamplesPerParameter():
		return [20, 20, 20, 20]

	@staticmethod
	def typeSamplingPerParameter():
		return ['linear', 'linear', 'linear', 'linear']

	@staticmethod
	def icdf_smoothing():
		return True

	@staticmethod
	def get_info():
		paramRange = ",".join("(%s,%s)" % tup for tup in KdV.paramRange())
		paramRange = '[' + paramRange + ']'

		total_samples = str(np.prod(KdV.nSamplesPerParameter()))

		nSamplesPerParameter = ",".join(str(n) for n in KdV.nSamplesPerParameter())
		nSamplesPerParameter = '[' + nSamplesPerParameter + ']'

		typeSamplingPerParameter = ",".join(s for s in KdV.typeSamplingPerParameter())
		typeSamplingPerParameter = '[' + typeSamplingPerParameter + ']'

		s = 'Problem type: ' + KdV.problemType() + '\n'
		s = s + 'Parameter domain: (t, y0, y1, y2) in ' + paramRange + '\n'
		s = s + 'Offline phase with ' + total_samples + ' snapshots.' +'\n'
		s = s + 'Cartesian product grid: ' + nSamplesPerParameter + ' snapshots per parameter. Distribution: ' + typeSamplingPerParameter + '\n'
		s = s + 'Smoothing for computation of exp: ' + str(KdV.icdf_smoothing()) + '\n'
		return s

	def eval(self, x):
		"""
		Explicit formula of u(t,x,y)
		"""
		if np.isscalar(x):
			return self._eval_scalar(x)
		else:
			val = list()
			for xi in x:
				val.append(self._eval_scalar(xi))
			return np.asarray(val)

	def _eval_scalar(self, x):
		return self.y0 * np.exp(-self.kappa*x) * np.cos(self.y1*self.t-self.kappa*x)

	def cdf(self, x):
		"""
		Explicit formula of cdf.
		$$
			cdf : Omega to [0,1]
		$$
		"""
		if np.isscalar(x):
			return self._cdf_scalar(x)
		else:
			val = list()
			for xi in x:
				val.append(self._cdf_scalar(xi))
			return np.asarray(val)

	def _cdf_scalar(self, x):
		A   = self.A(x)
		dA  = self.dA(x)

		f   = (1.+A[0,0]) * (1.+A[1,1]) - A[0,1]**2
		df  = dA[0,0]*(1.+A[1,1]) + (1.+A[0,0])*dA[1,1] - 2.*A[0,1]*dA[0,1]

		return 2. * df / f

	def icdf(self, r):
		"""
		Explicit formula of icdf
		$$
			icdf : r in [0, self.masse] to Omega
		$$
		"""
		return ApproxFactory.g_inv(self.r, self.x, self.cdf)

	def log(self, mu):
		"""
			Log map with respect to probability density mu
		"""
		return self.icdf - mu.icdf

	def exp(self, x, mu):
		"""
			Exp map with respect to probability density mu
		"""
		pass

	def plot(self, save=False, num=200, opt = 'density'):
		if opt == 'density':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.eval(x))
		elif opt == 'cdf':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.cdf(x))
		elif opt == 'icdf':
			r = np.linspace(0., self.m, num=num, endpoint=True)
			plt.plot(r, self.icdf(r))
		elif opt == 'test-cdf-icdf':
			x = np.linspace(self.xmin, self.xmax, num=num, endpoint=True)
			plt.plot(x, self.icdf(self.cdf(x)))
		plt.show()
		return

