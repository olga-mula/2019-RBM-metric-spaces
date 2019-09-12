import numpy as np
import random
from cvxopt import solvers, matrix

from space import W2space

class RBconstruction():

	def __init__(self, dictionary):
		self.dictionary = dictionary

class SpaceConstructionRandom(RBconstruction):
	"""
		We take randomly snapshot solutions.
	"""

	def __init__(self, dictionary):
		super().__init__(dictionary)

	def generateBasis(self, n):
		# Number of snapshots
		N = len(self.dictionary)
		assert N > 0

		norm_type = 'L2'

		# We use the Fisher-Yates algorithm, which takes O(N) operations
		a = np.arange(N)
		for i in range(N-1, N-n, -1):
			j = int(random.random()*i)
			a[i], a[j] = a[j], a[i]
		basis = [self.dictionary[i] for i in a[N-n: N]]

		return norm_type, basis

class SpaceConstructionLogPCA(RBconstruction):
	"""
		We do PCA in the log map.
	"""

	def __init__(self, dictionary):
		super().__init__(dictionary)

		# Compute barycenter
		self.barycenter = W2space().barycenter(self.dictionary)

		# Remove the barycenter from dictionary
		# super().__init__([snap for i, snap in enumerate(self.dictionary) if i!=idx])

		N  = len(self.dictionary)
		nx = len(self.dictionary[0].fun)

		# Correlation matrix C (C is also the Gramian of the log snapshots)
		total_mass = self.barycenter.m
		# Matrix of log snapshots evaluated at discrete points
		M = np.zeros((N,nx))
		for i, snapshot in enumerate(self.dictionary):
			# print(len(M[i,:]), len(snapshot.log(self.barycenter)))
			M[i,:] = snapshot.log(self.barycenter)
		self.C = np.dot(M, M.T)/nx

		# SVD
		self.U, self.S, self.V = np.linalg.svd(self.C/N, full_matrices=True)

	def generateBasis(self, n):
		err = np.sqrt(np.sum(self.S[n:]))
		return self.dictionary, self.barycenter, self.U[:, 0:n], self.C, err

class SpaceConstructionPCA(RBconstruction):
	"""
		We do classical PCA in the snapshots.
	"""

	def __init__(self, dictionary):

		super().__init__(dictionary)
		N = len(self.dictionary)
		nx = len(self.dictionary[0].fun)

		# Correlation matrix C (C is also the Gramian of the snapshots)
		M = np.zeros((N,nx)) # Matrix of log snapshots at discrete points
		for i, snapshot in enumerate(self.dictionary):
			M[i,:] = snapshot.fun
		self.C = np.dot(M, M.T)/nx

		# SVD
		self.U, self.S, self.V = np.linalg.svd(self.C/N, full_matrices=True)

	def generateBasis(self, n):
		err = np.sqrt(np.sum(self.S[n:]))
		return self.dictionary, self.U[:, 0:n], self.C, err

class SpaceConstructionGreedyBarycenter(RBconstruction):
	"""
		We do a greedy algorithm to build a barycenter approximation.
	"""
	def __init__(self, dictionary, Nmax=21):
		super().__init__(dictionary)

		print("Running Greedy algorithm")
		self.U, self.S, self.Sav, self.indices = self.run_greedy(Nmax=Nmax)

	def run_greedy(self, Nmax):
		"""
			Nmax: Dimension to run greedy
		"""

		N = len(self.dictionary)
		nr = len(self.dictionary[0].r)

		# Initializing options for solvers library of cvxopt
		solvers.options['show_progress'] = False
		solvers.options['maxiters'] = 10000
		solvers.options['abstol'] = 1e-14
		solvers.options['reltol'] = 1e-14
		solvers.options['feastol'] = 1e-14

		# Matrix of icdf at discrete points
		M = np.zeros((N,nr))
		for i, snapshot in enumerate(self.dictionary):
			M[i,:] = snapshot.icdf

		Ugreed = np.zeros((Nmax,nr))

		Sgreed = np.zeros(Nmax-1)
		Sgreedav = np.zeros(Nmax-1)
		indices = list()
		
		# Greedy: Step k=1
		# We select the first two basis functions
		i0 = 0
		j0 = 0
		dist = 0
		for i in range(0,N):
			for j in range(i+1,N):
				diffij = M[i,:] - M[j,:]
				distij = np.sqrt(1.0/nr*np.dot(diffij.T, diffij))
				if (distij > dist):
					dist = distij
					Ugreed[0,:] = M[i,:]
					Ugreed[1,:] = M[j,:]
					i0 = i
					j0 = j

		Sgreed[0] = dist
		Sgreedav[0] = dist
		indices.append(i0)
		indices.append(j0)
		print('k = 1 ; distk = ' + str(dist))

		# Greedy: step k > 1
		for k in range(2,Nmax):
			dist = 0;

			Ukm= Ugreed[0:k,:]
			Qk = 2*matrix(np.dot(Ukm,Ukm.T)) 

			Gk = matrix(-1.0*np.identity(k))
			hk = matrix(np.zeros(k))

			Ak = matrix(np.ones((1,k)));
			bk = matrix(np.ones(1))

			i0 = 0
			errav = 0
			for i in range(0,N):

				Mi = M[i,:]
				pk = matrix(-2*np.dot(Ukm,Mi.T))

				sol= solvers.qp(Qk, pk, Gk, hk, Ak, bk)
				coeffs = np.array(sol['x'])
				diffi = Mi - np.dot(coeffs[:,0].T, Ukm)
				dist_new = np.sqrt(1.0/nr*np.dot(diffi.T, diffi))
				errav = errav + 1.0/N*dist_new*dist_new
				if (dist_new > dist):
					dist = dist_new
					Ugreed[k,:] = Mi
					i0 = i
		
			Sgreed[k-1] = dist
			Sgreedav[k-1] = np.sqrt(errav)
			indices.append(i0)
			print('k = '+str(k)+' ; distk = ' + str(dist))

		return Ugreed, Sgreed, Sgreedav, indices

	def generateBasis(self, n):

		if (n<=30):
			err = self.S[n-1]
			errav = self.Sav[n-1]
			return self.dictionary, self.U[0:n, :], self.indices[0:n], err, errav
