import numpy as np
import itertools

class SamplingStrategy():

	def __init__(self, parameterDomain):
		self.parameterDomain = parameterDomain

class SamplingUniform(SamplingStrategy):
	def __init__(self, parameterDomain, nSamplesPerParameter):
		super().__init__(parameterDomain)
		self.nSamplesPerParameter = nSamplesPerParameter # Tuple
		self.params = list()
		for i, (pmin, pmax) in enumerate(parameterDomain):
			self.params.append(np.linspace(pmin, pmax, num=nSamplesPerParameter[i], endpoint=True))

	def __iter__(self):
		return itertools.product(*self.params)

class SamplingCartesianProduct(SamplingStrategy):
	def __init__(self, parameterDomain, nSamplesPerParameter, typeSamplingPerParameter):
		super().__init__(parameterDomain)
		self.nSamplesPerParameter = nSamplesPerParameter # Tuple
		self.params = list()
		for i, (pmin, pmax) in enumerate(parameterDomain):
			if typeSamplingPerParameter[i] == 'linear':
				self.params.append(np.linspace(pmin, pmax, num=nSamplesPerParameter[i], endpoint=True))
			elif typeSamplingPerParameter[i] == 'log':
				self.params.append(np.logspace(np.log10(pmin), np.log10(pmax), num=nSamplesPerParameter[i], endpoint=True))
			elif typeSamplingPerParameter[i] == 'random':
				self.params.append(np.random.uniform(low=pmin, high=pmax, size=nSamplesPerParameter[i]))
			else:
				raise Exception('Sampling' + typeSamplingPerParameter[i] +' not supported.')

	def __iter__(self):
		return itertools.product(*self.params)

class SamplingRandom(SamplingStrategy):
	def __init__(self, parameterDomain, nSamples):
		super().__init__(parameterDomain)
		self.idx = 0
		self.nSamples = nSamples
		self.paramMin = [p for (p, q) in parameterDomain.domain ]
		self.paramMax = [q for (p, q) in parameterDomain.domain ]

	def __iter__(self):
		return self

	def __next__(self):
		if self.idx < self.nSamples:
			self.idx += 1
			return np.random.uniform(low=self.paramMin, high=self.paramMax)
		else:
			self.idx = 0
			raise StopIteration