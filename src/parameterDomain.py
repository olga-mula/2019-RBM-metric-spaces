class ParameterDomain():
	def __init__(self, domain):
		self.domain = domain
		self.idx = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.idx < len(self.domain):
			self.idx += 1
			return self.domain[self.idx-1]
		else:
			self.idx = 0
			raise StopIteration

	def dimension(self):
		return len(self.domain)