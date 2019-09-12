from snapshots import Burgers, KdV, ViscousBurgers, CamassaHolm
from multiprocessing import Pool

class DictionaryFactory():
	def __init__(self, samplingStrategy):
		self.samplingStrategy = samplingStrategy

	def generateSnapshots(self, Snapshot):
		p = Pool()
		snapshot_list = p.map(Snapshot, self.samplingStrategy)
		p.close()
		return snapshot_list