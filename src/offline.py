import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import os
import sys
import jsonpickle
import pickle
import argparse

from snapshots import Burgers, KdV, ViscousBurgers, CamassaHolm
from parameterDomain import ParameterDomain
from sampling import SamplingUniform, SamplingRandom
from dictionary import DictionaryFactory
from spaceConstruction import SpaceConstructionRandom, SpaceConstructionPCA, SpaceConstructionLogPCA, SpaceConstructionGreedyBarycenter
from space import PCAspace, LogPCAspace, Barycenterspace


def load(Snapshot, directoryOffline):
	# Load objects from offline phase
	print('Loading objects from offline phase.')

	file = open(directoryOffline+'constructionPCA-'+Snapshot.problemType()+'.json')
	constructionPCA = jsonpickle.decode(file.read())

	file = open(directoryOffline+'constructionLogPCA-'+Snapshot.problemType()+'.json')
	constructionLogPCA = jsonpickle.decode(file.read())

	file = open(directoryOffline+'constructionBary-'+Snapshot.problemType()+'.json')
	constructionBary = jsonpickle.decode(file.read())

	file = open(directoryOffline+'paramDomainSnapshots-'+Snapshot.problemType()+'.json')
	paramDomainSnapshots = jsonpickle.decode(file.read())

	file = open(directoryOffline+'dict_snapshots_test-'+Snapshot.problemType()+'.json')
	dict_snapshots_test = jsonpickle.decode(file.read())

	return constructionPCA, constructionLogPCA, constructionBary, \
	paramDomainSnapshots, dict_snapshots_test

def run_offline_phase(Snapshot, directoryOffline, nSamplesTest=1000):
	start = time.time()
	print('Beginning computation of offline phase. This might take some time...')
	# Define parameter domain
	paramRange = Snapshot.paramRange()
	paramDomainSnapshots  = ParameterDomain(paramRange)

	# Sampling strategy: uniform or random (here we take a uniform mesh over the parameter domain)
	nSamplesPerParameter = Snapshot.nSamplesPerParameter()
	assert len(nSamplesPerParameter) == len(paramRange)
	samplingStrategy = SamplingUniform(paramDomainSnapshots, nSamplesPerParameter)

	# Dictionary for training
	print('Beginning computation of snapshots.')
	start_snap     = time.time()
	dict_factory   = DictionaryFactory(samplingStrategy)
	dict_snapshots = dict_factory.generateSnapshots(Snapshot)
	end_snap       = time.time()
	print('End computation of snapshots. Took '+str(end_snap-start_snap)+' sec.')

	# To create a reduced space,
	# we need to specify construction strategy:
	# we do PCA and log-PCA and Barycenter for the moment
	# =============================================
	print('Beginning space construction.')
	constructionPCA = SpaceConstructionPCA(dict_snapshots)
	constructionLogPCA = SpaceConstructionLogPCA(dict_snapshots)
	constructionBary = SpaceConstructionGreedyBarycenter(dict_snapshots)

	print('End space construction.')
	end = time.time()
	print('Finished offline phase. Took '+str(end-start)+' sec.')

	# Dictionary for test
	# New set of snapshots to do error tests. Sampling strategy: random
	samplingStrategyRandom = SamplingRandom(paramDomainSnapshots, nSamplesTest)
	# Dictionary
	dict_factory_test = DictionaryFactory(samplingStrategyRandom)
	dict_snapshots_test = dict_factory_test.generateSnapshots(Snapshot)

	# Save objects from offline phase
	print('Saving objects from offline phase')
	# Check if directory exists
	if not os.path.exists(directoryOffline):
		os.makedirs(directoryOffline)

	file = open(directoryOffline+'constructionPCA-'+Snapshot.problemType()+'.json', 'w')
	file.write(jsonpickle.encode(constructionPCA))

	file = open(directoryOffline+'constructionLogPCA-'+Snapshot.problemType()+'.json', 'w')
	file.write(jsonpickle.encode(constructionLogPCA))

	file = open(directoryOffline+'constructionBary-'+Snapshot.problemType()+'.json', 'w')
	file.write(jsonpickle.encode(constructionBary))

	file = open(directoryOffline+'paramDomainSnapshots-'+Snapshot.problemType()+'.json', 'w')
	file.write(jsonpickle.encode(paramDomainSnapshots))

	file = open(directoryOffline+'dict_snapshots_test-'+Snapshot.problemType()+'.json', 'w')
	file.write(jsonpickle.encode(dict_snapshots_test))

	return