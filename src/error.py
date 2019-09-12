import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import os
import sys
import pickle
import argparse

from offline import run_offline_phase, load
from snapshots import Burgers, KdV, ViscousBurgers, CamassaHolm
# from parameterDomain import ParameterDomain
# from sampling import SamplingUniform, SamplingRandom
# from dictionary import DictionaryFactory
from spaceConstruction import SpaceConstructionRandom, SpaceConstructionPCA, SpaceConstructionLogPCA, SpaceConstructionGreedyBarycenter
from space import PCAspace, LogPCAspace, Barycenterspace

# Error analysis for different dimensions of PCA
# Error in L1, L2 and W2
# ==============================================

def error_analysis(dict_snapshots_test, data, directoryResults):

	s_icdf = data['s_icdf']
	s_cdf = data['s_cdf']
	k = data['k']
	nl_err = data['nl_err']
	nl_video = data['nl_video']
	nl_recons = data['nl_recons']
	constructionPCA = data['constructionPCA']
	constructionLogPCA = data['constructionLogPCA']
	constructionBary = data['constructionBary']

	start = time.time()
	print('Beginning error analysis. This might take some time...')

	# Loop over PCA dimension and error computation
	err_av_summary = list()
	err_max_summary = list()

	for n in nl_err:
		print('Dimension: ', n)
		# Generate PCA and LogPCA space with dictionary trial
		Vn_PCA = PCAspace(constructionPCA, n)
		Vn_logPCA = LogPCAspace(constructionLogPCA, n)
		Vn_Bary = Barycenterspace(constructionBary, n)
		# Average error PCA (computed with singular values)
		errPCA    = Vn_PCA.err
		errLogPCA = Vn_logPCA.err
		errBary = Vn_Bary.err
		errBaryav = Vn_Bary.errav
		# Average error Log PCA: Error for each snapshot of the test set
		err_snapshots_PCA              = list()
		err_snapshots_logPCA           = list()
		# err_snapshots_logPCA_interp    = list()
		err_snapshots_bary             = list()
		err_snapshots_bary_interp      = list()
		# err_snapshots_bary_true        = list()
		# err_snapshots_bary_interp_true = list()
		for snapshot in dict_snapshots_test:
			# PCA
			approx, errL2, errHminus1_PCA = Vn_PCA.project(snapshot)
			err_snapshots_PCA.append({'L2': errL2, 'Hminus1': errHminus1_PCA})
			# logPCA
			approx, cdf_approx, icdf_exp_map, icdf_exp_map_smoothed, errW2, errL2, errL1, errHminus1 = Vn_logPCA.exp(snapshot, type_approx='project', s_icdf=s_icdf, s_cdf=s_cdf, k=k)
			err_snapshots_logPCA.append({'W2': errW2, 'L2': errL2, 'L1': errL1, 'Hminus1': errHminus1})
			# logPCA_interp
			# approx, cdf_approx, icdf_exp_map, icdf_exp_map_smoothed, errW2, errL2, errL1, errHminus1 = Vn_logPCA.exp(snapshot, type_approx='interpolate', s_icdf=s_icdf, s_cdf=s_cdf, k=k)
			# err_snapshots_logPCA_interp.append({'W2': errW2, 'L2': errL2, 'L1': errL1, 'Hminus1': errHminus1})
			# barycenter project
			approxBary, cdf_approxBary, approxBaryicdf, errL2, errW2, errHminus1, coeffs = Vn_Bary.reconstruction(snapshot, type_approx='project')
			err_snapshots_bary.append({'W2': errW2, 'L2': errL2, 'Hminus1': errHminus1})
		   # barycenter interpolate
			approxBary, cdf_approxBary, approxBaryicdf, errL2, errW2, errHminus1, coeffs = Vn_Bary.reconstruction(snapshot, type_approx='interpolate')
			err_snapshots_bary_interp.append({'W2': errW2, 'L2': errL2, 'Hminus1': errHminus1})


		err_av_summary.append({ \
			'PCA-L2-singular-val': errPCA, \
			'PCA-L2': np.mean([err['L2'] for err in err_snapshots_PCA]), \
			'PCA-Hminus1': np.mean([err['Hminus1'] for err in err_snapshots_PCA]), \

			'logPCA-L2-singular-val': errLogPCA, \
			'logPCA-W2': np.mean([err['W2'] for err in  err_snapshots_logPCA]), \
			'logPCA-L2': np.mean([err['L2'] for err in  err_snapshots_logPCA]), \
			'logPCA-L1': np.mean([err['L1'] for err in  err_snapshots_logPCA]), \
			'logPCA-Hminus1': np.mean([err['Hminus1'] for err in  err_snapshots_logPCA]), \
			# 'logPCA-W2-interp': np.mean([err['W2'] for err in  err_snapshots_logPCA_interp]), \
			# 'logPCA-L2-interp': np.mean([err['L2'] for err in  err_snapshots_logPCA_interp]), \
			# 'logPCA-L1-interp': np.mean([err['L1'] for err in  err_snapshots_logPCA_interp]), \
			# 'logPCA-Hminus1-interp': np.mean([err['Hminus1'] for err in  err_snapshots_logPCA_interp]), \

			'Bary-W2-singular-val-max': errBary, \
			'Bary-W2-singular-val-av': errBaryav, \
			'Bary-W2': np.mean([err['W2'] for err in  err_snapshots_bary]), \
			'Bary-L2': np.mean([err['L2'] for err in  err_snapshots_bary]), \
			'Bary-Hminus1': np.mean([err['Hminus1'] for err in  err_snapshots_bary]), \
			'Bary-W2-interp': np.mean([err['W2'] for err in  err_snapshots_bary_interp]), \
			'Bary-L2-interp': np.mean([err['L2'] for err in  err_snapshots_bary_interp]), \
			'Bary-Hminus1-interp': np.mean([err['Hminus1'] for err in  err_snapshots_bary_interp]), \
			})

		err_max_summary.append({ \
			'PCA-L2-singular-val': errPCA, \
			'PCA-L2': np.max([err['L2'] for err in err_snapshots_PCA]), \
			'PCA-Hminus1': np.max([err['Hminus1'] for err in err_snapshots_PCA]), \
			'logPCA-L2-singular-val': errLogPCA, \
			'logPCA-W2': np.max([err['W2'] for err in  err_snapshots_logPCA]), \
			'logPCA-L2': np.max([err['L2'] for err in  err_snapshots_logPCA]), \
			'logPCA-L1': np.max([err['L1'] for err in  err_snapshots_logPCA]), \
			'logPCA-Hminus1': np.max([err['Hminus1'] for err in  err_snapshots_logPCA]), \
			# 'logPCA-W2-interp': np.max([err['W2'] for err in  err_snapshots_logPCA_interp]), \
			# 'logPCA-L2-interp': np.max([err['L2'] for err in  err_snapshots_logPCA_interp]), \
			# 'logPCA-L1-interp': np.max([err['L1'] for err in  err_snapshots_logPCA_interp]), \
			# 'logPCA-Hminus1-interp': np.max([err['Hminus1'] for err in  err_snapshots_logPCA_interp]), \
			'Bary-W2-singular-val-max': errBary, \
			'Bary-W2-singular-val-av': errBaryav, \
			'Bary-W2': np.max([err['W2'] for err in  err_snapshots_bary]), \
			'Bary-L2': np.max([err['L2'] for err in  err_snapshots_bary]), \
			'Bary-Hminus1': np.max([err['Hminus1'] for err in  err_snapshots_bary]), \
			'Bary-W2-interp': np.max([err['W2'] for err in  err_snapshots_bary_interp]), \
			'Bary-L2-interp': np.max([err['L2'] for err in  err_snapshots_bary_interp]), \
			'Bary-Hminus1-interp': np.max([err['Hminus1'] for err in  err_snapshots_bary_interp]), \
			})

		directory = directoryResults
		if not os.path.exists(directory):
			os.makedirs(directory)

		np.save(directory+'err_av_summary.npy', err_av_summary)
		np.save(directory+'err_max_summary.npy', err_max_summary)
		np.save(directory+'data.npy', data)

	end = time.time()
	print('End error analysis. Took '+str(end-start)+' sec.')

	return err_av_summary, err_max_summary