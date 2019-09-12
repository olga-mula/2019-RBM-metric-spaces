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
from visualization import plots_reconstruction, plots_error, video
from error import error_analysis
from snapshots import Burgers, KdV, ViscousBurgers, CamassaHolm
from spaceConstruction import SpaceConstructionRandom, SpaceConstructionPCA, SpaceConstructionLogPCA, SpaceConstructionGreedyBarycenter
from space import PCAspace, LogPCAspace, Barycenterspace


# MAIN PROGRAM
# ============
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--problemType', help='{Burgers, KdV, ViscousBurgers, CamassaHolm}')
parser.add_argument('--offline', help='Run offline phase', action='store_true')
parser.add_argument('--id', help='Job ID for output folder')
args = parser.parse_args()

# Folder management
# -----------------
directoryPrefix = ''
if args.id is not None:
    directoryPrefix = args.problemType + '/' + args.id + '/'
else:
    directoryPrefix = args.problemType + '/default/'
directoryOffline = directoryPrefix + 'offline/'
directoryResults = directoryPrefix + 'img/'

# Check if directory exists
if not os.path.exists(directoryPrefix):
    os.makedirs(directoryPrefix)
if not os.path.exists(directoryOffline):
    os.makedirs(directoryOffline)
if not os.path.exists(directoryResults):
    os.makedirs(directoryResults)

# Dictionary of problems
# ----------------------
problemDict = {'Burgers': Burgers, 'KdV': KdV, 'ViscousBurgers': ViscousBurgers, 'CamassaHolm': CamassaHolm}
Snapshot = None
if args.problemType in problemDict:
    Snapshot = problemDict[args.problemType]
else:
    Snapshot = problemDict['Burgers']

print('General info')
print('============')
print(Snapshot.get_info())

# Offline phase
# -------------
print('Offline phase')
print('=============')
if args.offline:
    run_offline_phase(Snapshot, directoryOffline, nSamplesTest = 500) 

constructionPCA, constructionLogPCA, constructionBary, \
    paramDomainSnapshots, dict_snapshots_test=load(Snapshot, directoryOffline)
    

# Plots
# -----
# Plots of one snapshot

if Snapshot.problemType() == 'Burgers':
    # Snapshot to reconstruct
    param = (1.8985, 0.865) # param = (t, y)
    # For smoothing
    s_icdf  = 0.   # Not used in Burgers
    s_cdf   = 0.   # Not used in Burgers
    k       = 3
    # Dimensions
    nl_err = [1, 5, 10, 15, 20] # for error analysis
    nl_video = [5, 7, 10]          # for video
    nl_recons = [5, 10]             # for plots reconstruction

    data = {'param': param, 's_icdf': s_icdf, 's_cdf': s_cdf, 'k': k, 'nl_err': nl_err, 'nl_video': nl_video, 'nl_recons': nl_recons, 'constructionPCA': constructionPCA, 'constructionLogPCA': constructionLogPCA, 'constructionBary': constructionBary}

elif Snapshot.problemType() == 'KdV':
    # Snapshot to reconstruct
    param = (2.2e-3, 18.0)  # param = (t, k2)
    # For smoothing
    s_icdf  = 30 * 2.e-3 # 30 * nr
    s_cdf   = 100.  # nx/5
    k       = 3
    # Dimensions
    nl_err = [1, 5, 10, 15, 20] # for error analysis
    nl_video = [5, 10]          # for video
    nl_recons = [5, 7, 10]      # for plots reconstruction

    data = {'param': param, 's_icdf': s_icdf, 's_cdf': s_cdf, 'k': k, 'nl_err': nl_err, 'nl_video': nl_video, 'nl_recons': nl_recons, 'constructionPCA': constructionPCA, 'constructionLogPCA': constructionLogPCA, 'constructionBary': constructionBary}

elif Snapshot.problemType() == 'ViscousBurgers':
    # Snapshot to reconstruct
    param = (2./3., 0.865, 0.00257) # param = (t, y, nu)
    # For smoothing
    s_icdf  = 2.e-3
    s_cdf   = 0.
    k       = 3
    # Dimensions
    nl_err = [1, 5, 10, 15, 20]    # for error analysis
    nl_video = [3, 5, 7, 10]       # for video
    nl_recons = [3, 5, 7, 10]      # for plots reconstruction

    data = {'param': param, 's_icdf': s_icdf, 's_cdf': s_cdf, 'k': k, 'nl_err': nl_err, 'nl_video': nl_video, 'nl_recons': nl_recons, 'constructionPCA': constructionPCA, 'constructionLogPCA': constructionLogPCA, 'constructionBary': constructionBary}

elif Snapshot.problemType() == 'CamassaHolm':
    # Snapshot to reconstruct
    param = (9., -1.1342, 0.3231) # param = (t, y, nu)
    # For smoothing
    s_icdf  = 2.e-3
    s_cdf   = 0.
    k       = 3
    # Dimensions
    nl_err = [1, 5, 10, 15, 20] # for error analysis
    nl_video = [3, 5, 7, 10]          # for video
    nl_recons = [3, 5, 7, 10]      # for plots reconstruction

    data = {'param': param, 's_icdf': s_icdf, 's_cdf': s_cdf, 'k': k, 'nl_err': nl_err, 'nl_video': nl_video, 'nl_recons': nl_recons, 'constructionPCA': constructionPCA, 'constructionLogPCA': constructionLogPCA, 'constructionBary': constructionBary}


# Error analysis
# ==============
err_av_summary, err_max_summary = \
error_analysis(dict_snapshots_test, data, directoryResults)
plots_error(directoryResults, data)

# Plot decay error
# ================
eig_tol = 1.e-13
sigPCA = constructionPCA.S[constructionPCA.S > eig_tol]
errPCA = [ np.sqrt(np.sum(sigPCA[n:])) for n in range(len(sigPCA))]

sigLogPCA = constructionLogPCA.S[constructionLogPCA.S > eig_tol]
errLogPCA = [ np.sqrt(np.sum(sigLogPCA[n:])) for n in range(len(sigLogPCA))]

plt.figure()
plt.semilogy(errPCA, label='PCA-L_2 norm')
plt.semilogy(errLogPCA, label='tPCA-W_2 norm')
plt.legend()
plt.savefig(directoryResults+'decay-err-training.pdf')


# Plot of a snapshot and its reconstruction
# =========================================
plots_reconstruction(Snapshot, data, directoryResults)

# Videos
# ======
video(Snapshot, data, directoryResults)




