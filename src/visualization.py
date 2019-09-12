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


def plots_reconstruction(Snapshot, data, folder):

	param = data['param']
	s_icdf = data['s_icdf']
	s_cdf = data['s_cdf']
	k = data['k']
	nl_err = data['nl_err']
	nl_video = data['nl_video']
	nl_recons = data['nl_recons']
	constructionPCA = data['constructionPCA']
	constructionLogPCA = data['constructionLogPCA']
	constructionBary = data['constructionBary']
	
	print('Starting reconstruction plots.')

	# Snapshot to reconstruct
	print(param)
	snapshot = Snapshot(param)

	# Space and density variables
	x = snapshot.x
	r = snapshot.r

	# Plot of all reconstructions
	font = {'family' : 'DejaVu Sans', 'weight' : 'normal', 'size'   : 14}
	matplotlib.rc('font', **font)

	for n in nl_recons:

		f = folder+'/n-'+str(n)+'/'
		if not os.path.exists(f):
			os.makedirs(f)

		# Create spaces and compute reconstructions
		# -----------------------------------------
		Vn_PCA = PCAspace(constructionPCA, n)
		Vn_logPCA = LogPCAspace(constructionLogPCA, n)
		Vn_Bary = Barycenterspace(constructionBary, n)

		start = time.time()
		print('Begin L2 projection n='+str(n))
		approxPCA, errL2, errHminus1_PCA = Vn_PCA.project(snapshot)
		end = time.time()
		print('End L2 projection n='+str(n)+'. Took '+str(end-start)+' sec.')

		start = time.time()
		print('Begin Exp mapping n='+str(n))
		approxLogPCA, cdf_approxLogPCA, icdf_approxLogPCA, icdf_exp_map_smoothed, errW2, errL2, errL1, errHminus1_logPCA = Vn_logPCA.exp(snapshot, type_approx='project', s_icdf=s_icdf, s_cdf=s_cdf, k=k)
		end = time.time()
		print('End Exp mapping n='+str(n)+'. Took '+str(end-start)+' sec.')

		start = time.time()
		print('Begin Wasserstein projection n='+str(n))
		approxBary, cdf_approxBary, icdf_approxBary, errL2, errW2, errHminus1, coeffs = Vn_Bary.reconstruction(snapshot, type_approx='project')
		end = time.time()
		print('End Wasserstein projection n='+str(n)+'. Took '+str(end-start)+' sec.')

		start = time.time()
		print('Begin Wasserstein interpolation n='+str(n))
		approxBary_i, cdf_approxBary_i, icdf_approxBary_i, errL2, errW2, errHminus1, coeffs = Vn_Bary.reconstruction(snapshot, type_approx='project')
		end = time.time()
		print('End Wasserstein interpolation n='+str(n)+'. Took '+str(end-start)+' sec.')

		# Plots reconstruction
		# --------------------

		# Exact
		plt.figure()
		plt.plot(x, snapshot.fun, 'k',label='Exact')
		plt.savefig(f+'fun-exact.pdf')
		plt.close()

		# PCA
		plt.figure()
		plt.plot(x, approxPCA, 'r',label='PCA n='+str(n))
		plt.savefig(f+'fun-PCA.pdf')
		plt.close()

		# logPCA
		plt.figure()
		plt.plot(x, approxLogPCA, 'g',label='tPCA n='+str(n))
		plt.savefig(f+'fun-tPCA.pdf')
		plt.close()

		# approxBary
		plt.figure()
		plt.plot(x, approxBary, 'b',label='Bary n='+str(n))
		plt.savefig(f+'fun-bary.pdf')
		plt.close()

		# approxBary
		plt.figure()
		plt.plot(x, approxBary_i, 'm',label='Bary interp n='+str(n))
		plt.savefig(f+'fun-bary-i.pdf')
		plt.close()

		# Plots cdf
		# ---------

		# Exact
		plt.figure()
		plt.plot(x, snapshot.cdf, 'k',label='Exact')
		plt.savefig(f+'cdf-exact.pdf')
		plt.close()

		# logPCA
		plt.figure()
		plt.plot(x, cdf_approxLogPCA, 'g',label='tPCA n='+str(n))
		plt.savefig(f+'cdf-tPCA.pdf')
		plt.close()

		# approxBary
		plt.figure()
		plt.plot(x, cdf_approxBary, 'b',label='Bary n='+str(n))
		plt.savefig(f+'cdf-bary.pdf')
		plt.close()

		# approxBary_i
		plt.figure()
		plt.plot(x, cdf_approxBary_i, 'm',label='Bary interp n='+str(n))
		plt.savefig(f+'cdf-bary-i.pdf')
		plt.close()

		# Plots icdf
		# ----------

		# Exact
		plt.figure()
		plt.plot(r, snapshot.icdf, 'k',label='Exact')
		plt.savefig(f+'icdf-exact.pdf')
		plt.close()

		# logPCA
		plt.figure()
		plt.plot(r, icdf_approxLogPCA, 'g',label='tPCA n='+str(n))
		plt.savefig(f+'icdf-tPCA.pdf')
		plt.close()

		# approxBary
		plt.figure()
		plt.plot(r, icdf_approxBary, 'b',label='Bary n='+str(n))
		plt.savefig(f+'icdf-bary.pdf')
		plt.close()

		# approxBary_i
		plt.figure()
		plt.plot(r, icdf_approxBary_i, 'm',label='Bary interp n='+str(n))
		plt.savefig(f+'icdf-bary-i.pdf')
		plt.close()


# def plots_error(nl_err, err_av_summary, err_max_summary, directoryPrefix):

def plots_error(directoryResults, data):

	err_av_summary = np.load(directoryResults+'err_av_summary.npy', allow_pickle=True)
	err_max_summary = np.load(directoryResults+'err_max_summary.npy', allow_pickle=True)

	nl_err = data['nl_err']

	print('Creating error plots.')

	# Directory to save plot
	directory = directoryResults + '/error/'
	# Check if directory exists
	if not os.path.exists(directory):
			os.makedirs(directory)

	# Average error in natural norms
	plt.figure()

	plt.semilogy(nl_err, [err['PCA-L2'] for err in  err_av_summary], 'r-x', label='PCA ($L_2$ norm)') 	# PCA
	plt.semilogy(nl_err, [err['logPCA-W2'] for err in  err_av_summary], 'g-x', label='tPCA ($W_2$ norm)') # logPCA
	plt.semilogy(nl_err, [err['Bary-W2'] for err in  err_av_summary], 'b-x', label='Bary ($W_2$ norm)') # Barycenter
	plt.semilogy(nl_err, [err['Bary-W2-interp'] for err in  err_av_summary], 'm-x', label='Bary interp ($W_2$ norm)') # Barycenter interp
	plt.legend()
	plt.savefig(directory + 'av-natural-norms.pdf')
	plt.close()

	# Worst error in natural norms
	plt.figure()

	plt.semilogy(nl_err, [err['PCA-L2'] for err in  err_max_summary], 'r-x', label='PCA ($L_2$ norm)') 	# PCA
	plt.semilogy(nl_err, [err['logPCA-W2'] for err in  err_max_summary], 'g-x', label='tPCA ($W_2$ norm)') # logPCA
	plt.semilogy(nl_err, [err['Bary-W2'] for err in  err_max_summary], 'b-x', label='Bary ($W_2$ norm)') # Barycenter
	plt.semilogy(nl_err, [err['Bary-W2-interp'] for err in  err_max_summary], 'm-x', label='Bary interp ($W_2$ norm)') # Barycenter interp
	plt.legend()
	plt.savefig(directory + 'wc-natural-norms.pdf')
	plt.close()

	# Average error in H^{-1}
	plt.figure()

	plt.semilogy(nl_err, [err['PCA-Hminus1'] for err in  err_av_summary], 'r-x', label='PCA ($H^{-1}$ norm)') 	# PCA
	plt.semilogy(nl_err, [err['logPCA-Hminus1'] for err in  err_av_summary], 'g-x', label='tPCA ($H^{-1}$ norm)') # logPCA
	plt.semilogy(nl_err, [err['Bary-Hminus1'] for err in  err_av_summary], 'b-x', label='Bary ($H^{-1}$ norm)') # Barycenter
	plt.semilogy(nl_err, [err['Bary-Hminus1-interp'] for err in  err_av_summary], 'm-x', label='Bary interp ($H^{-1}$ norm)') # Barycenter interp
	plt.legend()
	plt.savefig(directory + 'av-Hminus1.pdf')
	plt.close()

	# Worst error in H^{-1}
	plt.figure()

	plt.semilogy(nl_err, [err['PCA-Hminus1'] for err in  err_max_summary], 'r-x', label='PCA ($H^{-1}$ norm)') 	# PCA
	plt.semilogy(nl_err, [err['logPCA-Hminus1'] for err in  err_max_summary], 'g-x', label='tPCA ($H^{-1}$ norm)') # logPCA
	plt.semilogy(nl_err, [err['Bary-Hminus1'] for err in  err_max_summary], 'b-x', label='Bary ($H^{-1}$ norm)') # Barycenter
	plt.semilogy(nl_err, [err['Bary-Hminus1-interp'] for err in  err_max_summary], 'm-x', label='Bary interp ($H^{-1}$ norm)') # Barycenter interp
	plt.legend()
	plt.savefig(directory + 'wc-Hminus1.pdf')
	plt.close()

def video(Snapshot, data, directoryResults):

	paramRef = data['param']
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

	# Parameters
	param = None
	if Snapshot.problemType() == 'Burgers':
		y = paramRef[1]
		param = (0., y) # param = (t, y)
		titlePrefix = '(t, y) = ({:.4f}, {:.4f})'
		nFrames = 50
		ymin = -1.
		ymax = 4.
	elif Snapshot.problemType() == 'KdV':
		k2 = paramRef[1] # k2 in [16, 30]
		titlePrefix = '(t, k2) = ({:.4f}, {:.4f})'
		param = (0., k2) # param = (t, k2)
		nFrames = 30
		ymin = -100.
		ymax = 1000
	elif Snapshot.problemType() == 'ViscousBurgers':
		y = paramRef[1]
		nu = paramRef[2]
		param = (0., y, nu) # param = (t, y, nu)
		titlePrefix = '(t, y, nu) = ({:.4f}, {:.4f}, {:.4f})'
		nFrames = 30
		ymin = -1.
		ymax = 1.5/y
	elif Snapshot.problemType() == 'CamassaHolm':
		q1 = paramRef[1]
		a1 = paramRef[2]
		titlePrefix = '(t, q1, a1) = ({:.4f}, {:.4f}, {:.4f})'
		param = (0., q1, a1) # param = (t, q1, a1)
		nFrames = 30
		ymin = -0.5
		ymax = 1.5

	for n in nl_video:
		print('Making movie with dim n='+str(n))
		
		# Dimension of PCA and creation of PCA and logPCA spaces
		Vn_PCA = PCAspace(constructionPCA, n)
		Vn_logPCA = LogPCAspace(constructionLogPCA, n)
		Vn_Bary = Barycenterspace(constructionBary, n)
		
		# Space and time
		(xmin, xmax) = Snapshot.xRange()
		(tmin, tmax) = Snapshot.paramRange()[0]
		t = np.linspace(tmin, tmax, num=nFrames+1, endpoint=True)

		# dimensionnement de la fenêtre d'affichage
		fig,ax=plt.subplots()
		ax.set_title(titlePrefix.format(*param))
		ax.set_xlim((xmin,xmax))
		ax.set_ylim((ymin,ymax))

		# tracé de deux courbes (une pour la solution, une pour son approximation numérique)
		line, =ax.plot([],[], lw=2, color='k', label='Exact')
		line2,=ax.plot([],[], lw=2, color='r', label='PCA n='+str(n))
		line3,=ax.plot([],[], lw=2, color='g', label='LogPCA n='+str(n))
		line4,=ax.plot([],[], lw=2, color='b', label='Bary n='+str(n))
		plt.legend(handles=[line, line2, line3, line4])

		# fonction d'initialisation des tracés
		def init():
			line.set_data([],[])
			line2.set_data([],[])
			line3.set_data([],[])
			line4.set_data([],[])
			return (line,line2,line3,line4,)

		# fonction contruisant l'animation
		def animate(i):
			print('Computed frame '+str(i+1)+'/'+str(nFrames))
			param = None
			if Snapshot.problemType() == 'Burgers':
				param = (t[i], y) # param = (t, y)
			elif Snapshot.problemType() == 'KdV':   
				param = (t[i], k2) # param = (t, k2)
			elif Snapshot.problemType() == 'ViscousBurgers':
				param = (t[i], y, nu) # param = (t, y, nu)
			elif Snapshot.problemType() == 'CamassaHolm':
				param = (t[i], q1, a1) # param = (t, y, nu)
			
			snapshot = Snapshot(param)
			
			approxPCA, errL2, errHminus1_PCA = Vn_PCA.project(snapshot)
			
			approxLogPCA, cdf_approxLogPCA, icdf_exp_map, icdf_exp_map_smoothed, errW2, errL2, errL1, errHminus1_logPCA = Vn_logPCA.exp(snapshot, type_approx='project', s_icdf=s_icdf, s_cdf=s_cdf, k=k)
			
			approxBary, cdf_approxBary, approxBaryicdf, errL2, errW2, errHminus1, coeffs = Vn_Bary.reconstruction(snapshot, type_approx = 'project')
			
			ax.set_title(titlePrefix.format(*param)) # affichage de la valeur de l'instant courant (avec deux décimales)
			line.set_data(snapshot.x, snapshot.fun) # Exact solution at time t[i]
			line2.set_data(snapshot.x, approxPCA) # Approx solution at time t[i]
			line3.set_data(snapshot.x, approxLogPCA) # Approx solution at time t[i]
			line4.set_data(snapshot.x, approxBary) # Approx solution at time t[i]
			return (line,line2,line3,line4,)

		anim=animation.FuncAnimation(fig,animate,init_func=init,frames=nFrames,interval=150,blit=True)

		# Save movie
		print('Saving movie')

		directory = directoryResults
		if not os.path.exists(directory):
			os.makedirs(directory)

		Writer = animation.writers['ffmpeg']
		writer = Writer(fps=3, metadata=dict(artist='Olga Mula'), bitrate=1800)
		anim.save(directory + 'video-n-'+str(n)+'.mp4', writer=writer)

		end = time.time()
		print('Finished production of the movie. Took '+str(end-start)+' sec.')
		print()