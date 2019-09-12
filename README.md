Description
===========

This folder contains the sources of the paper

  `Nonlinear model reduction on metric spaces. Application to one-dimensional conservative PDEs in Wasserstein spaces`

by V. Ehrlacher, D. Lombardi, O. Mula et F.-X. Vialard.

You can also find some videos of reconstructed dynamics with our algorithms.

Required software
=================
Python >= 3.7

Required modules: scipy, numpy, matplotlib, multiprocessing, cvxopt, itertools, os, sys, time, jsonpickle, pickle, argparse

Running the code
=================
The main file is test-all.py. To reproduce the results of the paper, run the command

  python3 test-all.py -p \<p\> --id \<id\> --offline
 
 where:
  -  \<p\> is the type of PDE problem (keys are: Burgers, ViscousBurgers, KdV, CamassaHolm)
  - \<id\> is the filename of the folder where results are stored
  - offline is an optional parameter to compute the offline phase

For instance, to reproduce the results on inviscous Burger's equation, run

  python3 test-all.py -p Burgers --id Burgers --offline
  
The whole computation takes about an hour. For the other problems, the computational time is longer.
