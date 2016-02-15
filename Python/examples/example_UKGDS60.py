# -*- coding: utf-8 -*-
"""

Application of the NLO-IEKF to the subgrid of UKGDS60 with data from previous SmartGrid project

"""
# if run as script, add parent path for relative importing
if __name__ == '__main__' and __package__ is None:
	from os import sys, path
	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


# indices of measured active power
PMeasIdx = [1,2]
nPMeas = len(PMeasIdx)

# indices of measured reactive power
QMeasIdx = [1,2]
nQMeas = len(QMeasIdx)

# indices of measured bus voltage
VMeasIdx = [3,4,5,9]
nVMeas = len(VMeasIdx)
##########################################

import UKGDS60 as network
from matplotlib.pyplot import *
from scipy.io import loadmat
from NLO.dynamic_models import SimpleModel
from NLO.nodal_load_observer import IteratedExtendedKalman
# from NLO.nodal_load_observer import LinearKalmanFilter

# load data
topology = network.get_topology()
S, Vs, V, Y, Sfc = network.measured_data()

nNodes = S.shape[0]/2
nT = S.shape[1]
t = np.arange(15,15*(nT+1),15)

# Forecasts
if nPMeas == 0:
	SNMeasFc = Sfc
else:
	idx = np.zeros(2*nNodes,dtype=bool)
	idx[:nNodes][PMeasIdx] = 1
	idx[nNodes:][QMeasIdx] = 1
	notidx = ~idx
	SNMeasFc = Sfc[notidx,:]

############################################## IEKF
# set initial values
basekV = 11
Vhat0 = np.r_[basekV/(np.sqrt(3)*np.ones(nNodes)), np.zeros(nNodes)]

#--------------------------------------
n = 2*nNodes-nPMeas-nQMeas
model = SimpleModel(n, alpha = 0.95, q=10)
meas_idx = { "Pk": PMeasIdx, "Qk":  QMeasIdx, "Vm": VMeasIdx, "Va": VMeasIdx}
meas = { "Pk": S[:nNodes,:][PMeasIdx,:], "Qk": S[nNodes:,:][QMeasIdx,:], "Vm": V[:nNodes,:][VMeasIdx,:], "Va": V[nNodes:,:][VMeasIdx,:]}
pseudo_meas = {"Pk": SNMeasFc[:n,:], "Qk": SNMeasFc[n:,:]}
meas_unc = { "Vm": 1e-2*np.ones(nVMeas), "Va": 1e-2*np.ones(nVMeas) }

Shat, Vhat, uShat, DeltaS, uDeltaS = IteratedExtendedKalman(topology, meas, meas_unc, meas_idx, pseudo_meas, model,Vhat0,Vs,Y=Y)
# Shat, Vhat, uShat, DeltaS, uDeltaS = LinearKalmanFilter(topology, meas, meas_unc, meas_idx, pseudo_meas, model,Vhat0,Vs,Y=Y)
#--------------------------------------

Veff = np.sqrt( V[:nNodes,:]**2 + V[nNodes:,:]**2)
Vhateff = np.sqrt(Vhat[:nNodes,:]**2 + Vhat[nNodes:,:]**2)
Delta_Vhateff = Veff - Vhateff

###################################################
# load results from original MATLAB code for comparison
mat = loadmat("UKGDS60_data/results_matlab.mat")

figure(1,figsize = (18,10));clf()
for i in range(nNodes):
	subplot(3,4,i+1)
	plot(t, -1000*S[i,:],label="reference")
	plot(t, -1000*Shat[i,:],label="estimate")
	plot(t, -1000*Sfc[i,:],label="forecast")
	plot(t, -1000*mat["Shat"][i,:],'+',label="estimate MATLAB")
	if i in PMeasIdx:
		title("bus %d (measured)"%i)
	else:
		title("bus %d"%i)
legend(loc="upper left",bbox_to_anchor=(1.2,0.7))
subplots_adjust(left=0.05,right=0.98,top=0.95,bottom=0.05)

figure(2,figsize = (18,10));clf()
for i in range(nNodes):
	subplot(3,4,i+1)
	plot(t, -1000*S[nNodes+i,:],label="reference")
	plot(t, -1000*Shat[nNodes+i,:],label="estimate")
	plot(t, -1000*Sfc[nNodes+i,:],label="forecast")
	plot(t, -1000*mat["Shat"][nNodes+i,:],'+',label="estimate MATLAB")
	title("bus %d"%i)
legend(loc="upper left",bbox_to_anchor=(1.2,0.7))
subplots_adjust(left=0.05,right=0.98,top=0.95,bottom=0.05)

figure(3,figsize = (18,10));clf()
for i in range(nNodes):
	subplot(3,4,i+1)
	plot(t, np.sqrt(3)*Veff[i,:],'b',label="reference")
	plot(t, np.sqrt(3)*Vhateff[i,:],'r',label="estimate")
	title("bus %d"%i)
legend(loc="upper left",bbox_to_anchor=(1.2,0.7))
subplots_adjust(left=0.05,right=0.98,top=0.95,bottom=0.05)

figure(4,figsize = (18,10));clf()
up = np.sqrt(3)*Delta_Vhateff.max()*1.05
down = np.sqrt(3)*Delta_Vhateff.min()
for i in range(nNodes):
	subplot(3,4,i+1)
	plot(t, np.sqrt(3)*Delta_Vhateff[i,:])
	ylim(down,up)
	title("bus %d"%i)
subplots_adjust(left=0.05,right=0.98,top=0.95,bottom=0.05)

show()