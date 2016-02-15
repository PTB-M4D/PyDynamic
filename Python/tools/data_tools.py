# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:16:50 2014

@author: Sascha Eichstaedt
"""
import numpy as np
from pypower.idx_bus import BUS_I,BUS_TYPE,PD,QD,GS,BS,BUS_AREA,VM,VA,BASE_KV,ZONE,VMAX,VMIN
from pypower.idx_brch import F_BUS,T_BUS,BR_R,BR_X,BR_B,RATE_A,RATE_B,RATE_C,TAP,SHIFT,BR_STATUS,ANGMIN,ANGMAX
from pypower.idx_gen import GEN_BUS,PG,QG,QMAX,QMIN,VG,MBASE,GEN_STATUS,PMAX,PMIN,PC1,PC2,QC1MIN,QC1MAX,QC2MIN,QC2MAX,RAMP_AGC,RAMP_10,RAMP_30,RAMP_Q,APF

bustypes = {"PQ":1,"PV":2,"Slack":3,"None":4}

def bus_branch_from_UKGDSexcel(filename,header=28):
	"""
	Load Excel file with UKGDS data format and build bus and branch matrices
	in the format required by PyPower. Some elements appear to missing in UKGDS
	data format whereas others are not considered in bus and branch matrices by
	PyPower.
	"""
	import numpy as np
	import pandas as pd

	if isinstance(filename,pd.io.excel.ExcelFile):  # filename is already a pandas object
		data = filename
	elif isinstance(filename,str):
		data = pd.ExcelFile(filename)
	else:
		raise ValueError("filename needs to be either a string or a pandas ExcelFile object")

	bus_data    =   data.parse("Buses",header=header)
	load_data   =   data.parse("Loads",header=header)
	shunt_data  =   data.parse("Shunts",header=header)
	branch_data =   data.parse("Branches",header=header)
	trans_data  =   data.parse("Transformers",header=header)
	system = data.parse("System",header=header)

	# base MVA
	baseMVA = system[system["Symbol"]=="SMB"]["Value"][0]

############ Building bus data matrix in PyPower format
	nb = bus_data.shape[0]
	bus_matrix = np.zeros((nb,13))
	# bus numbers
	bus_matrix[:,BUS_I] = bus_data["BNU"].values[:]
	# bus type
	for k in range(nb):
		bus_matrix[k,BUS_TYPE] = bustypes[bus_data["BTY"][k]]
	# real power demand
	bus_matrix[2:,PD] = load_data["LPO"].values[:] # no load @ reference
	# reactive power demand
	bus_matrix[2:,QD] = load_data["LQA"].values[:]
	# shunt conductance and susceptance
	if shunt_data.shape[0]==1 and shunt_data["SHB"][0]==np.NaN:
		bus_matrix[:,GS] = 0
		bus_matrix[:,BS] = 0
	# bus area number
	bus_matrix[:,BUS_AREA] = 1         # not available in data
	# voltage magnitude
	bus_matrix[:,VM] = bus_data["BTV"][:]
	bus_matrix[(bus_matrix[:,BUS_TYPE]==3).nonzero()[0],VM] *= 3        # ASSERTION THAT SLACK NODES VOLTAGES HAVE TO BE MULTIPLIED BY 3
	# voltage angle
	bus_matrix[:,VA] = bus_data["BVA"][:]
	# base voltage
	bus_matrix[:,BASE_KV] = bus_data["BBV"][:]
	# loss zone
	bus_matrix[:,ZONE]= 1         # not available in data
	# max voltage magnitude
	bus_matrix[:,VMAX]= bus_data["BVX"][:]
	# min voltage magnitude
	bus_matrix[:,VMIN]= bus_data["BVN"][:]

############# Building branch data matrix in PyPower format
	nbr = branch_data.shape[0]
	nt  = len( (trans_data["TID"]==1).nonzero()[0])
	branch_matrix = np.zeros((nbr+nt,13))
	# store branches that aren't transformers in upper part of matrix

	# from and to bus numbers           # WHAT IS THE CONVENTION ON WHERE TO PUT TRANSFORMERS IN BRANCH MATRIX??
	for k in range(nt):
		branch_matrix[k,F_BUS] = trans_data["TFB"][k]
		branch_matrix[k,T_BUS] = trans_data["TTB"][k]
	for k in range(nbr):
		branch_matrix[nt+k,F_BUS] = branch_data["CFB"][k]
		branch_matrix[nt+k,T_BUS] = branch_data["CTB"][k]

	# resistance
	branch_matrix[:,BR_R][nt:] = branch_data["CR1"][:]
	# reactance
	branch_matrix[:,BR_X][nt:] = branch_data["CX1"][:]
	# line susceptance
	branch_matrix[:,BR_B][nt:] = branch_data["CB1"][:]
	# MVA long term rating
	branch_matrix[:,RATE_A][nt:] = branch_data["CM1"][:]
	# MVA short term
	branch_matrix[:,RATE_B][nt:] = branch_data["CM2"][:]
	# MVA emergency rating
	branch_matrix[:,RATE_C][nt:] = branch_data["CM3"][:]
	# branch status (on/off)
	branch_matrix[:,BR_STATUS][nt:] = branch_data["CST"][:]
	# transformer off nominal turns ratio and others
	branch_matrix[:,TAP] = 1
	for k in range(nt):
		branch_matrix[k,BR_R]= trans_data["TR1"][k]
		branch_matrix[k,BR_X]= trans_data["TX1"][k]
		branch_matrix[k,RATE_A] = 0 #trans_data["TM1"][k]
		branch_matrix[k,RATE_B] = 0 #trans_data["TM2"][k]
		branch_matrix[k,RATE_C] = 0 #trans_data["TM3"][k]
		branch_matrix[k,TAP] = abs(1-trans_data["TTR"][k])*100
		branch_matrix[k,SHIFT] = trans_data["TPS"][k]
		branch_matrix[k,BR_STATUS] = trans_data["TST"][k]
		branch_matrix[k,]

	branch_matrix[:,ANGMIN] = -360     # not found in data
	branch_matrix[:,ANGMAX] =  360     # not found in data

	return baseMVA,bus_matrix,branch_matrix



def UKGDS_to_casedata(filename,header=28):
	"""
	Load Excel file with UKGDS data format and build casedata dict with bus, branch
	generator and generator cost information in the format required by PyPower.
	"""
	import numpy as np
	import pandas as pd

	casedata = dict()

	# load baseMVA, bus and branch data
	casedata["baseMVA"],casedata["bus"],casedata["branch"] = bus_branch_from_UKGDSexcel(filename,header)

	# load generator data
	data = pd.ExcelFile(filename)
	gen_data  = data.parse("Generators",header=header)
	ng = gen_data.shape[0]
	gen_matrix = np.zeros((ng,21))

	# bus number
	gen_matrix[:,GEN_BUS] = gen_data["GBN"][:]
	# real power output (MW)
	gen_matrix[:,PG] = gen_data["GPO"][:]
	# reactive power output (MVAr)
	gen_matrix[:,QG] = gen_data["GQA"][:]
	# max reactive power output (MVAr)
	gen_matrix[:,QMAX] = gen_data["GQX"][:]
	# min reactive power output (MVAr)
	gen_matrix[:,QMIN] = gen_data["GQN"][:]
	# voltage magn setpoint (p.u.)
	for kb in range(ng):
		gen_matrix[kb,VG] = casedata["bus"][(casedata["bus"][:,BUS_I]==gen_matrix[kb,GEN_BUS]).nonzero()[0],VM]
	# total MVA base of machine, defaults to baseMVA
	gen_matrix[:,MBASE] = gen_data["GMB"][:]
	# generator status
	gen_matrix[:,GEN_STATUS] = gen_data["GST"][:]
	# max real power (MW)
	gen_matrix[:,PMAX] = gen_data["GPX"][:]
	# min real power (MW)
	gen_matrix[:,PMIN] = gen_data["GPN"][:]
	# lower real power of PQ capability curve (MW)
		# not set in EXCEL file
	# upper real power of PQ capability curve (MW)
		# not set in EXCEL file
	# min react power at PC1
		# not set in EXCEL file
	# max react power at PC1
		# not set in EXCEL file
	# min react power at PC2
		# not set in EXCEL file
	# max react power in PC2
		# not set in EXCEL file
	# ramp rate for load following/AGC
		# not set in EXCEL file
	# ramp rate for 10 minute reserve
		# not set in EXCEL file
	# ramp rate for 30 minute reserve
		# not set in EXCEL file
	# ramp rate for reactive power (2 sec timescale)
		# not set in EXCEL file
	# are participation factor
		# not set in EXCEL file

	casedata["gen"] = gen_matrix

	return casedata


def separate_Yslack(Y,slack_idx,**kwargs):
	from scipy.sparse import issparse
	if issparse(Y):
		Y = Y.copy().toarray()
	n = Y.shape[0]
	Yslack = np.delete(Y[:,0],slack_idx)[:,np.newaxis]
	Y = np.delete(np.delete(Y,slack_idx,0),slack_idx,1)
	Yslack = np.vstack((np.hstack((Yslack.real, -Yslack.imag)), np.hstack((Yslack.imag, Yslack.real))))
	Y = np.vstack((np.hstack((Y.real, -Y.imag)), np.hstack((Y.imag, Y.real))))

	if not Y.shape == (2*(n-1),2*(n-1)):
			raise ValueError("Shape of calculated admittance w/o slack is wrong!\nShould be (%s,%s), but is actually (%s,%s)"%(2*(n-1),2*(n-1),Y.shape[0],Y.shape[1]))
	if not Yslack.shape == (2*(n-1),2):
			raise ValueError("Shape of calculated slack admittance is wrong!\nShould be (%s,%s), but is actually (%s,%s)"%(2*(n-1),2,Yslack.shape[0],Yslack.shape[1]))

	return Y, Yslack


def makeYbus(baseMVA,bus,branch,do_separate_Yslack=False):
	from pypower.makeYbus import makeYbus
	if do_separate_Yslack:
		Y = makeYbus(baseMVA,bus,branch)[0].toarray()
		slack_ind = (bus[:,BUS_TYPE]==3).nonzero()[0][0]
		return separate_Yslack(Y,slack_ind)
	else:
		return makeYbus(baseMVA,bus,branch)[0]

def admittance_from_UKGDS(filename,header=28,separate_Yslack=True):
	baseMVA,bus_matrix,branch_matrix = bus_branch_from_UKGDSexcel(filename,header)
	return makeYbus(baseMVA,bus_matrix,branch_matrix,separate_Yslack)


def process_admittance(Y,slack_idx=0):
	"""Decompose nodal admittance matrix into blocks required for NLO
	"""
	from numpy import real,imag
	# admittance at slack
	Y,Ys = separate_Yslack(Y,slack_idx)
	Gs = real(Ys[:])
	Bs = imag(Ys[:])
#    Ys = np.asarray(np.bmat([ [Gs, -Bs], [Bs, Gs] ]))
	# admittance w/o slack
	G = real(Y)
	B = imag(Y)
	Y00 = np.asarray(np.bmat([ [G, -B], [B, G] ]))
	return G,B,Gs,Bs,Y00,Ys



def draw_UKGDS(filename,header=28):
	"""
	Load Excel file with UKGDS data format and plot network graph.
	"""
	from networkx import draw
	from matplotlib.pyplot import show
	G,pos = network_UKGDS(filename,header=header)
	draw(G,pos)
	show()


def draw_network(bus,branch,bus_names=None,coordinates=None,ax=None):
	"""Generate networkx Graph object and draw network diagram
	It is assumed that bus and branch ordering corresponds to PyPower format
	"""
	from networkx import Graph,draw
	from matplotlib.pyplot import show
	if isinstance(coordinates,np.ndarray):
		pos = {}
		if coordinates.shape[0]==2:
			coordinates = coordinates.T
		for node in range(len(bus)):
			pos.update({node:coordinates[node,:]})
	else:
		pos = None
	net = []
	for k in range(len(branch)):
		von = np.where(bus[:,BUS_I]==branch[k,F_BUS])[0][0]
		zu  = np.where(bus[:,BUS_I]==branch[k,T_BUS])[0][0]
		net.append([von,zu])
	nodes = set([n1 for n1,n2 in net] + [n2 for n1,n2 in net])
	G = Graph()
	for node in nodes:
		G.add_node(node)
	for edge in net:
		G.add_edge(edge[0],edge[1])
	draw(G,pos,ax=ax)
	show()



def draw_graph(graph,pos=None):
	"""Plot graphical representation of network represented by graph
	Parameter `pos` optionally contains a dict of x and y coordinates for each node,
	see :function:`network_UKGDS`
	"""
	import networkx as nx
	from matplotlib.pyplot import show
	nodes = set([n1 for n1,n2 in graph] + [n2 for n1,n2 in graph])
	G = nx.Graph()
	for node in nodes:
		G.add_node(node)
	for edge in graph:
		G.add_edge(edge[0],edge[1])
	nx.draw(G,pos)
	show()


def network_UKGDS(filename,header=28):
	"""
	Load Excel file with UKGDS data format and build dict array of bus coordinates
	and graph structure suitable for plotting with the networkx module.
	"""
	from numpy import array,where
	from pandas import ExcelFile
	from networkx import Graph

	data = ExcelFile(filename)
	bus = data.parse("Buses",header=header)
	branch = data.parse("Branches",header=header)
	pos = {}
	for node in range(len(bus["BNU"])):
		pos.update({node:array([bus["BXC"][node],bus["BYC"][node]])})
	net = []
	for k in range(len(branch["CFB"])):
		von = where(bus["BNU"]==branch["CFB"][k])[0][0]
		zu  = where(bus["BNU"]==branch["CTB"][k])[0][0]
		net.append([von,zu])
	nodes = set([n1 for n1,n2 in net] + [n2 for n1,n2 in net])
	G = Graph()
	for node in nodes:
		G.add_node(node)
	for edge in net:
		G.add_edge(edge[0],edge[1])
	return G,pos


from pypower.idx_gen import GEN_BUS


def meas_at_time(meas,meas_unc,ind):
	"""Returns dictionary with the entries of meas at time index ind.
	:param meas: dict of measurements
	:param ind: time index
	:return: meas_ind
	"""
	meas_ind = dict([])
	umeas_ind = dict([])
	for key in meas.keys():
		if len(meas[key].shape)==2:
			meas_ind[key] = meas[key][:,ind]
		elif len(meas[key])>0:
			meas_ind[key] = meas[key][ind]
		else:
			meas_ind[key] = []

		if len(meas_unc[key].shape)==2:
			umeas_ind[key] = meas[key][:,ind]
		elif len(meas_unc[key])>0:
			umeas_ind[key] = meas[key][ind]
		else:
			umeas_ind[key] = []
	return meas_ind, umeas_ind


def calc_admittance(network_branches):
	""" From network branch information in PyPower format calculate the bus and network admittances.

	:param network_branches: numpy array contain all information about the network branches
	:returns: bus admittance matrix Y, line admittances y and line capacities cap

	"""
	nK = int(network_branches[:,:2].max())
	if network_branches[:,:2].min() == 0: # assume python indices
		nK += 1
	dim  = network_branches.shape[0]
	r = np.zeros((nK,nK))
	x = np.zeros_like(r)
	cap=np.zeros_like(r)        # Capacity between branch and earth

	for i in range(dim):
		k_start = network_branches[i,0]-1
		k_end = network_branches[i,1]-1
		r[k_start,k_end] = network_branches[i,2]
		x[k_start,k_end] = network_branches[i,3]
		cap[k_start,k_end]=network_branches[i,4]

	r += r.T
	x += x.T
	cap += cap.T
	z = r + 1j*x
	y = np.zeros_like(z)
	for i in range(nK):
		for j in range(nK):
			if z[i,j] != 0:
				y[i,j] = 1/z[i, j]
	Y = -y
	for i in range(nK):
		Y[i,i] = np.sum(y[i,:])+1j*np.sum(cap[i,:]/2)
	return Y, y, cap



def jacobian_dSdV(Y, nK):
	# Set up equations of power flow (bus power from nodal voltage) as symbolic equations and
	# calculate the corresponding Jacobian matrix.
	from sympy import symbols, Matrix

	G = Matrix(np.real(Y))
	B = Matrix(np.imag(Y))

	e_J = symbols("e1:%d"%(nK+1))
	f_J = symbols("f1:%d"%(nK+1))

	hSK = []
	for i in range(nK):
		hSK.append(e_J[i]*(G[i,:].dot(e_J) - B[i,:].dot(f_J)) + f_J[i]*(G[i,:].dot(f_J) + B[i,:].dot(e_J)))
	for i in range(nK):
		hSK.append(f_J[i]*(G[i,:].dot(e_J) - B[i,:].dot(f_J)) - e_J[i]*(G[i,:].dot(f_J) + B[i,:].dot(e_J)))
	hSK = Matrix(hSK)
	ef_J = e_J[1:] + f_J[1:]

	Jac_equ = hSK.jacobian(ef_J)

	return Jac_equ, hSK


def jacobian_dHdV(nK, y, cap, inds):
	# Set up equations of power flow (power at line from nodal voltage) as symbolic equations and
	# calculate the corresponding Jacobian matrix.
	from sympy import symbols, Matrix

	g = Matrix(np.real(y))
	b = Matrix(np.imag(y))

	e_J = symbols("e1:%d"%(nK+1))
	f_J = symbols("f1:%d"%(nK+1))
	hSluV = []

	if isinstance(inds["Pl"],np.ndarray):
		nPl = inds["Pl"].astype(int)
	elif isinstance(inds["Pl"],list):
		nPl = inds["Pl"]
	else:
		raise ValueError
	for k in range(len(nPl)):
		i,j = nPl[k,:]
		hSluV.append((e_J[i]**2+f_J[i]**2)*g[i,j]-(e_J[i]*e_J[j]+f_J[i]*f_J[j])*g[i,j]+(e_J[i]*f_J[j]-e_J[j]*f_J[i])*b[i,j])

	if isinstance(inds["Ql"],np.ndarray):
		nQl = inds["Ql"].astype(int)
	elif isinstance(inds["Ql"],list):
		nQl = inds["Ql"]
	else:
		raise ValueError
	for k in range(len(nQl)):
		i,j = nQl[k,:]
		hSluV.append(-(e_J[i]**2+f_J[i]**2)*b[i,j]+(e_J[i]*e_J[j]+f_J[i]*f_J[j])*b[i,j]+(e_J[i]*f_J[j]-e_J[j]*f_J[i])*g[i,j]-(e_J[i]**2+f_J[i]**2)*cap[i,j]/2)

	if isinstance(inds["Vm"],np.ndarray):
		nVk = inds["Vm"].astype(int)
	elif isinstance(inds["Vm"],list):
		nVk = inds["Vm"]
	else:
		raise ValueError
	for k in range(len(nVk)):
		i = nVk[k]
		hSluV.append((e_J[i]**2+f_J[i]**2)**0.5)
	hSluV = Matrix(hSluV)
	ef_J = e_J[1:] + f_J[1:]
	JMatrix_dHdV = hSluV.jacobian(ef_J)

	return JMatrix_dHdV, hSluV


def getV0(bus, type_initialguess=1, V0 = 1.0, gen = None):
	"""
	Get initial voltage profile for power flow calculation.
	Note: The pv bus voltage will remain at the given value even for flat start.
	type_initialguess: 1 - initial guess from case data
					   2 - flat start
					   3 - from input
	"""
	try:
		from pypower.idx_gen import GEN_STATUS, GEN_BUS, VG
		from pypower.idx_bus import VA, VM
	except ImportError:
		GEN_STATUS = 7
		GEN_BUS = 0
		VG = 5
		VM = 7
		VA = 8

	if type_initialguess == 1: # using previous value in case data
		# NOTE: angle is in degree in case data, but in radians in pf solver,
		# so conversion from degree to radians is needed here
		V0  = bus[:, VM] * np.exp(1j * np.pi/180 * bus[:, VA])
	elif type_initialguess == 2: # using flat start
		V0 = np.ones(bus.shape[0])
	elif type_initialguess == 3: # using given initial voltage
		V0 = V0
	else:
		raise ValueError("unknown type")
	if isinstance(gen,np.ndarray):
		# set the voltages of PV bus and reference bus into the initial guess	# generator info
		on = np.nonzero(gen[:, GEN_STATUS])[0]   # which generators are on?
		gbus = gen[on, GEN_BUS].astype(int)      # what buses are they at?
		V0[gbus] = gen[on, VG] / np.abs(V0[gbus]) * V0[gbus]
	return V0


def repair_meas(meas, meas_idx, meas_unc, expected_indices=None):
	"""
	The dictionaries meas, meas_idx and meas_unc are expected to contain certain keys.
	If any of these keys are missing, this function sets up corresponding default values.
	:param meas: dict containing measurement data
	:param meas_unc: dict containg uncertainty associated with measurement data
	:param meas_idx: dict containing index arrays indicating index of measurement in bus/branch array
	:return: meas, meas_idx, meas_unc
	"""
	if not isinstance(expected_indices,list):
		expected_indices = ['Pk','Qk','Vm','Va','Ql','Pl']

	for key in expected_indices:
		if key in meas:
			if not key in meas_idx: raise ValueError("Key %s not found in 'meas_idx'"%key)
			if not key in meas_unc:
				print "Uncertainty is not specified for measurements '%s'. Using sigma=1 instead."%key
				meas_unc[key] = np.ones_like(meas[key])*meas[key].max()*0.2
		else:
			meas[key] = np.array([])
			meas_idx[key] = []
			meas_unc[key] = np.array([])
	return meas, meas_idx, meas_unc


def voltage2power(V,topology):
	"""Calculation of nodal and line power from complex nodal voltages
	Returns dictionary of corresponding powers and a dictionary with the partial derivatives

	:param V: numpy array of complex nodal voltages
	:param topology: dict in pypower casefile format
	:returns: dict of calculated power values, dict of jacobians if V is one-dimensional

	"""
	from pypower.idx_brch import F_BUS,T_BUS
	from pypower.idx_gen import GEN_BUS
	from pypower.makeYbus import makeYbus
	from pypower.idx_bus import PD,QD
	from pypower.dSbus_dV import dSbus_dV
	from pypower.dSbr_dV import dSbr_dV

	assert(str(V.dtype)[:7]=="complex")

	list_f = topology["branch"][:,F_BUS].tolist()
	list_t = topology["branch"][:,T_BUS].tolist()

	Ybus,Yfrom,Yto = makeYbus(topology["baseMVA"],topology["bus"],topology["branch"])
	Ybus, Yfrom, Yto = Ybus.toarray(), Yfrom.toarray(), Yto.toarray()

	powers = {}
	if len(V.shape)==1:
		powers["Sf"] = V[list_f]*np.conj(np.dot(Yfrom,V))
		powers["St"] = V[list_t]*np.conj(np.dot(Yto,V))
		gbus = topology["gen"][:,GEN_BUS].astype(int)
		Sgbus= V[gbus]*np.conj(np.dot(Ybus[gbus,:],V))
		powers["Sg"] = ( Sgbus*topology["baseMVA"] + topology["bus"][gbus,PD] + 1j*topology["bus"][gbus,QD] ) / topology["baseMVA"]
		powers["Sk"] = V*np.conj(np.dot(Ybus,V))
	else:
		powers = dict([(name,[]) for name in ["Sf","St","Sg","Sk"]])
		for k in range(V.shape[0]):
			powers["Sf"].append(V[k,list_f]*np.conj(np.dot(Yfrom,V[k,:])))
			powers["St"].append(V[k,list_t]*np.conj(np.dot(Yto,V[k,:])))
			gbus = topology["gen"][:,GEN_BUS].astype(int)
			Sgbus= V[k,gbus]*np.conj(np.dot(Ybus[gbus,:],V[k,:]))
			powers["Sg"].append(( Sgbus*topology["baseMVA"] + topology["bus"][gbus,PD] + 1j*topology["bus"][gbus,QD] ) / topology["baseMVA"])
			powers["Sk"].append(V[k,:]*np.conj(np.dot(Ybus,V[k,:])))
		for key in powers.keys():
			powers[key] = np.asarray(powers[key])
		# calculate partial derivative
	jacobians = dict([])
	if len(V.shape)==1:
		jacobians["dSbus_dVm"], jacobians["dSbus_dVa"] = dSbus_dV(Ybus,V)
		jacobians["dSf_dVa"], jacobians["dSf_dVm"], jacobians["dSt_dVa"], \
			jacobians["dSt_dVm"], jacobians["Sf"], jacobians["St"] = dSbr_dV(topology["branch"],Yfrom,Yto,V)

	return powers, jacobians

def network_observability(topology, meas_idx, V):
	from pypower.api import makeYbus
	from pypower.bustypes import bustypes
	from pypower.dSbus_dV import dSbus_dV
	from pypower.dSbr_dV import dSbr_dV

	# build admittances
	Ybus,Yfrom,Yto = makeYbus(topology["baseMVA"],topology["bus"],topology["branch"])
	Ybus, Yfrom, Yto = Ybus.toarray(), Yfrom.toarray(), Yto.toarray()
	Nk = Ybus.shape[0]

	# get non-reference buses
	ref,pv,pq = bustypes(topology["bus"],topology["gen"])
	nonref = np.r_[pv,pq]
	gbus = topology["gen"][:,GEN_BUS].astype(int)

	# calculate partial derivative
	dSbus_dVm, dSbus_dVa = dSbus_dV(Ybus,V)
	dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St = dSbr_dV(topology["branch"],Yfrom,Yto,V)
	# create Jacobian matrix
	## submatrices related to line flow
	dPf_Va = np.real(dSf_dVa)[meas_idx["Pf"],:][:,nonref]
	dPf_Vm = np.real(dSf_dVm)[meas_idx["Pf"],:][:,nonref]
	dPt_Va = np.real(dSt_dVa)[meas_idx["Pt"],:][:,nonref]
	dPt_Vm = np.real(dSt_dVm)[meas_idx["Pt"],:][:,nonref]
	dQf_Va = np.imag(dSf_dVa)[meas_idx["Qf"],:][:,nonref]
	dQf_Vm = np.imag(dSf_dVm)[meas_idx["Qf"],:][:,nonref]
	dQt_Va = np.imag(dSt_dVa)[meas_idx["Qt"],:][:,nonref]
	dQt_Vm = np.imag(dSt_dVm)[meas_idx["Qt"],:][:,nonref]
	## submatrix related to generator output
	dPg_Va = np.real(dSbus_dVa)[gbus,:][meas_idx["Pg"],:][:,nonref]
	dPg_Vm = np.real(dSbus_dVm)[gbus,:][meas_idx["Pg"],:][:,nonref]
	dQg_Va = np.imag(dSbus_dVa)[gbus,:][meas_idx["Qg"],:][:,nonref]
	dQg_Vm = np.imag(dSbus_dVm)[gbus,:][meas_idx["Qg"],:][:,nonref]
	## submatrix related to bus injection
	dPk_Va = np.real(dSbus_dVa)[meas_idx["Pk"],:][:,nonref]
	dPk_Vm = np.real(dSbus_dVm)[meas_idx["Pk"],:][:,nonref]
	dQk_Va = np.imag(dSbus_dVa)[meas_idx["Qk"],:][:,nonref]
	dQk_Vm = np.imag(dSbus_dVm)[meas_idx["Qk"],:][:,nonref]
	## submatrix related to voltage angle
	dVa_Va = np.eye(Nk)[meas_idx["Va"],:][:,nonref]
	dVa_Vm = np.zeros((Nk,Nk))[meas_idx["Va"],:][:,nonref]
	## submatrix related to voltage magnitude
	dVm_Va = np.zeros((Nk,Nk))[meas_idx["Vm"],:][:,nonref]
	dVm_Vm = np.eye(Nk)[meas_idx["Vm"],:][:,nonref]

	H = np.r_[np.c_[dPf_Va, dPf_Vm],\
			  np.c_[dPt_Va, dPt_Vm],\
			  np.c_[dPg_Va, dPg_Vm],\
			  np.c_[dQf_Va, dQf_Vm],\
			  np.c_[dQt_Va, dQt_Vm],\
			  np.c_[dQg_Va, dQg_Vm],\
			  np.c_[dPk_Va, dPk_Vm],\
			  np.c_[dQk_Va, dQk_Vm],\
			  np.c_[dVa_Va, dVa_Vm],\
			  np.c_[dVm_Va, dVm_Vm]]

	return isobservable(H,pv,pq)



def isobservable(H, pv, pq,tol=1e-5,check_reason=True):
	"""ISOBSERVABLE  Test for observability.
	   returns 1 if the system is observable, 0 otherwise.
	   created by Rui Bo on Jan 9, 2010
	   MATPOWER
	   $Id: isobservable.m,v 1.3 2010/04/26 19:45:26 ray Exp $
	   by Rui Bo
	   Copyright (c) 2009-2010 by Rui Bo

	   This file is part of MATPOWER.
	   See http://www.pserc.cornell.edu/matpower/ for more info.

	   MATPOWER is free software: you can redistribute it and/or modify
	   it under the terms of the GNU General Public License as published
	   by the Free Software Foundation, either version 3 of the License,
	   or (at your option) any later version.

	   MATPOWER is distributed in the hope that it will be useful,
	   but WITHOUT ANY WARRANTY; without even the implied warranty of
	   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	   GNU General Public License for more details.

	   You should have received a copy of the GNU General Public License
	   along with MATPOWER. If not, see <http://www.gnu.org/licenses/>.

	   Additional permission under GNU GPL version 3 section 7

	   If you modify MATPOWER, or any covered work, to interface with
	   other modules (such as MATLAB code and MEX-files) available in a
	   MATLAB(R) or comparable environment containing parts covered
	   under other licensing terms, the licensors of MATPOWER grant
	   you additional permission to convey the resulting work.
	"""
# options

# test if H is full rank
	m, n = H.shape
	r    = np.linalg.matrix_rank(H)
	if r < min(m, n):
		TorF = False
	else:
		TorF = True

#% look for reasons for system being not observable
	if check_reason and not TorF:
# look for variables not being observed
		idx_trivialColumns = []
		varNames = []
		for j in range(n):
			normJ = np.linalg.norm(H[:, j], ord=np.inf)
			if normJ < tol:      # found a zero column:
				idx_trivialColumns.append(j)
				varName = getVarName(j, pv, pq)
				varNames.append(varName)
		if not len(idx_trivialColumns)==0: # found zero-valued column vector
			print 'Warning: The following variables are not observable since they are not related with any measurement!'
			print "var name",
			print varNames
			print "var column",
			print idx_trivialColumns
		else: # no zero-valued column vector
	# look for dependent column vectors
			for j in range(n):
				rr = np.linalg.matrix_rank(H[:, :j+1])
				if rr != j+1: # found dependent column vector
			# look for linearly depedent vector
					colJ = H[:, j] # j(th) column of H
					varJName = getVarName(j, pv, pq)
					for k in range(j-1):
						colK = H[:, k]
						if np.linalg.matrix_rank(np.c_[colK,colJ]) < 2: # k(th) column vector is linearly dependent of j(th) column vector
							varKName = getVarName(k, pv, pq)
							print 'Warning: %d(th) column vector (w.r.t. %s) of H is linearly dependent of %d(th) column vector (w.r.t. %s)!'%(j, varJName, k, varKName)
							return TorF
			print 'Warning: No specific reason was found for system being not observable.'
	return TorF

def getVarName(varIndex, pv, pq):
	"""GETVARNAME  Get variable name by variable index (as in H matrix).
		[OUTPUT PARAMETERS]
		varName: comprise both variable type ('Va', 'Vm') and the bus number of
		the variable. For instance, Va8, Vm10, etc.
		created by Rui Bo on Jan 9, 2010

		MATPOWER
		$Id: getVarName.m 1635 2010-04-26 19:45:26Z ray $
		by Rui Bo
		Copyright (c) 2009-2010 by Rui Bo

		This file is part of MATPOWER.
		See http://www.pserc.cornell.edu/matpower/ for more info.

		MATPOWER is free software: you can redistribute it and/or modify
		it under the terms of the GNU General Public License as published
		by the Free Software Foundation, either version 3 of the License,
		or (at your option) any later version.

		MATPOWER is distributed in the hope that it will be useful,
		but WITHOUT ANY WARRANTY; without even the implied warranty of
		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
		GNU General Public License for more details.

		You should have received a copy of the GNU General Public License
		along with MATPOWER. If not, see <http://www.gnu.org/licenses/>.

		Additional permission under GNU GPL version 3 section 7

		If you modify MATPOWER, or any covered work, to interface with
		other modules (such as MATLAB code and MEX-files) available in a
		MATLAB(R) or comparable environment containing parts covered
		under other licensing terms, the licensors of MATPOWER grant
		you additional permission to convey the resulting work.
		"""
# get non reference buses
	nonref = np.r_[pv, pq] + 1

	if varIndex < len(nonref):
		varType = 'Va'
		newIdx = varIndex
	else:
		varType = 'Vm'
		newIdx = varIndex - len(nonref)

	varName = '%s%d'%(varType, nonref[newIdx])
	return varName

