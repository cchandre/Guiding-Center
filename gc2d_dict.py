###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/Guiding-Center            ##
###################################################################################################

import numpy as xp

Potential = 'turbulent'
Method = 'diffusion_gc'

FLR = ('all', 'all')

A = 0.7
rho = 0.2
eta = 0.05

Ntraj = 100
Tf = 500
threshold = 4
TwoStepIntegration = True
Tmid = 100
TimeStep = 5e-2  # recommended values (gc: 5e-3, ions: 5e-4)
check_energy = False
check_mu = False
init = 'fixed'
modulo = False
grid = False

SaveData = False
PlotResults = True
Parallelization = (False, 4)

dpi = 800
darkmode = True

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################
if (FLR[0] == 'none') and (FLR[1] == 'none'):
	rho = 0
if xp.all((rho == 0)):
	FLR = ('none', 'none')
if xp.all((eta == 0)):
	GCorder = 1
else:
	GCorder = 2
if Method.endswith('_gc'):
	check_mu = False
val_params = xp.meshgrid(A, rho, eta, indexing='ij')
num_dict = len(val_params[0].flatten())
dict_list = [{'Potential': Potential} for _ in range(num_dict)]
if Tmid >= Tf:
	Tmid = Tf // 2 if TwoStepIntegration else Tf
for _, dict in enumerate(dict_list):
	dict.update({
		'A': val_params[0].flatten()[_],
		'rho': val_params[1].flatten()[_],
		'eta': val_params[2].flatten()[_],
		'FLR': FLR,
		'GCorder': GCorder,
		'Method': Method,
		'modulo': modulo,
		'threshold': threshold,
		'grid': grid,
		'Ntraj': Ntraj,
		'Tf': Tf,
		'TwoStepIntegration': TwoStepIntegration,
		'Tmid': Tmid,
		'init': init,
		'TimeStep': TimeStep,
		'check_energy': check_energy,
		'check_mu': check_mu,
		'SaveData': SaveData,
		'PlotResults': PlotResults,
		'dpi': dpi,
		'darkmode': darkmode})
	if Potential == 'KMdCN':
		dict.update({'M': 5})
	elif Potential == 'turbulent':
		dict.update({'M': 25, 'N': 2**10})
###################################################################################################
