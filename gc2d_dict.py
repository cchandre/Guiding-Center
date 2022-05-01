###################################################################################################
##            Dictionary of parameters: https://github.com/cchandre/Guiding-Center               ##
###################################################################################################

import numpy as xp

Potential = 'turbulent'
Method = 'potentials'

FLR = ('all', 'all')

A = 0.75
rho = 0.1
eta = 0.1

Ntraj = 8192
Tf = 20000
threshold = 4
TwoStepIntegration = True
Tmid = 1000
TimeStep = 0.05
init = 'random'
modulo = False
grid = False

SaveData = False
PlotResults = True
Parallelization = (False, 50)

dpi = 300
darkmode = True

###################################################################################################
##                             DO NOT EDIT BELOW                                                 ##
###################################################################################################
if FLR[0] == 'none' and FLR[1] == 'none':
	rho = 0
if xp.all((rho == 0)):
	FLR = ('none', 'none')
if xp.all((eta == 0)):
	GCorder = 1
else:
	GCorder = 2
val_params = xp.meshgrid(A, rho, eta, indexing='ij')
num_dict = len(val_params[0].flatten())
dict_list = [{'Potential': Potential} for _ in range(num_dict)]
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
		'SaveData': SaveData,
		'PlotResults': PlotResults,
		'dpi': dpi,
		'darkmode': darkmode})
	if Potential == 'KMdCN':
		dict.setdefault('M', 5)
	elif Potential == 'turbulent':
		dict.setdefault('M', 25)
		dict.setdefault('N', 2**10)
###################################################################################################
