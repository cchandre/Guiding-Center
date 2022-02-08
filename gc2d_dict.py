###################################################################################################
##            Dictionary of parameters: https://github.com/cchandre/Guiding-Center               ##
###################################################################################################

import numpy as xp

Potential = 'turbulent'
Method = 'diffusion'

FLR = ('all', 'all')
GCorder = 1

A = 0.7
rho = xp.linspace(0, 0.3, 2)
eta = 0.0

Ntraj = 100
Tf = 200
TwoStepIntegration = True
Tmid = 100
TimeStep = 0.05
init = 'random'
modulo = False
threshold = 4
grid = False

SaveData = False
PlotResults = True
Parallelization = (False, 34)

darkmode = True

###################################################################################################
##                             DO NOT EDIT BELOW                                                 ##
###################################################################################################
if FLR[0] == 'none' and FLR[1] == 'none':
	rho = 0
if GCorder == 1:
	eta = 0
val_params = xp.meshgrid(A, rho, eta, indexing='ij')
num_dict = len(val_params[0].flatten())
dict_list = [{'Potential': Potential} for _ in range(num_dict)]
for _ in range(num_dict):
	dict_list[_].update({
		'A': val_params[0].flatten()[_],
		'rho': val_params[1].flatten()[_],
		'eta': val_params[2].flatten()[_]})
for dict in dict_list:
	if Potential == 'KMdCN':
		dict.update({'M': 5})
	elif Potential == 'turbulent':
		dict.update({
			'M': 25,
			'N': 2**10})
	dict.update({
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
		'darkmode': darkmode})
###################################################################################################
