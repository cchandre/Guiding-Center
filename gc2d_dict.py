###################################################################################################
##            Dictionary of parameters: https://github.com/cchandre/Guiding-Center               ##
###################################################################################################

import numpy as xp

Potential = 'turbulent'
Method = 'diffusion'

FLR = ('all', 'all')
GCorder = 1

iterable_name = 'rho'
iterable_values = xp.linspace(0, 0, 1)
A = 0.6
#rho = 0.7
eta = 0.0

Ntraj = 100
Tf = 2000
TwoStepIntegration = True
Tmid = 100
TimeStep = 0.05
init = 'random'
modulo = False
threshold = 4
grid = False

SaveData = False
PlotResults = True
Parallelization = (False, 2)

darkmode = True

###################################################################################################
##                             DO NOT EDIT BELOW                                                 ##
###################################################################################################
dict_list = [{'Potential': Potential} for _ in range(len(iterable_values))]
for it in range(len(iterable_values)):
	dict_list[it].update({iterable_name: iterable_values[it]})

if FLR[0] == 'none' and FLR[1] == 'none':
	rho = 0
if GCorder == 1:
	eta = 0

for dict in dict_list:
	if Potential == 'KMdCN':
		dict.update({'M': 5})
	elif Potential == 'turbulent':
		dict.update({
			'M': 25,
			'N': 2 ** 10})
	if not 'A' in dict:
		dict.update({'A': A})
	if not 'rho' in dict:
		dict.update({'rho': rho})
	if not 'eta' in dict:
		dict.update({'eta': eta})
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
