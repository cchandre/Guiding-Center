###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/Guiding-Center            ##
###################################################################################################

import numpy as xp

Potential = 'turbulent'
Method = 'diffusion_gc'

FLR = ('all', 'all')

A = 0.7
rho = xp.linspace(0, 0.3, 128)
eta = xp.linspace(-0.2, 0.3, 128)

Ntraj = 1024
Tf = 5000
threshold = 4
TwoStepIntegration = True
Tmid = 1500
TimeStep = 5e-2  # recommended values (gc: 5e-3, fo: 5e-4)
check_energy = False
init = 'fixed'

SaveData = False
PlotResults = False
Parallelization = (True, 100)

modulo = False
grid = False 
darkmode = True
fig_extension = '.pdf'

M = 25 if Potential == 'turbulent' else 5
N = 2**10

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
		'SaveData': SaveData,
		'PlotResults': PlotResults,
		'dpi': 200 if Method == 'potentials' else 800,
		'darkmode': darkmode,
		'extension': fig_extension,
		'M': M,
		'N': N})
###################################################################################################
