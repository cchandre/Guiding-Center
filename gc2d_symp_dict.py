###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/Guiding-Center            ##
###################################################################################################

import numpy as xp

A = 0.7

Ntraj = 10
Tf = 500
TimeStep = 5e-2  # recommended values (gc: 5e-3, fo: 5e-4)
init = 'fixed'

SaveData = False

M = 25
N = 2**10

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################
num_dict = len(val_params[0].flatten())

dict.update({
	'A': A,
	'Ntraj': Ntraj,
	'Tf': Tf,
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
