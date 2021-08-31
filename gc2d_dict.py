########################################################################################################################
##                                   Definition of the parameters for GC2D                                            ##
########################################################################################################################
##                                                                                                                    ##
##   Potential: string; 'KMdCN' or 'turbulent'                                                                        ##
##   Method: string; 'plot_potentials' (only for 'turbulent'), 'diffusion', 'poincare'                                ##
##   FLR: array of length 2; 'none', 'all' or integer; FLR order for each GC order                                    ##
##   GCorder: 1 or 2; order in the guiding-center expansion                                                           ##
##   A: float; amplitude of the electrostatic potential                                                               ##
##   rho: float; value of the Larmor radius                                                                           ##
##   eta: float; amplitude of the GC order 2 potential                                                                ##
##   M: integer; number of modes (default = 5 for 'KMdCN' and 25 for 'turbulent')                                     ##
##   Ntraj: integer; number of trajectories to be integrated                                                          ##
##   Tf: integer; number of periods for the integration of the trajectories                                           ##
##   init: boolean; 'random' or 'fixed'                                                                               ##
##   modulo: boolean; only for Method='poincare'; if True, x and y are taken modulo 2*pi                              ##
##   N: integer; number of points on each axis for 'turbulent' (default = 2 ** 10)                                    ##
##   TimeStep: float; time step used by the integrator                                                                ##
##   SaveData: boolean; if True, the results are saved in a .mat file                                                 ##
##   PlotResults: boolean; if True, the results are plotted right after the computation                               ##
##   Parallelization: tuple (boolean, int); True for parallelization, int is the number of cores to be used           ##
##                                                                                                                    ##
########################################################################################################################
import numpy as xp

Potential = 'turbulent'
Method = 'plot_potentials'

FLR = ['all', 'all']
GCorder = 2

iterable_name = 'rho'
iterable_values = xp.linspace(0.0, 1.0, 3)
A = 0.6
#rho = 0.7
eta = 0.1

Ntraj = 100
Tf = 500
init = 'fixed'
modulo = True
TimeStep = 0.03
SaveData = False
PlotResults = True
Parallelization = (True, 3)

########################################################################################################################
##                                                DO NOT EDIT BELOW                                                   ##
########################################################################################################################
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
		'Ntraj': Ntraj,
		'Tf': Tf,
		'init': init,
		'TimeStep': TimeStep,
		'SaveData': SaveData,
		'PlotResults': PlotResults})
########################################################################################################################
