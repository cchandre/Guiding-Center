## Definition of the parameters for GC2Ds
##
## M: integer; number of modes
## A: float; amplitude of the electrostatic potential
## FLR: boolean array of length 2; Finite Larmor Radius effects for GC order 1 and order 2
## flr_order: array of length 2; 'all' or integer; FLR order for each GC order
## rho: float; Larmor Radius
## gc_order: 1 or 2
## eta: float; amplitude of the GC order 2 potential
## method: string; diffusion', 'poincare'
## modulo: boolean; only for method='poincare'; if True, x and y are taken modulo 2*pi
## Ntraj: integer; number of trajectories to be integrated
## Tf: integer; number of periods for the integration of the traejctories
## timestep: float; time step used by the integrator
## save_results: boolean; if True, the results are saved in a .mat file
## plot_results: boolean; if True, the results are plotted right after the computation

array = [0.4, 0.5, 0.6]
iteratable = 'A'

dict_list = [{'M': 5} for _ in range(len(array))]
for it in range(len(array)):
	dict_list[it].update({iteratable: array[it]})

for dict in dict_list:
	dict.update({
		'FLR': [False, False],
    	'flr_order': ['all', 'all'],
    	'rho': 0.7,
    	'gc_order': 1,
    	'eta': 0.1})

for dict in dict_list:
	dict.update({
    	#'method': 'poincare',
    	'method': 'diffusion',
    	'modulo': True,
    	'Ntraj': 1000,
	    'Tf': 1000,
    	'timestep': 0.03,
    	'save_results': False,
    	'plot_results': False})
