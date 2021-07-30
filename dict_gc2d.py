## Definition of the parameters for GC2D
##
## M: integer; number of modes
## N: integer; number of points in each direction of the 2D grid
## A: float; amplitude of the electrostatic potential
## FLR: boolean array of length 2; Finite Larmor Radius effects for GC order 1 and order 2
## flr_order: array of length 2; 'all' or integer; FLR order for each GC order
## rho: float; Larmor Radius
## gc_order: 1 or 2
## eta: float; amplitude of the GC order 2 potential
## method: string; 'plot_potentials', 'diffusion', 'poincare'
## modulo: boolean; only for method='poincare'; if True, x and y are taken modulo 2*pi
## Ntraj: integer; number of trajectories to be integrated
## Tf: integer; number of periods for the integration of the traejctories
## timestep: float; time step used by the integrator
## save_results: boolean; if True, the results are saved in a .mat file
## plot_results: boolean; if True, the results are plotted right after the computation

array = [0.01, 0.03, 0.05, 0.07, 0.09]
iteratable = 'rho'

dict_list = [{'M': 25} for _ in range(len(array))]
for it in range(len(array)):
	dict_list[it].update({iteratable: array[it]})

for dict in dict_list:
	dict.update({
		'N': 2 ** 10, 
		'A': 1.0,
#		'rho': 0.7,
		'FLR': [True, False],
    	'flr_order': ['all', 'all'],
    	'gc_order': 1,
    	'eta': 0.0})

for dict in dict_list:
	dict.update({
    	#'method': 'poincare',
    	'method': 'diffusion',
    	'modulo': False,
    	'Ntraj': 2000,
	    'Tf': 5000,
    	'timestep': 0.05,
    	'save_results': True,
    	'plot_results': False})
