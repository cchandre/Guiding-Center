########################################################################################################################
##                                   Definition of the parameters for GC2D
########################################################################################################################
##
## potential: string; 'KMdCN' for Kryukov-Martinell-delCastilloNegrete, 'turbulent' for the turbulent potential
## method: string; 'plot_potentials' (only for 'turbulent'), 'diffusion', 'poincare'
## flr: array of length 2; 'none', 'all' or integer; FLR order for each GC order
## gc_order: 1 or 2; GC order
## A: float; amplitude of the electrostatic potential
## rho: float; value of the Larmor radius
## eta: float; amplitude of the GC order 2 potential
## M: integer; number of modes (default = 5 for 'KMdCN' and 25 for 'turbulent')
## N: integer; number of points on each axis for 'turbulent' (default = 2 ** 10)
## Ntraj: integer; number of trajectories to be integrated
## Tf: integer; number of periods for the integration of the trajectories
## modulo: boolean; only for method='poincare'; if True, x and y are taken modulo 2*pi
## timestep: float; time step used by the integrator
## save_results: boolean; if True, the results are saved in a .mat file
## plot_results: boolean; if True, the results are plotted right after the computation
##
########################################################################################################################

potential = 'KMdCN'
method = 'diffusion'

flr = ['none', 'all']
gc_order = 2

vec_iteratable = [0.4, 0.5, 0.6]
iteratable = 'A'
#A = 0.5
rho = 0.7
eta = 0.1

Ntraj = 1000
Tf = 1000
modulo = False
timestep = 0.03
save_results = True
plot_results = False

########################################################################################################################
dict_list = [{'potential': potential} for _ in range(len(vec_iteratable))]
for it in range(len(vec_iteratable)):
	dict_list[it].update({iteratable: vec_iteratable[it]})

if flr[0] == 'none' and flr[1] == 'none':
	rho = 0
if gc_order == 1:
	eta = 0

for dict in dict_list:
	if potential == 'KMdCN':
		dict.update({'M': 5})
	elif potential == 'turbulent':
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
    	'flr': flr,
    	'gc_order': gc_order,
		'method': method,
    	'modulo': modulo,
    	'Ntraj': Ntraj,
	    'Tf': Tf,
    	'timestep': timestep,
    	'save_results': save_results,
    	'plot_results': plot_results})
########################################################################################################################
