###################################################################################################
##               Dictionary of parameters: https://github.com/cchandre/Guiding-Center            ##
###################################################################################################

A = 0.7

Ntraj = 5
Tf = 50
TimeStep = 1e-3  # recommended value: 5e-3
init = 'fixed'
solve_method = 'symp'
ode_solver = 'Verlet'
omega = 10

SaveData = False

M = 25
N = 2**12

###################################################################################################
##                              DO NOT EDIT BELOW                                                ##
###################################################################################################
dictparams = {
	'A': A,
	'Ntraj': Ntraj,
	'Tf': Tf,
	'init': init,
    'solve_method': solve_method,
    'ode_solver': ode_solver,
    'omega': omega,
	'TimeStep': TimeStep,
	'SaveData': SaveData,
	'M': M,
	'N': N}
###################################################################################################
