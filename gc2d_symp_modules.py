#
# BSD 2-Clause License
#
# Copyright (c) 2021, Cristel Chandre
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as xp
from scipy.integrate import solve_ivp
from scipy.io import savemat
import time
from datetime import date

def run_method(case):
	print(f"\033[92m   Integration of {case.__str__()} using {'symplectic integrator' if case.solve_method=='symp' else 'linear interpolation'} \033[00m")
	t_eval = 2 * xp.pi * xp.arange(0, case.Tf + 1)
	if case.init == 'random':
		y0 = 2 * xp.pi * xp.random.rand(2 * case.Ntraj)
	elif case.init == 'fixed':
		y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(case.Ntraj)), endpoint=False)
		y_mat = xp.meshgrid(y_vec, y_vec)
		y0 = xp.concatenate((y_mat[0], y_mat[1]), axis=None)
		case.Ntraj = int(xp.sqrt(case.Ntraj))**2
	y0 = xp.concatenate((xp.zeros(case.Ntraj), y0, xp.zeros(case.Ntraj)), axis=None)
	start = time.time()
	if case.solve_method == 'interp':
		sol = solve_ivp(case.eqn_interp, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.TimeStep, atol=1, rtol=1).y
	elif case.solve_method == 'symp':
		sol = case.integr_e(t_eval, y0)
	print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
	energy = case.compute_energy(sol)
	err_energy = xp.abs(energy - energy[:, 0][:, xp.newaxis])
	print(f'\033[90m           with error in energy = {xp.max(err_energy)}')
	save_data(case, sol, 'sol_' + case.solve_method)

def save_data(case, data, filestr, info=[]):
	if case.SaveData:
		x, y = xp.split(data, 4)[1:3]
		mdic = case.DictParams.copy()
		mdic.update({'x': x, 'y': y, 'info': info})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print(f'\033[90m        Results saved in {filestr}.mat \033[00m')