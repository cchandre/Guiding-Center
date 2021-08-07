import numpy as xp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import linregress
from scipy.io import savemat
import time
from datetime import date


def run_method(case):
	print('\033[92m    {} \033[00m'.format(case.__str__()))
	print('\033[92m    A = {:.2f}   rho = {:.2f}   eta = {:.2f} \033[00m'.format(case.A, case.rho, case.eta))
	filestr = 'A{:.2f}_RHO{:.2f}'.format(case.A, case.rho).replace('.', '')
	if case.gc_order == 2:
		filestr += '_ETA{:.2f}'.format(case.eta).replace('.', '')
	if case.method == 'plot_potentials':
		data = xp.array([case.phi, case.phi_, case.phi_gc2_0, case.phi_gc2_2])
		save_data(case, 'potentials', data, filestr)
		if case.plot_results:
			plt.figure(figsize=(8, 8))
			plt.pcolor(case.xv, case.xv, case.phi.imag, shading='auto')
			plt.colorbar()
			plt.figure(figsize=(8, 8))
			plt.pcolor(case.xv, case.xv, case.phi_.imag, shading='auto')
			plt.colorbar()
			plt.figure(figsize=(8, 8))
			plt.pcolor(case.xv, case.xv, case.phi_gc2_0 - case.phi_gc2_2.real, shading='auto')
			plt.colorbar()
			plt.show()
	elif case.method == 'poincare':
		y0 = 2.0 * xp.pi * xp.random.rand(2 * case.Ntraj)
		t_eval = 2.0 * xp.pi * xp.arange(0, case.Tf)
		start = time.time()
		sol = solve_ivp(case.eqn_phi, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.timestep, atol=1, rtol=1)
		print('\033[92m    Computation finished in {} seconds \033[00m'.format(int(time.time() - start)))
		if case.modulo:
			sol.y = sol.y % (2.0 * xp.pi)
		save_data(case, 'poincare', xp.array([sol.y[:case.Ntraj, :], sol.y[case.Ntraj:, :]]).transpose(), filestr)
		if case.plot_results:
			plt.figure(figsize=(8, 8))
			plt.plot(sol.y[:case.Ntraj, :], sol.y[case.Ntraj:, :], 'b.', markersize=2)
			plt.show()
	elif case.method == 'diffusion':
		y0 = 2.0 * xp.pi * xp.random.rand(2 * case.Ntraj)
		t_eval = 2.0 * xp.pi * xp.arange(0, case.Tf)
		start = time.time()
		sol = solve_ivp(case.eqn_phi, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.timestep, atol=1, rtol=1)
		print('\033[92m    Computation finished in {} seconds \033[00m'.format(int(time.time() - start)))
		r2 = xp.zeros(case.Tf)
		for t in range(case.Tf):
			r2[t] += (xp.abs(sol.y[:, t:] - sol.y[:, :case.Tf-t]) ** 2).sum() / (case.Ntraj * (case.Tf - t))
		diff_data = linregress(t_eval[case.Tf//8:7*case.Tf//8], r2[case.Tf//8:7*case.Tf//8])
		max_y = (xp.abs(sol.y[:case.Ntraj, :] - sol.y[:case.Ntraj, 0].reshape(case.Ntraj,1)) ** 2\
		+ xp.abs(sol.y[case.Ntraj:, :] - sol.y[case.Ntraj:, 0].reshape(case.Ntraj,1)) ** 2).max(axis=1)
		trapped = (max_y <= 3.0 * xp.pi).sum()
		save_data(case, 'diffusion', [trapped, diff_data.slope, diff_data.rvalue**2], filestr, info='trapped particles / diffusion coefficient / R2')
		print('\033[96m          trapped particles = {} \033[00m'.format(trapped))
		print('\033[96m          diffusion coefficient = {:.6f} \033[00m'.format(diff_data.slope))
		print('\033[96m                     with an R2 = {:.6f} \033[00m'.format(diff_data.rvalue**2))
		if case.plot_results:
			plt.figure(figsize=(8, 8))
			plt.plot(t_eval, r2, 'b', linewidth=2)
			plt.show()


def save_data(case, name, data, filestr, info=[]):
	if case.save_results:
		mdic = case.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		date_today = date.today().strftime(" %B %d, %Y\n")
		mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
		name_file = type(case).__name__ + '_' + name + '_' + filestr + '.mat'
		savemat(name_file, mdic)
		print('\033[92m    Results saved in {} \033[00m'.format(name_file))
