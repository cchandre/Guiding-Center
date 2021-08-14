import numpy as xp
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
from matplotlib import colors
from scipy.integrate import solve_ivp
from scipy.stats import linregress
from scipy.io import savemat
import time
from datetime import date


def run_method(case):
	print('\033[92m    {} \033[00m'.format(case.__str__()))
	print('\033[92m    A = {:.2f}   rho = {:.2f}   eta = {:.2f} \033[00m'.format(case.A, case.rho, case.eta))
	filestr = 'A{:.2f}_RHO{:.2f}'.format(case.A, case.rho).replace('.', '')
	plt.rcParams.update({
		'text.usetex': True,
		'font.family': 'serif',
		'font.sans-serif': ['Palatino'],
		'font.size': 24,
		'figure.figsize': [25, 8],
		'image.cmap': 'bwr'})
	if case.gc_order == 2:
		filestr += '_ETA{:.2f}'.format(case.eta).replace('.', '')
	if case.method == 'plot_potentials' and case.potential == 'turbulent':
		data = xp.array([case.phi, case.phi_gc1_1, case.phi_gc2_0, case.phi_gc2_2])
		save_data(case, 'potentials', data, filestr)
		trange = xp.linspace(0, 2 * xp.pi, 50)
		min_phi = (case.phi[:, :, xp.newaxis] * xp.exp(-1j * trange[xp.newaxis, xp.newaxis, :])).imag.min()
		max_phi = (case.phi[:, :, xp.newaxis] * xp.exp(-1j * trange[xp.newaxis, xp.newaxis, :])).imag.max()
		min_phi_gc1 = (case.phi_gc1_1[:, :, xp.newaxis] * xp.exp(-1j * trange[xp.newaxis, xp.newaxis, :])).imag.min()
		max_phi_gc1 = (case.phi_gc1_1[:, :, xp.newaxis] * xp.exp(-1j * trange[xp.newaxis, xp.newaxis, :])).imag.max()
		vmin = min(min_phi, min_phi_gc1)
		vmax = max(max_phi, max_phi_gc1)
		extent = (0, 2 * xp.pi, 0, 2 * xp.pi)
		divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
		fig, axs = plt.subplots(1, case.gc_order+1)
		ims = []
		for t in trange:
			im = [axs[0].imshow((data[0] * xp.exp(-1j * t)).imag, origin='lower', extent=extent, animated=True, norm=divnorm), axs[1].imshow((data[1] * xp.exp(-1j * t)).imag, origin='lower', extent=extent, animated=True, norm=divnorm)]
			if case.gc_order == 2:
				im.append(axs[2].imshow((data[2] - data[3] * xp.exp(-2j * t)).real, origin='lower', extent=extent, animated=True, norm=divnorm))
			for ax in axs:
				ax.set_xlabel('$x$', fontsize=30)
				ax.set_ylabel('$y$', fontsize=30)
				ax.set_xticks([0, 2, 4, 6])
				ax.set_yticks([0, 2, 4, 6])
			ims.append(im)
		fig.colorbar(im[case.gc_order], ax=axs.ravel().tolist())
		ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
		ani.save(filestr + '.gif', writer=PillowWriter(fps=30))
	elif case.method in ['poincare', 'diffusion']:
		if case.init == 'random':
			y0 = 2.0 * xp.pi * xp.random.rand(2 * case.Ntraj)
		elif case.init == 'fixed':
			y_vec = xp.linspace(0.0, 2.0 * xp.pi, int(xp.sqrt(case.Ntraj)), endpoint=False)
			y_mat = xp.meshgrid(y_vec, y_vec)
			y0 = xp.concatenate((y_mat[0].flatten(), y_mat[1].flatten()))
			case.Ntraj = int(xp.sqrt(case.Ntraj)) ** 2
		t_eval = 2.0 * xp.pi * xp.arange(0, case.Tf)
		start = time.time()
		sol = solve_ivp(case.eqn_phi, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.timestep, atol=1, rtol=1)
		print('\033[92m    Computation finished in {} seconds \033[00m'.format(int(time.time() - start)))
		if case.method == 'poincare':
			if case.modulo:
				sol.y = sol.y % (2.0 * xp.pi)
			save_data(case, 'poincare', xp.array([sol.y[:case.Ntraj, :], sol.y[case.Ntraj:, :]]).transpose(), filestr)
			if case.plot_results:
				plt.figure(figsize=(8, 8))
				plt.plot(sol.y[:case.Ntraj, :], sol.y[case.Ntraj:, :], 'b.', markersize=2)
				plt.pause(0.5)
		if case.method == 'diffusion':
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
				plt.pause(0.5)


def save_data(case, name, data, filestr, info=[]):
	if case.save_results:
		mdic = case.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		date_today = date.today().strftime(" %B %d, %Y\n")
		mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
		name_file = type(case).__name__ + '_' + name + '_' + filestr + '.mat'
		savemat(name_file, mdic)
		print('\033[92m    Results saved in {} \033[00m'.format(name_file))
