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
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
from matplotlib import colors
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.metrics import r2_score
from scipy.io import savemat
import time
from datetime import date
import os

def run_method(case):
	if case.darkmode:
		cs = ['k', 'w', 'c', 'm']
	else:
		cs = ['w', 'k', 'b', 'm']
	if case.PlotResults:
		plt.rc('figure', facecolor=cs[0], titlesize=30, figsize=[8,8])
		plt.rc('text', usetex=True, color=cs[1])
		plt.rc('font', family='serif', size=24)
		plt.rc('axes', facecolor=cs[0], edgecolor=cs[1], labelsize=30, labelcolor=cs[1], titlecolor=cs[1])
		plt.rc('xtick', color=cs[1], labelcolor=cs[1])
		plt.rc('ytick', color=cs[1], labelcolor=cs[1])
		plt.rc('image', cmap='bwr')
	print("\033[92m    {} \033[00m".format(case.__str__()))
	print("\033[92m    A = {:.2f}   rho = {:.2f}   eta = {:.2f} \033[00m".format(case.A, case.rho, case.eta))
	filestr = type(case).__name__ + '_' + 'A{:.2f}_RHO{:.2f}'.format(case.A, case.rho).replace('.', '')
	if case.GCorder == 2:
		filestr += '_ETA{:.2f}'.format(case.eta).replace('.', '')
	filestr += '_' + case.Method
	if case.Method == 'potentials' and case.Potential == 'turbulent':
		start = time.time()
		plt.rcParams.update({'figure.figsize': [14, 7 * case.GCorder]})
		data = xp.array([case.phi, case.phi_gc1_1, case.phi_gc2_0, case.phi_gc2_2])
		save_data(case, data, filestr)
		time_range = xp.linspace(0, 2 * xp.pi, 50)
		min_phi = (case.phi[:, :, xp.newaxis] * xp.exp(-1j * time_range[xp.newaxis, xp.newaxis, :])).imag.min()
		max_phi = (case.phi[:, :, xp.newaxis] * xp.exp(-1j * time_range[xp.newaxis, xp.newaxis, :])).imag.max()
		min_phi_gc1 = (case.phi_gc1_1[:, :, xp.newaxis] * xp.exp(-1j * time_range[xp.newaxis, xp.newaxis, :])).imag.min()
		max_phi_gc1 = (case.phi_gc1_1[:, :, xp.newaxis] * xp.exp(-1j * time_range[xp.newaxis, xp.newaxis, :])).imag.max()
		vmin, vmax = min(min_phi, min_phi_gc1), max(max_phi, max_phi_gc1)
		extent = (0, 2 * xp.pi, 0, 2 * xp.pi)
		divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
		fig, axs = plt.subplots(case.GCorder, 2)
		ims = []
		for t in time_range:
			frame_ul = (data[0] * xp.exp(-1j * t)).imag
			frame_ll = (data[1] * xp.exp(-1j * t)).imag
			frame_lr = (data[2] - data[3] * xp.exp(-2j * t)).real
			frame_ur = frame_ll + frame_lr
			if case.GCorder == 1:
				im = [axs[0].imshow(frame_ul, origin='lower', extent=extent, animated=True, norm=divnorm), axs[1].imshow(frame_ur, origin='lower', extent=extent, animated=True, norm=divnorm)]
				axs[0].set_title(r'$\phi$')
				axs[1].set_title(r'$\psi$')
			if case.GCorder == 2:
				im = [axs[0, 0].imshow(frame_ul, origin='lower', extent=extent, animated=True, norm=divnorm), axs[0, 1].imshow(frame_ur, origin='lower', extent=extent, animated=True, norm=divnorm), axs[1, 0].imshow(frame_ll, origin='lower', extent=extent, animated=True, norm=divnorm), axs[1, 1].imshow(frame_lr, origin='lower', extent=extent, animated=True, norm=divnorm)]
				axs[0, 0].set_title(r'$\phi$')
				axs[0, 1].set_title(r'$\psi$')
				axs[1, 0].set_title(r'$\psi^{(1)}$')
				axs[1, 1].set_title(r'$\psi^{(2)}$')
			for ax in axs.flat:
				ax.set_xlabel('$x$')
				ax.set_ylabel('$y$')
				ax.set_xticks([0, xp.pi, 2 * xp.pi])
				ax.set_yticks([0, xp.pi, 2 * xp.pi])
				ax.grid(case.grid)
				ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
				ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
			ims.append(im)
		if case.GCorder == 1:
			fig.colorbar(im[1], ax=axs.ravel().tolist())
		elif case.GCorder == 2:
			fig.colorbar(im[1], ax=axs[0, :].ravel().tolist())
			fig.colorbar(im[3], ax=axs[1, :].ravel().tolist())
		ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000).save(filestr + '.gif', writer=PillowWriter(fps=30), dpi=case.dpi)
		print("\033[90m        Computation finished in {} seconds \033[00m".format(int(time.time() - start)))
		print("\033[90m        Animation saved in {}.gif \033[00m".format(filestr))
	elif case.Method in ['poincare_gc', 'poincare_ions', 'diffusion_gc', 'diffusion_ions']:
		order_mu = 1
		if case.init == 'random':
			y0 = 2 * xp.pi * xp.random.rand(2 * case.Ntraj)
		elif case.init == 'fixed':
			y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(case.Ntraj)), endpoint=False)
			y_mat = xp.meshgrid(y_vec, y_vec)
			y0 = xp.concatenate((y_mat[0], y_mat[1]), axis=None)
			case.Ntraj = int(xp.sqrt(case.Ntraj))**2
		if case.Method.endswith('_ions'):
			phi_perp = 2 * xp.pi * xp.random.rand(case.Ntraj)
			y0 = xp.concatenate((y0, xp.cos(phi_perp), xp.sin(phi_perp)), axis=None)
		if case.check_energy:
			y0 = xp.concatenate((y0, xp.zeros(case.Ntraj)), axis=None)
		t_eval = 2 * xp.pi * xp.arange(0, case.Tf + 1)
		start = time.time()
		if not case.TwoStepIntegration:
			if case.Method.endswith('_gc'):
				sol = solve_ivp(case.eqn_gc, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.TimeStep, atol=1, rtol=1)
				sol_ = xp.split(sol.y, 2 + case.check_energy)
				x, y = sol_[:2]
				if case.check_energy:
					k = sol_[2]
			elif case.Method.endswith('_ions'):
				sol = solve_ivp(case.eqn_ions, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.TimeStep, atol=1, rtol=1)
				sol_ = xp.split(sol.y, 4 + case.check_energy)
				x, y, vx, vy = sol_[:4]
				if case.check_energy:
					k = sol_[4]
			untrapped = compute_untrapped((x, y), thresh=case.threshold)
			trapped = xp.logical_not(untrapped)
			x_un, y_un = x[untrapped, :], y[untrapped, :]
			x_tr, y_tr = x[trapped, :], y[trapped, :]
			if case.check_energy:
				k_un, k_tr = k[untrapped, :], k[trapped, :]
			if case.Method.endswith('_ions'):
				vx_un, vy_un = vx[untrapped, :], vy[untrapped, :]
				vx_tr, vy_tr = vx[trapped, :], vy[trapped, :]
		else:
			if case.Method.endswith('_gc'):
				sol = solve_ivp(case.eqn_gc, (0, t_eval[case.Tmid]), y0, t_eval=t_eval[:case.Tmid+1], max_step=case.TimeStep, atol=1, rtol=1)
				sol_ = xp.split(sol.y, 2 + case.check_energy)
				x, y = sol_[:2]
				if case.check_energy:
					k = sol_[2]
			elif case.Method.endswith('_ions'):
				sol = solve_ivp(case.eqn_ions, (0, t_eval[case.Tmid]), y0, t_eval=t_eval[:case.Tmid+1], max_step=case.TimeStep, atol=1, rtol=1)
				sol_ = xp.split(sol.y, 4 + case.check_energy)
				x, y, vx, vy = sol_[:4]
				if case.check_energy:
					k = sol_[4]
			untrapped = compute_untrapped((x, y), thresh=case.threshold)
			trapped = xp.logical_not(untrapped)
			x_un, y_un = x[untrapped, :], y[untrapped, :]
			x_tr, y_tr = x[trapped, :], y[trapped, :]
			if case.check_energy:
				k_un, k_tr = k[untrapped, :], k[trapped, :]
			if case.Method.endswith('_ions'):
				vx_un, vy_un = vx[untrapped, :], vy[untrapped, :]
				vx_tr, vy_tr = vx[trapped, :], vy[trapped, :]
			print("\033[90m        Continuing with the integration of {} untrapped particles... \033[00m".format(untrapped.sum()))
			if case.Method.endswith('_gc'):
				y0 = xp.concatenate((x_un[:, -1], y_un[:, -1]), axis=None)
				sol = solve_ivp(case.eqn_gc, (t_eval[case.Tmid], t_eval.max()), y0, t_eval=t_eval[case.Tmid:], max_step=case.TimeStep, atol=1, rtol=1)
				sol_ = xp.split(sol.y, 2 + case.check_energy)
				x, y = sol_[:2]
				if case.check_energy:
					k = sol_[2]
			elif case.Method.endswith('_ions'):
				y0 = xp.concatenate((x_un[:, -1], y_un[:, -1], vx_un[:, -1], vy_un[:, -1]), axis=None)
				sol = solve_ivp(case.eqn_ions, (t_eval[case.Tmid], t_eval.max()), y0, t_eval=t_eval[case.Tmid:], max_step=case.TimeStep, atol=1, rtol=1)
				sol_ = xp.split(sol.y, 4 + case.check_energy)
				x, y, vx, vy = sol_[:4]
				if case.check_energy:
					k = sol_[4]
			x_un = xp.concatenate((x_un, x[:, 1:]), axis=1)
			y_un = xp.concatenate((y_un, y[:, 1:]), axis=1)
			if case.check_energy:
				k_un = xp.concatenate((k_un, k[:, 1:]), axis=1)
			if case.Method.endswith('_ions'):
				vx_un = xp.concatenate((vx_un, vx[:, 1:]), axis=1)
				vy_un = xp.concatenate((vy_un, vy[:, 1:]), axis=1)
		print("\033[90m        Computation finished in {} seconds \033[00m".format(int(time.time() - start)))
		if case.Method.startswith('poincare'):
			if case.Method == 'poincare_gc':
				data = xp.array([x_un, y_un, x_tr, y_tr], dtype=object)
				info = 'x_untrapped / y_untrapped / x_trapped / y_trapped'
				if case.check_energy:
					h_un = case.compute_energy(t_eval, x_un, y_un, k_un, type='gc')
					h_tr = case.compute_energy(t_eval[:case.Tmid+1], x_tr, y_tr, k_tr, type='gc')
					h_un = ((h_un.T - h_un[:, 0]) / h_un[:, 0]).T
					h_tr = ((h_tr.T - h_tr[:, 0]) / h_tr[:, 0]).T
					data = xp.array([*data, h_un, h_tr], dtype=object)
					info += ' / h_untrapped / h_trapped'
			elif case.Method == 'poincare_ions':
				x_gc_un, y_gc_un = case.ions2gc(t_eval, x_un, y_un, vx_un, vy_un, order=1)
				x_gc_tr, y_gc_tr = case.ions2gc(t_eval[:case.Tmid+1], x_tr, y_tr, vx_tr, vy_tr, order=1)
				mu_un = case.compute_mu(t_eval, x_un, y_un, vx_un, vy_un, order=order_mu)
				mu_tr = case.compute_mu(t_eval[:case.Tmid+1], x_tr, y_tr, vx_tr, vy_tr, order=order_mu)
				data = xp.array([x_un, y_un, vx_un, vy_un, x_tr, y_tr, vx_tr, vy_tr, x_gc_un, y_gc_un, x_gc_tr, y_gc_tr, mu_un, mu_tr], dtype=object)
				info = 'x_untrapped / y_untrapped / vx_untrapped / vy_untrapped / x_trapped / y_trapped / vx_trapped / vy_trapped / x_gc_untrapped / y_gc_untrapped / x_gc_trapped / y_gc_trapped / mu_untrapped / mu_trapped'
				if case.check_energy:
					h_un = case.compute_energy(t_eval, x_un, y_un, vx_un, vy_un, k_un, type='ions')
					h_tr = case.compute_energy(t_eval[:case.Tmid+1], x_tr, y_tr, vx_tr, vy_tr, k_tr, type='ions')
					h_un = ((h_un.T - h_un[:, 0]) / h_un[:, 0]).T
					h_tr = ((h_tr.T - h_tr[:, 0]) / h_tr[:, 0]).T
					data = xp.array([*data, h_un, h_tr], dtype=object)
					info += ' / h_untrapped / h_trapped'
			save_data(case, data, filestr, info=info)
			if case.PlotResults:
				fig, ax = plt.subplots(1, 1)
				ax.set_xlabel('$x$')
				ax.set_ylabel('$y$')
				ax.grid(case.grid)
				if case.modulo:
					if case.Method == 'poincare_gc':
						ax.plot(x_un % (2 * xp.pi), y_un % (2 * xp.pi), '.', color=cs[2], markersize=3, markeredgecolor='none')
						ax.plot(x_tr % (2 * xp.pi), y_tr % (2 * xp.pi), '.', color=cs[3], markersize=3, markeredgecolor='none')
					elif case.Method == "poincare_ions":
						ax.plot(x_un % (2 * xp.pi), y_un % (2 * xp.pi), '.', color=cs[2], markersize=1, markeredgecolor='none')
						ax.plot(x_tr % (2 * xp.pi), y_tr % (2 * xp.pi), '.', color=cs[3], markersize=1, markeredgecolor='none')
						ax.plot(x_gc_un % (2 * xp.pi), y_gc_un % (2 * xp.pi), '.', color=cs[2], markersize=3, markeredgecolor='none')
						ax.plot(x_gc_tr % (2 * xp.pi), y_gc_tr % (2 * xp.pi), '.', color=cs[3], markersize=3, markeredgecolor='none')
					ax.set_xlim(0, 2 * xp.pi)
					ax.set_ylim(0, 2 * xp.pi)
					ax.set_xticks([0, xp.pi, 2 * xp.pi])
					ax.set_yticks([0, xp.pi, 2 * xp.pi])
					ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
					ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
				if not case.modulo:
					if case.Method == 'poincare_gc':
						ax.plot(x_un, y_un, '.', color=cs[2], markersize=3, markeredgecolor='none')
						ax.plot(x_tr, y_tr, '.', color=cs[3], markersize=3, markeredgecolor='none')
					elif case.Method == "poincare_ions":
						ax.plot(x_un, y_un, '.', color=cs[2], markersize=1, markeredgecolor='none')
						ax.plot(x_tr, y_tr, '.', color=cs[3], markersize=1, markeredgecolor='none')
						ax.plot(x_gc_un, y_gc_un, '.', color=cs[2], markersize=3, markeredgecolor='none')
						ax.plot(x_gc_tr, y_gc_tr, '.', color=cs[3], markersize=3, markeredgecolor='none')
					ax.add_patch(Rectangle((0, 0), 2 * xp.pi, 2 * xp.pi, facecolor='None', edgecolor='r', lw=2))
					ax.set_aspect('equal')
				if case.SaveData:
					fig.savefig(filestr + '.png', dpi=case.dpi)
					print("\033[90m        Figure saved in {}.png \033[00m".format(filestr))
				plt.pause(0.5)
		if case.Method.startswith('diffusion'):
			if untrapped.sum() <= 5:
				print("\033[33m          Warning: not enough untrapped trajectories ({}) \033[00m".format(untrapped.sum()))
			else:
				t, t_win, r2, r2_win, r2_fit, slope, popt, rvalue, R2 = compute_r2(case, t_eval, x_un, y_un)
				n_trapped = xp.logical_not(untrapped).sum()
				print("\033[96m          trapped particles = {} \033[00m".format(n_trapped))
				print("\033[96m          diffusion data : D = {:.6f}  /  interp = '.format(res.slope) + ', '.join(['{:.6f}'.format(p) for p in popt]) + '\033[00m")
				print("\033[96m              with R2        = {:.6f}  /  {:.6f} \033[00m".format(rvalue**2, R2))
				vec_data = [case.A, case.rho, case.eta, n_trapped / case.Ntraj, slope, *popt, rvalue**2, R2]
				file = open(type(case).__name__ + '_' + case.Method + '.txt', 'a')
				if os.path.getsize(file.name) == 0:
					file.writelines('%  diffusion laws: r^2 = D t   and   r^2 = (a t)^b \n')
					file.writelines('%  A        rho      eta    trapped     D       a        b     R2(diff)   R2(interp)' + '\n')
				file.writelines(' '.join(['{:.6f}'.format(data) for data in vec_data]) + '\n')
				file.close()
				if case.Method == 'diffusion_gc':
					data = xp.array([x_un, y_un, x_tr, y_tr, t, r2], dtype=object)
					info = 'x_untrapped / y_untrapped / x_trapped / y_trapped / t / r2'
				elif case.Method == 'diffusion_ions':
					x_gc_un, y_gc_un = case.ions2gc(x_un, y_un, vx_un, vy_un, order=1)
					x_gc_tr, y_gc_tr = case.ions2gc(x_tr, y_tr, vx_tr, vy_tr, order=1)
					mu_un = case.compute_mu(t_eval, x_un, y_un, vx_un, vy_un, order=order_mu)
					mu_tr = case.compute_mu(t_eval, x_tr, y_tr, vx_tr, vy_tr, order=order_mu)
					t, t_win, r2_gc, r2_win_gc, r2_fit_gc, slope_gc, popt_gc, rvalue_gc, R2_gc = compute_r2(case, t_eval, x_gc_un, y_gc_un)
					data = xp.array([x_un, y_un, vx_un, vy_un, x_tr, y_tr, vx_tr, vy_tr, x_gc_un, y_gc_un, x_gc_tr, y_gc_tr, mu_un, mu_tr, t, r2, r2_gc], dtype=object)
					info = 'x_untrapped / y_untrapped / vx_untrapped / vy_untrapped / x_trapped / y_trapped / vx_trapped / vy_trapped / x_gc_untrapped / y_gc_untrapped / x_gc_trapped / y_gc_trapped / mu_untrapped / mu_trapped / t / r2 / r2_gc'
				save_data(case, data, filestr, info=info)
				if case.PlotResults:
					fig, ax = plt.subplots(1, 1)
					ax.set_xlabel('$t$')
					ax.set_ylabel('$r^2$')
					plt.plot(t, r2, cs[1], lw=1)
					plt.plot(t_win, r2_win, cs[2], lw=2)
					plt.plot(t_win, r2_fit, cs[3], lw=2)
					if case.SaveData:
						fig.savefig(filestr + '.png', dpi=case.dpi)
						print("\033[90m        Figure saved in {}.png \033[00m".format(filestr))
					plt.pause(0.5)

def compute_untrapped(x, thresh=0, axis=1, output=[True, False]):
	vec = xp.sqrt(xp.sum([xel.ptp(axis=axis)**2 for xel in x], axis=0)) > thresh
	return xp.where(vec==True, *output)

def compute_r2(case, teval, x, y):
	func_fit = lambda t, a, b: (a * t)**b
	r2 = xp.zeros(case.Tf)
	for t in range(case.Tf):
		r2[t] = ((x[:, t:] - x[:, :-t if t else None])**2 + (y[:, t:] - y[:, :-t if t else None])**2).mean()
	t = teval[:-1]
	t_win, r2_win = teval[case.Tf//8:7*case.Tf//8], r2[case.Tf//8:7*case.Tf//8]
	popt, pcov = curve_fit(func_fit, t_win, r2_win, bounds=((0, 0.25), (xp.inf, 3)))
	res = linregress(t_win, r2_win)
	r2_fit = func_fit(t_win, *popt)
	R2 = r2_score(r2_win, r2_fit)
	return t, t_win, r2, r2_win, r2_fit, res.slope, popt, res.rvalue, R2

def save_data(case, data, filestr, info=[]):
	if case.SaveData:
		mdic = case.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print("\033[90m        Results saved in {}.mat \033[00m".format(filestr))
