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
	print('\033[92m    {} \033[00m'.format(case.__str__()))
	print('\033[92m    A = {:.2f}   rho = {:.2f}   eta = {:.2f} \033[00m'.format(case.A, case.rho, case.eta))
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
		print('\033[90m        Computation finished in {} seconds \033[00m'.format(int(time.time() - start)))
		print('\033[90m        Animation saved in {}.gif \033[00m'.format(filestr))
	elif case.Method in ['poincare_gc', 'poincare_ions', 'diffusion_gc', 'diffusion_ions']:
		if case.init == 'random':
			y0 = 2 * xp.pi * xp.random.rand(2 * case.Ntraj)
		elif case.init == 'fixed':
			y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(case.Ntraj)), endpoint=False)
			y_mat = xp.meshgrid(y_vec, y_vec)
			y0 = xp.concatenate((y_mat[0], y_mat[1]), axis=None)
			case.Ntraj = int(xp.sqrt(case.Ntraj))**2
		if case.Method in ['poincare_ions', 'diffusion_ions']:
			v_perp = xp.random.normal(scale=xp.sqrt(case.Temperature), size=case.Ntraj)
			phi_perp = 2 * xp.pi * xp.random.rand(case.Ntraj)
			vx = v_perp * xp.cos(phi_perp)
			vy = v_perp * xp.sin(phi_perp)
			y0 = xp.concatenate((y0, vx, vy), axis=None)
		t_eval = 2 * xp.pi * xp.arange(0, case.Tf + 1)
		start = time.time()
		if not case.TwoStepIntegration:
			if case.Method in ['poincare_gc', 'diffusion_gc']:
				sol = solve_ivp(case.eqn_gc, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.TimeStep, atol=1, rtol=1)
				x, y = xp.split(sol.y, 2)
			elif case.Method in ['poincare_ions', 'diffusion_ions']:
				sol = solve_ivp(case.eqn_ions, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.TimeStep, atol=1, rtol=1)
				x, y, vx, vy = xp.split(sol.y, 4)
			untrapped = compute_untrapped((x, y), thresh=case.threshold)
			x_un, y_un = x[untrapped, :], y[untrapped, :]
			x_tr, y_tr = x[xp.logical_not(untrapped), :], y[xp.logical_not(untrapped), :]
			if case.Method in ['poincare_ions', 'diffusion_ions']:
				vx_un, vy_un = vx[untrapped, :], vy[untrapped, :]
				vx_tr, vy_tr = vx[xp.logical_not(untrapped), :], vy[xp.logical_not(untrapped), :]
		else:
			if case.Method in ['poincare_gc', 'diffusion_gc']:
				sol = solve_ivp(case.eqn_gc, (0, t_eval[case.Tmid]), y0, t_eval=t_eval[:case.Tmid+1], max_step=case.TimeStep, atol=1, rtol=1)
				x, y = xp.split(sol.y, 2)
			elif case.Method in ['poincare_ions', 'diffusion_ions']:
				sol = solve_ivp(case.eqn_ions, (0, t_eval[case.Tmid]), y0, t_eval=t_eval[:case.Tmid+1], max_step=case.TimeStep, atol=1, rtol=1)
				x, y, vx, vy = xp.split(sol.y, 4)
			untrapped = compute_untrapped((x, y), thresh=case.threshold)
			x_un, y_un = x[untrapped, :], y[untrapped, :]
			x_tr, y_tr = x[xp.logical_not(untrapped), :], y[xp.logical_not(untrapped), :]
			if case.Method in ['poincare_ions', 'diffusion_ions']:
				vx_un, vy_un = vx[untrapped, :], vy[untrapped, :]
				vx_tr, vy_tr = vx[xp.logical_not(untrapped), :], vy[xp.logical_not(untrapped), :]
			print('\033[90m        Continuing with the integration of {} untrapped particles... \033[00m'.format(untrapped.sum()))
			if case.Method in ['poincare_gc', 'diffusion_gc']:
				y0 = xp.concatenate((x_un[:, -1], y_un[:, -1]), axis=None)
				sol = solve_ivp(case.eqn_gc, (t_eval[case.Tmid], t_eval.max()), y0, t_eval=t_eval[case.Tmid:], max_step=case.TimeStep, atol=1, rtol=1)
				x, y = xp.split(sol.y, 2)
			elif case.Method in ['poincare_ions', 'diffusion_ions']:
				y0 = xp.concatenate((x_un[:, -1], y_un[:, -1], vx_un[:, -1], vy_un[:, -1]), axis=None)
				sol = solve_ivp(case.eqn_ions, (t_eval[case.Tmid], t_eval.max()), y0, t_eval=t_eval[case.Tmid:], max_step=case.TimeStep, atol=1, rtol=1)
				x, y, vx, vy = xp.split(sol.y, 4)
			x_un = xp.concatenate((x_un, x[:, 1:]), axis=1)
			y_un = xp.concatenate((y_un, y[:, 1:]), axis=1)
			if case.Method in ['poincare_ions', 'diffusion_ions']:
				vx_un = xp.concatenate((vx_un, vx[:, 1:]), axis=1)
				vy_un = xp.concatenate((vy_un, vy[:, 1:]), axis=1)
		print('\033[90m        Computation finished in {} seconds \033[00m'.format(int(time.time() - start)))
		if case.Method in ['poincare_gc', 'poincare_ions']:
			if case.Method == 'poincare_gc':
				data = xp.array([x_un, y_un, x_tr, y_tr], dtype=object)
				info = 'x_untrapped / y_untrapped / x_trapped / y_trapped'
			elif case.Method == 'poincare_ions':
				data = xp.array([x_un, y_un, vx_un, vy_un, x_tr, y_tr, vx_tr, vy_tr], dtype=object)
				info = 'x_untrapped / y_untrapped / vx_untrapped / vy_untrapped / x_trapped / y_trapped / vx_trapped / vy_trapped'
			save_data(case, data, filestr, info=info)
			if case.PlotResults:
				fig, ax = plt.subplots(1, 1)
				ax.set_xlabel('$x$')
				ax.set_ylabel('$y$')
				ax.grid(case.grid)
				if case.modulo:
					ax.plot(x_un % (2 * xp.pi), y_un % (2 * xp.pi), '.', color=cs[2], markersize=2, markeredgecolor='none')
					ax.plot(x_tr % (2 * xp.pi), y_tr % (2 * xp.pi), '.', color=cs[3], markersize=2, markeredgecolor='none')
					ax.set_xlim(0, 2 * xp.pi)
					ax.set_ylim(0, 2 * xp.pi)
					ax.set_xticks([0, xp.pi, 2 * xp.pi])
					ax.set_yticks([0, xp.pi, 2 * xp.pi])
					ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
					ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
				if not case.modulo:
					ax.plot(x_un, y_un, '.', color=cs[2], markersize=2, markeredgecolor='none')
					ax.plot(x_tr, y_tr, '.', color=cs[3], markersize=2, markeredgecolor='none')
					ax.add_patch(Rectangle((0, 0), 2 * xp.pi, 2 * xp.pi, facecolor='None', edgecolor='r', lw=2))
					ax.set_aspect('equal')
				if case.SaveData:
					fig.savefig(filestr + '.png', dpi=case.dpi)
					print('\033[90m        Figure saved in {}.png \033[00m'.format(filestr))
				plt.pause(0.5)
		if case.Method in ['diffusion_gc', 'diffusion_ions']:
			if untrapped.sum() <= 5:
				print('\033[33m          Warning: not enough untrapped trajectories ({}) \033[00m'.format(untrapped.sum()))
			else:
				r2 = xp.zeros(case.Tf)
				for t in range(case.Tf):
					r2[t] = ((x_un[:, t:] - x_un[:, :-t if t else None])**2 + (y_un[:, t:] - y_un[:, :-t if t else None])**2).mean()
				t_win, r2_win = t_eval[case.Tf//8:7*case.Tf//8], r2[case.Tf//8:7*case.Tf//8]
				func_fit = lambda t, a, b: (a * t)**b
				popt, pcov = curve_fit(func_fit, t_win, r2_win, bounds=((0, 0.25), (xp.inf, 3)))
				t = t_eval[:-1]
				r2_fit = func_fit(t_win, *popt)
				R2 = r2_score(r2_win, r2_fit)
				trapped = xp.logical_not(untrapped).sum()
				print('\033[96m          trapped particles = {} \033[00m'.format(trapped))
				print('\033[96m          diffusion data    = [' + ', '.join(['{:.6f}'.format(p) for p in popt]) + ']\033[00m')
				print('\033[96m              with an R2    = {:.6f} \033[00m'.format(R2))
				vec_data = [case.A, case.rho, case.eta, trapped / case.Ntraj, *popt, R2]
				file = open(type(case).__name__ + '_' + case.Method + '.txt', 'a')
				if os.path.getsize(file.name) == 0:
					file.writelines('%  diffusion law: r^2 = (a t)^b \n')
					file.writelines('%  A        rho      eta    trapped    a        b        R2' + '\n')
				file.writelines(' '.join(['{:.6f}'.format(data) for data in vec_data]) + '\n')
				file.close()
				if case.Method == 'diffusion_gc':
					data = xp.array([x_un, y_un, x_tr, y_tr, t, r2, r2_fit], dtype=object)
					info = 'x_untrapped / y_untrapped / x_trapped / y_trapped / t / r2 / r2_fit'
				elif case.Method == 'diffusion_ions':
					data = xp.array([x_un, y_un, vx_un, vy_un, x_tr, y_tr, vx_tr, vy_tr, t, r2, r2_fit], dtype=object)
					info = 'x_untrapped / y_untrapped / vx_untrapped / vy_untrapped / x_trapped / y_trapped / vx_trapped / vy_trapped / t / r2 / r2_fit'
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
						print('\033[90m        Figure saved in {}.png \033[00m'.format(filestr))
					plt.pause(0.5)

def compute_untrapped(x, thresh=0, axis=1, output=[True, False]):
	vec = xp.sqrt(xp.sum([xel.ptp(axis=axis)**2 for xel in x], axis=0)) > thresh
	return xp.where(vec==True, *output)

def save_data(case, data, filestr, info=[]):
	if case.SaveData:
		mdic = case.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr, mdic)
		print('\033[90m        Results saved in {}.mat \033[00m'.format(filestr))
