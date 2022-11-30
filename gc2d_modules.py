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
		cs = ['k', 'w', 'c', 'm', 'r']
	else:
		cs = ['w', 'k', 'b', 'm', 'r']
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
			sol = solve_ivp(case.eqn, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.TimeStep, atol=1, rtol=1)
			sol_ = xp.split(sol.y, case.dim)
			Trapped = Trajectory(case, t_eval, sol_, 'trap')
			Diffusive = Trajectory(case, t_eval, sol_, 'diff')
			Ballistic = Trajectory(case, t_eval, sol_, 'ball')
		else:
			sol = solve_ivp(case.eqn, (0, t_eval[case.Tmid]), y0, t_eval=t_eval[:case.Tmid+1], max_step=case.TimeStep, atol=1, rtol=1)
			sol_ = xp.split(sol.y, case.dim)
			Trapped = Trajectory(case, t_eval[:case.Tmid+1], sol_, 'trapped')
			Untrapped = Trajectory(case, t_eval[:case.Tmid+1], sol_, 'untrapped')
			print("\033[90m        Continuing with the integration of {} untrapped particles... \033[00m".format(Untrapped.x[:, 0].size))
			y0 = xp.concatenate((Untrapped.x[:, -1], Untrapped.y[:, -1]), axis=None)
			if case.Method.endswith('_ions'):
				y0 = xp.concatenate((y0, Untrapped.vx[:, -1], Untrapped.vy[:, -1]), axis=None)
			if case.check_energy:
				y0 = xp.concatenate((y0, Untrapped.k[:, -1]), axis=None)
			sol = solve_ivp(case.eqn, (t_eval[case.Tmid], t_eval.max()), y0, t_eval=t_eval[case.Tmid:], max_step=case.TimeStep, atol=1, rtol=1)
			sol_ = xp.split(sol.y, case.dim)
			x, y = sol_[:2]
			Untrapped.x = xp.concatenate((Untrapped.x, x[:, 1:]), axis=1)
			Untrapped.y = xp.concatenate((Untrapped.y, y[:, 1:]), axis=1)
			vec_un = (Untrapped.x, Untrapped.y)
			if case.Method.endswith('_ions'):
				vx, vy = sol_[2:4]
				Untrapped.vx = xp.concatenate((Untrapped.vx, vx[:, 1:]), axis=1)
				Untrapped.vy = xp.concatenate((Untrapped.vy, vy[:, 1:]), axis=1)
				vec_un += (Untrapped.vx, Untrapped.vy)
			if case.check_energy:
				k = sol_[-1]
				Untrapped.k = xp.concatenate((Untrapped.k, k[:, 1:]), axis=1)
				vec_un += (Untrapped.k,)
			Diffusive = Trajectory(case, t_eval, vec_un, 'diff')
			Ballistic = Trajectory(case, t_eval, vec_un, 'ball')
		data = [Trapped, Diffusive, Ballistic]
		info = 'Trapped / Diffusive / Ballistic'
		print("\033[90m        Computation finished in {} seconds \033[00m".format(int(time.time() - start)))
		if case.Method.startswith('poincare') and case.PlotResults:
			fig, ax = plt.subplots(1, 1)
			ax.set_xlabel('$x$')
			ax.set_ylabel('$y$')
			ax.grid(case.grid)
			if case.modulo:
				if case.Method == 'poincare_gc':
					for traj in [Trapped, Diffusive, Ballistic]:
						if traj.size:
							ax.plot(traj.x % (2 * xp.pi), traj.y % (2 * xp.pi), '.', color=traj.color, markersize=3, markeredgecolor='none')
				elif case.Method == "poincare_ions":
					for traj in [Trapped, Diffusive, Ballistic]:
						if traj.size:
							ax.plot(traj.x % (2 * xp.pi), traj.y % (2 * xp.pi), '.', color=traj.color, markersize=1, markeredgecolor='none')
							ax.plot(traj.x_gc % (2 * xp.pi), traj.y_gc % (2 * xp.pi), '.', color=traj.color, markersize=3, markeredgecolor='none')
				ax.set_xlim(0, 2 * xp.pi)
				ax.set_ylim(0, 2 * xp.pi)
				ax.set_xticks([0, xp.pi, 2 * xp.pi])
				ax.set_yticks([0, xp.pi, 2 * xp.pi])
				ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
				ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
			elif not case.modulo:
				if case.Method == 'poincare_gc':
					for traj in [Trapped, Diffusive, Ballistic]:
						if traj.size:
							ax.plot(traj.x, traj.y, '.', color=traj.color, markersize=3, markeredgecolor='none')
				elif case.Method == "poincare_ions":
					for traj in [Trapped, Diffusive, Ballistic]:
						if traj.size:
							ax.plot(traj.x, traj.y, '.', color=traj.color, markersize=1, markeredgecolor='none')
							ax.plot(traj.x_gc, traj.y_gc, '.', color=traj.color, markersize=3, markeredgecolor='none')
				ax.add_patch(Rectangle((0, 0), 2 * xp.pi, 2 * xp.pi, facecolor='None', edgecolor='g', lw=2))
				ax.set_aspect('equal')
			if case.SaveData:
				fig.savefig(filestr + '.png', dpi=case.dpi)
				print("\033[90m        Figure saved in {}.png \033[00m".format(filestr))
			plt.pause(0.5)
		if case.Method.startswith('diffusion'):
			vec_data = [case.A, case.rho, case.eta, Trapped.size / case.Ntraj]
			print("\033[96m          trap ({}) \033[00m".format(Trapped.size))
			for traj in [Diffusive, Ballistic]:
				if traj.size:
					print("\033[96m          {} ({}) : D = ({:.6f} ; {:.6f})  /  interp = (".format(traj.type, traj.size, traj.diff_data[0], traj.diff_data[1]**2) + ", ".join(["{:.6f}".format(p) for p in traj.interp_data[0]]) + " ; {:.6f})\033[00m".format(traj.interp_data[1]))
					vec_data.extend([traj.size / case.Ntraj, traj.diff_data[0], *traj.interp_data[0], traj.diff_data[1]**2, traj.interp_data[1]])
			file = open(type(case).__name__ + '_' + case.Method + '.txt', 'a')
			if os.path.getsize(file.name) == 0:
				file.writelines('%  diffusion laws: r^2 = D t   and   r^2 = (a t)^b \n')
				file.writelines('%  A        rho      eta   trapped  diffusive    D        a        b    R2(diff) R2(interp) ballistic  D        a        b    R2(diff) R2(interp)' + '\n')
			file.writelines(' '.join(['{:.6f}'.format(data) for data in vec_data]) + '\n')
			file.close()
			if case.PlotResults:
				fig, ax = plt.subplots(1, 1)
				ax.set_xlabel('$t$')
				ax.set_ylabel('$r^2$')
				for traj in [Diffusive, Ballistic]:
					if traj.size:
						plt.plot(traj.t, traj.r2, ':', color=traj.color, lw=1)
						plt.plot(traj.t_win, traj.r2_win, '-', color=traj.color, lw=2)
						plt.plot(traj.t_win, traj.r2_fit, '-.', color=traj.color, lw=2)
				if case.SaveData:
					fig.savefig(filestr + '.png', dpi=case.dpi)
					print("\033[90m        Figure saved in {}.png \033[00m".format(filestr))
				plt.pause(0.5)
		save_data(case, data, filestr, info=info)

def define_type(case, x, axis=1, output=[0, 1, 2]):
	vec = xp.repeat(output[1], x[0][:, 0].size)
	delta = [xel.ptp(axis=axis)**2 for xel in x[:2]]
	vec[xp.sqrt(xp.sum(delta, axis=0)) <= case.threshold] = output[0]
	vec[delta[0] / delta[1] > case.threshold**2] = output[2]
	vec[delta[1] / delta[0] > case.threshold**2] = output[2]
	return vec

def save_data(case, data, filestr, info=[]):
	if case.SaveData:
		mdic = case.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print("\033[90m        Results saved in {}.mat \033[00m".format(filestr))

class Trajectory:
	def __str__(self):
		return "Guiding-center trajectory (gc or ion modes)"

	def __init__(self, case, t, x, type):
		if type in ['trap', 'diff', 'ball']:
			type_ = define_type(case, x, output=['trap', 'diff', 'ball'])
		elif type in ['trapped', 'untrapped']:
			type_ = define_type(case, x, output=['trapped', 'untrapped', 'untrapped'])
		x_ = [x[it][type_==type, :] for it in range(case.dim)]
		self.t, self.x, self.y = t, x_[0], x_[1]
		if self.x.size:
			if case.Method.endswith('_ions'):
				self.vx, self.vy = x_[2], x_[3]
				self.x_gc, self.y_gc = case.ions2gc(t, *x_)
			if case.check_energy:
				h = case.compute_energy(t, *x_)
				self.h = ((h.T - h[:, 0]) / h[:, 0]).T
			if case.check_mu:
				self.mu = case.compute_mu(t, *x_)
			if case.darkmode:
				cs = ['k', 'w', 'c', 'm', 'r']
			else:
				cs = ['w', 'k', 'b', 'm', 'r']
			self.color = {type in ['trap', 'trapped']: cs[2], type in ['diff', 'untrapped']: cs[3], type == 'ball': cs[4]}.get(True, cs[1])
			self.type = type
			self.size = self.x[:, 0].size
			if case.Method.startswith('diffusion') and type in ['diff', 'ball']:
				nt = self.x[0, :].size
				xd, yd = (self.x, self.y) if case.Method.endswith('_gc') else (self.x_gc, self.y_gc)
				self.r2 = xp.zeros(nt)
				for _ in range(nt):
					self.r2[_] = ((xd[:, _:] - xd[:, :-_ if _ else None])**2 + (yd[:, _:] - yd[:, :-_ if _ else None])**2).mean()
				self.t_win, self.r2_win = self.t[nt//8:7*nt//8], self.r2[nt//8:7*nt//8]
				res = linregress(self.t_win, self.r2_win)
				self.diff_data = [res.slope, res.rvalue]
				func_fit = lambda t, a, b: (a * t)**b
				popt, pcov = curve_fit(func_fit, self.t_win, self.r2_win, bounds=((0, 0.25), (xp.inf, 3)))
				self.r2_fit = func_fit(self.t_win, *popt)
				R2 = r2_score(self.r2_win, self.r2_fit)
				self.interp_data = [popt, R2]
		else:
			self.size = 0
