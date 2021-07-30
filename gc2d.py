import numpy as xp
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.integrate import solve_ivp
from scipy.special import jv
import sympy as sp
from scipy.stats import linregress
from scipy.io import savemat
import time
from datetime import date

def main():
	dict_params = {
        'N': 2 ** 10,
        'M': 25,
        'A': 0.9}
	dict_params.update({
        'FLR': [True, False],
        'flr_order': ['all', 'all'],
        'rho': 0.03,
        'gc_order': 1,
        'eta': 0.1})
	dict_params.update({
        #'method': 'plot_potentials',
        #'method': 'poincare',
        'method': 'diffusion',
        'modulo': False,
        'Ntraj': 2000,
        'Tf': 5000,
        'timestep': 0.05,
        'save_results': True,
        'plot_results': True})

	case = GC2D(dict_params)
	print('\033[92m    {} \033[00m'.format(case.__str__()))
	print('\033[92m    A = {}   rho = {}   eta = {} \033[00m'.format(case.A, case.rho, case.eta))
	filestr = ('A' + str(case.A) + '_FLR' + str(case.rho) + '_GC' + str(case.gc_order)).replace('.', '')
	if case.gc_order == 2:
		filestr += ('_eta' + str(case.eta)).replace('.', '')
	if case.method == 'plot_potentials':
		data = xp.array([case.phi, case.phi_, case.phi_gc2_0, case.phi_gc2_2])
		case.save_data('potentials', data, filestr)
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
		case.save_data('poincare', xp.array([sol.y[:case.Ntraj, :], sol.y[case.Ntraj:, :]]).transpose(), filestr)
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
		trapped = (max_y <= xp.pi).sum()
		case.save_data('diffusion', [trapped, diff_data.slope], filestr, info='trapped particles / diffusion coefficient')
		print('\033[96m          trapped particles = {} \033[00m'.format(trapped))
		print('\033[96m          diffusion coefficient = {:.6f} \033[00m'.format(diff_data.slope))
		print('\033[96m                     with an R2 = {:.6f} \033[00m'.format(diff_data.rvalue**2))
		if case.plot_results:
			plt.figure(figsize=(8, 8))
			plt.plot(t_eval, r2, 'b', linewidth=2)
			plt.show()


class GC2D:
	def __repr__(self):
		return '{self.__class__.__name__}({self.DictParams})'.format(self=self)

	def __str__(self):
		return '2D Guiding Center ({self.__class__.__name__}) for the turbulent potential with FLR = {self.flr_order} and GC order = {self.gc_order}'.format(self=self)

	def __init__(self, dict_params):
		if not dict_params['FLR'][0]:
			dict_params['flr_order'][0] = 0
		if not dict_params['FLR'][1]:
			dict_params['flr_order'][1] = 0
		if not dict_params['FLR'][0] and not dict_params['FLR'][1]:
			dict_params['rho'] = 0
		if dict_params['gc_order'] == 1:
			dict_params['eta'] = 0
		for key in dict_params:
			setattr(self, key, dict_params[key])
		self.DictParams = dict_params
		xp.random.seed(27)
		phases = 2.0 * xp.pi * xp.random.random((self.M, self.M))
		n = xp.meshgrid(xp.arange(1, self.M+1), xp.arange(1, self.M+1), indexing='ij')
		self.xv = xp.linspace(0, 2.0 * xp.pi, self.N, endpoint=False, dtype=xp.float64)
		self.xv_ = xp.linspace(0, 2.0 * xp.pi, self.N + 1, dtype=xp.float64)
		nm = xp.meshgrid(fftfreq(self.N, d=1/self.N), fftfreq(self.N, d=1/self.N), indexing='ij')
		fft_phi = xp.zeros((self.N, self.N), dtype=xp.complex128)
		fft_phi[1:self.M+1, 1:self.M+1] = (self.A / (n[0] ** 2 + n[1] ** 2) ** 1.5).astype(xp.complex128) * xp.exp(1j * phases)
		fft_phi[nm[0] ** 2 + nm[1] ** 2 > self.M **2] = 0.0
		if self.flr_order[0] in range(2):
			flr1_coeff = 1.0
		elif self.flr_order[0] == 'all':
			flr1_coeff = jv(0, self.rho * xp.sqrt(nm[0] ** 2 + nm[1] ** 2))
		else:
			x = sp.Symbol('x')
			flr_expansion = sp.besselj(0, x).series(x, 0, self.flr_order[0] + 1).removeO()
			flr_func = sp.lambdify(x, flr_expansion)
			flr1_coeff = flr_func(self.rho * xp.sqrt(nm[0] ** 2 + nm[1] ** 2))
		fft_phi_ = flr1_coeff * fft_phi
		self.phi = ifft2(fft_phi) * (self.N ** 2)
		self.phi_ = ifft2(fft_phi_) * (self.N ** 2)
		self.dphidx = ifft2(1j * nm[0] * fft_phi) * (self.N ** 2)
		self.dphidy = ifft2(1j * nm[1] * fft_phi) * (self.N ** 2)
		if self.flr_order[1] in range(3):
			self.phi_gc2_0 = - self.eta * (xp.abs(self.dphidx) ** 2 + xp.abs(self.dphidy) ** 2) / 2.0
			self.phi_gc2_2 = - self.eta * (self.dphidx ** 2 + self.dphidy ** 2) / 2.0
		else:
			if self.flr_order[1] == 'all':
				flr2_coeff = - xp.sqrt(nm[0] ** 2 + nm[1] ** 2) * jv(1, self.rho * xp.sqrt(nm[0] ** 2 + nm[1] ** 2)) / self.rho
			else:
				x = sp.Symbol('x')
				flr_expansion = sp.besselj(1, x).series(x, 0, self.flr_order[1] + 1).removeO()
				flr_func = sp.lambdify(x, flr_expansion)
				flr2_coeff = - xp.sqrt(nm[0] ** 2 + nm[1] ** 2) * flr_func(self.rho * xp.sqrt(nm[0] ** 2 + nm[1] ** 2)) / self.rho
			self.flr2 = lambda psi: ifft2(fft2(psi) * flr2_coeff)
			self.phi_gc2_0 = - self.eta * (self.flr2(xp.abs(self.phi) **2) - self.phi_ * self.flr2(self.phi.conjugate()) - self.phi_.conjugate() * self.flr2(self.phi)).real / 2.0
			self.phi_gc2_2 = - self.eta * (self.flr2(self.phi ** 2) - 2.0 * self.phi_ * self.flr2(self.phi)) / 2.0
		self.dphidx_gc1_1 = xp.pad(ifft2(1j * nm[0] * fft_phi_) * (self.N ** 2), ((0, 1),), mode='wrap')
		self.dphidy_gc1_1 = xp.pad(ifft2(1j * nm[1] * fft_phi_) * (self.N ** 2), ((0, 1),), mode='wrap')
		self.dphidx_gc2_0 = xp.pad(ifft2(1j * nm[0] * fft2(self.phi_gc2_0)), ((0, 1),), mode='wrap')
		self.dphidy_gc2_0 = xp.pad(ifft2(1j * nm[1] * fft2(self.phi_gc2_0)), ((0, 1),), mode='wrap')
		self.dphidx_gc2_2 = xp.pad(ifft2(1j * nm[0] * fft2(self.phi_gc2_2)), ((0, 1),), mode='wrap')
		self.dphidy_gc2_2 = xp.pad(ifft2(1j * nm[1] * fft2(self.phi_gc2_2)), ((0, 1),), mode='wrap')

	def eqn_phi(self, t, y):
		yr = xp.array([y[:self.Ntraj], y[self.Ntraj:]]).transpose() % (2.0 * xp.pi)
		dphidx = interpn((self.xv_, self.xv_), self.dphidx_gc1_1, yr)
		dphidy = interpn((self.xv_, self.xv_), self.dphidy_gc1_1, yr)
		dy_gc1 = xp.concatenate((- (dphidy * xp.exp(- 1j * t)).imag, (dphidx * xp.exp(- 1j * t)).imag), axis=None)
		if self.gc_order == 1:
			return dy_gc1
		elif self.gc_order == 2:
			dphidx_0 = interpn((self.xv_, self.xv_), self.dphidx_gc2_0, yr)
			dphidy_0 = interpn((self.xv_, self.xv_), self.dphidy_gc2_0, yr)
			dphidx_2 = interpn((self.xv_, self.xv_), self.dphidx_gc2_2, yr)
			dphidy_2 = interpn((self.xv_, self.xv_), self.dphidy_gc2_2, yr)
			dy_gc2 = xp.concatenate((- dphidy_0.real + (dphidy_2 * xp.exp(- 2j * t)).real, dphidx_0.real - (dphidx_2 * xp.exp(- 2j * t)).real), axis=None)
			return dy_gc1 + dy_gc2

	def save_data(self, name, data, filestr, info=[]):
		if self.save_results:
			mdic = self.DictParams.copy()
			mdic.update({'data': data, 'info': info})
			date_today = date.today().strftime(" %B %d, %Y\n")
			mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
			name_file = type(self).__name__ + '_' + name + '_' + filestr + '.mat'
			savemat(name_file, mdic)
			print('\033[92m    Results saved in {} \033[00m'.format(name_file))

if __name__ == "__main__":
	main()
