import numpy as xp
from numpy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import interpn
from scipy.special import jv, eval_chebyu
import sympy as sp
from gc2d_modules import run_method
from gc2d_dict import dict_list

def main():
	for dict in dict_list:
		if dict['potential'] == 'KMdCN':
			case = GC2Dk(dict)
		elif dict['potential'] == 'turbulent':
			case = GC2Dt(dict)
		run_method(case)


class GC2Dt:
	def __repr__(self):
		return '{self.__class__.__name__}({self.DictParams})'.format(self=self)

	def __str__(self):
		return '2D Guiding Center ({self.__class__.__name__}) for the turbulent potential with FLR = {self.flr} and GC order = {self.gc_order}'.format(self=self)

	def __init__(self, dict):
		for key in dict:
			setattr(self, key, dict[key])
		self.DictParams = dict
		xp.random.seed(27)
		phases = 2.0 * xp.pi * xp.random.random((self.M, self.M))
		n = xp.meshgrid(xp.arange(1, self.M+1), xp.arange(1, self.M+1), indexing='ij')
		self.xv = xp.linspace(0, 2.0 * xp.pi, self.N, endpoint=False, dtype=xp.float64)
		self.xv_ = xp.linspace(0, 2.0 * xp.pi, self.N + 1, dtype=xp.float64)
		nm = xp.meshgrid(fftfreq(self.N, d=1/self.N), fftfreq(self.N, d=1/self.N), indexing='ij')
		sqrt_nm = xp.sqrt(nm[0] ** 2 + nm[1] ** 2)
		fft_phi = xp.zeros((self.N, self.N), dtype=xp.complex128)
		fft_phi[1:self.M+1, 1:self.M+1] = (self.A / (n[0] ** 2 + n[1] ** 2) ** 1.5).astype(xp.complex128) * xp.exp(1j * phases)
		fft_phi[sqrt_nm > self.M] = 0.0
		if (self.flr[0] == 'none') or (self.flr[0] in range(2)):
			flr1_coeff = 1.0
		elif self.flr[0] == 'all':
			flr1_coeff = jv(0, self.rho * sqrt_nm)
		elif isinstance(self.flr[0], int):
			x = sp.Symbol('x')
			flr_expansion = sp.besselj(0, x).series(x, 0, self.flr[0] + 1).removeO()
			flr_func = sp.lambdify(x, flr_expansion)
			flr1_coeff = flr_func(self.rho * sqrt_nm)
		fft_phi_ = flr1_coeff * fft_phi
		self.phi = ifft2(fft_phi) * (self.N ** 2)
		self.phi_ = ifft2(fft_phi_) * (self.N ** 2)
		self.dphidx = ifft2(1j * nm[0] * fft_phi) * (self.N ** 2)
		self.dphidy = ifft2(1j * nm[1] * fft_phi) * (self.N ** 2)
		if (self.flr[1] == 'none') or (self.flr[1] in range(3)):
			self.phi_gc2_0 = - self.eta * (xp.abs(self.dphidx) ** 2 + xp.abs(self.dphidy) ** 2) / 2.0
			self.phi_gc2_2 = - self.eta * (self.dphidx ** 2 + self.dphidy ** 2) / 2.0
		else:
			if self.flr[1] == 'all':
				flr2_coeff = - sqrt_nm * jv(1, self.rho * sqrt_nm) / self.rho
			elif isinstance(self.flr[1], int):
				x = sp.Symbol('x')
				flr_exp = sp.besselj(1, x).series(x, 0, self.flr[1] + 1).removeO()
				flr2_coeff = - sqrt_nm * sp.lambdify(x, flr_exp)(self.rho * sqrt_nm) / self.rho
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


class GC2Dk:
	def __repr__(self):
		return '{self.__class__.__name__}({self.DictParams})'.format(self=self)

	def __str__(self):
		return '2D Guiding Center ({self.__class__.__name__}) for the Kryukov-Martinell-delCastilloNegrete potential with FLR = {self.flr} and GC order = {self.gc_order}'.format(self=self)

	def __init__(self, dict):
		for key in dict:
			setattr(self, key, dict[key])
		self.DictParams = dict
		if (self.flr[0] == 'none') or (self.flr[0] in range(2)):
			flr1_coeff = 1.0
		elif self.flr[0] == 'all':
			flr1_coeff = jv(0, self.rho * xp.sqrt(2.0))
		elif isinstance(self.flr[0], int):
			x = sp.Symbol('x')
			flr_exp = sp.besselj(0, x).series(x, 0, self.flr[0] + 1).removeO()
			flr1_coeff = sp.lambdify(x, flr_exp)(self.rho * xp.sqrt(2.0))
		self.A1 = self.A * flr1_coeff
		if (self.flr[1] == 'none') or (self.flr[1] in range(2)):
			flr2_coeff20 = 0.0
			flr2_coeff22 = - 0.25
		elif self.flr[1] == 'all':
			flr2_coeff20 = - 2.0 * (jv(1, self.rho * 2.0) - xp.sqrt(2.0) *  jv(0, self.rho * xp.sqrt(2.0)) * jv(1, self.rho * xp.sqrt(2.0))) / self.rho
			flr2_coeff22 = - xp.sqrt(2.0) * (jv(1, self.rho * 2.0 * xp.sqrt(2.0)) - jv(0, self.rho * xp.sqrt(2.0)) * jv(1, self.rho * xp.sqrt(2.0))) / self.rho
		elif isinstance(self.flr[1], int):
			x = sp.Symbol('x')
			flr2_exp20 = - 2 * ((sp.besselj(1, 2 * x) - sp.sqrt(2) * sp.besselj(0, sp.sqrt(2) * x) * sp.besselj(1, sp.sqrt(2) * x)) / x).series(x, 0, self.flr[1] + 1).removeO()
			flr2_exp22 = - sp.sqrt(2) * ((sp.besselj(1, 2 * sp.sqrt(2) * x) - sp.besselj(0, sp.sqrt(2) * x) * sp.besselj(1, sp.sqrt(2) * x)) / x).series(x, 0, self.flr[1] + 1).removeO()
			flr2_coeff20 = sp.lambdify(x, flr2_exp20)(self.rho)
			flr2_coeff22 = sp.lambdify(x, flr2_exp22)(self.rho)
		self.A20 = - (self.A ** 2) * self.eta * flr2_coeff20
		self.A22 = - (self.A ** 2) * self.eta * flr2_coeff22

	def eqn_phi(self, t, y):
		cheby_coeff = eval_chebyu(self.M - 1, xp.cos(t))
		alpha_b = 0.5 + (xp.cos((self.M + 1) * t) + xp.cos(self.M * t)) * cheby_coeff
		beta_b = 0.5 + (xp.cos((self.M + 1) * t) - xp.cos(self.M * t)) * cheby_coeff
		smxy = xp.sin(y[:self.Ntraj] - y[self.Ntraj:])
		spxy = xp.sin(y[:self.Ntraj] + y[self.Ntraj:])
		dy_gc1 = self.A1 * xp.array([- alpha_b * smxy + beta_b * spxy, - alpha_b * smxy - beta_b * spxy])
		if self.gc_order == 1:
			return dy_gc1.reshape(2 * self.Ntraj)
		elif self.gc_order == 2:
			v20 = self.A20 * alpha_b * beta_b * xp.sin(2.0 * y)
			v2m2 = self.A22 * (alpha_b ** 2) * xp.sin(2.0 * (y[:self.Ntraj] - y[self.Ntraj:]))
			v2p2 = self.A22 * (beta_b ** 2) * xp.sin(2.0 * (y[:self.Ntraj] + y[self.Ntraj:]))
			dy_gc2 = 2.0 * xp.array([v20[self.Ntraj:] + v2p2 - v2m2, - v20[:self.Ntraj] - v2p2 - v2m2])
			return (dy_gc1 + dy_gc2).reshape(2 * self.Ntraj)


if __name__ == "__main__":
	main()
