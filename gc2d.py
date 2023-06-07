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
from numpy.fft import fft2, ifft2, ifftn, fftfreq
from scipy.interpolate import interpn
from scipy.special import jv, eval_chebyu
from gc2d_modules import run_method
from gc2d_dict import dict_list, Parallelization
import multiprocessing
from typing import Tuple

def run_case(dict) -> None:
	if dict['Potential'] == 'KMdCN':
		case = GC2Dk(dict)
	elif dict['Potential'] == 'turbulent':
		case = GC2Dt(dict)
	run_method(case)

def main() -> None:
	if Parallelization[0]:
		if Parallelization[1] == 'all':
			num_cores = multiprocessing.cpu_count()
		else:
			num_cores = min(multiprocessing.cpu_count(), Parallelization[1])
		pool = multiprocessing.Pool(num_cores)
		pool.map(run_case, dict_list)
	else:
		for dict in dict_list:
			run_case(dict)
	plt.show()

class GC2Dt:
	def __repr__(self) -> str:
		return "{self.__class__.__name__}({self.DictParams})".format(self=self)

	def __str__(self) -> str:
		if self.Method.endswith('_fo'):
			return f'2D Guiding Center ({self.__class__.__name__}) for the turbulent potential for the full orbits'
		elif self.Method.endswith('_gc') or self.Method == 'potentials':
			return f'2D Guiding Center ({self.__class__.__name__}) for the turbulent potential with FLR = {self.FLR} and GC order = {self.GCorder}'
		
	def __init__(self, dict_:dict) -> None:
		for key in dict_:
			setattr(self, key, dict_[key])
		self.DictParams = dict_
		xp.random.seed(27)
		self.phases = 2 * xp.pi * xp.random.random((self.M, self.M))
		n = xp.meshgrid(xp.arange(1, self.M+1), xp.arange(1, self.M+1), indexing='ij')
		self.xy_ = 2 * (xp.linspace(0, 2 * xp.pi, self.N+1, dtype=xp.float64),)
		nm = xp.meshgrid(fftfreq(self.N, d=1/self.N), fftfreq(self.N, d=1/self.N), indexing='ij')
		sqrt_nm = xp.sqrt(nm[0]**2 + nm[1]**2)
		self.elts_nm = sqrt_nm, xp.angle(nm[0] + 1j * nm[1])
		fft_phi = xp.zeros((self.N, self.N), dtype=xp.complex128)
		fft_phi[1:self.M+1, 1:self.M+1] = (self.A / (n[0]**2 + n[1]**2)**1.5).astype(xp.complex128) * xp.exp(1j * self.phases)
		fft_phi[sqrt_nm > self.M] = 0
		self.phi = ifft2(fft_phi) * (self.N**2)
		self.pad = lambda psi: xp.pad(psi, ((0, 1),), mode='wrap')
		self.derivs = lambda psi: [self.pad(ifft2(1j * nm[_] * fft2(psi))) for _ in range(2)]
		if self.FLR[0] == 'all':
			flr1_coeff = jv(0, self.rho * sqrt_nm)
		elif self.FLR[0] == 'pade':
			flr1_coeff = 1 / (1 + self.rho**2 * sqrt_nm**2 / 4)
		else:
			flr1_coeff = 1
		self.phi_gc1_1 = ifft2(flr1_coeff * fft_phi) * (self.N**2)
		if self.Method.endswith('_fo'):
			self.dim = 4
			stack = self.derivs(self.phi)
			if self.check_energy:
				self.dim += 1
				stack = (*stack, self.pad(self.phi))
		else:
			self.dim = 2
			if self.GCorder == 1:
				stack = self.derivs(self.phi_gc1_1)
				if self.check_energy:
					stack = (*stack, self.pad(self.phi_gc1_1))
			elif self.GCorder == 2:
				if self.FLR[1] == 'all' and (self.rho != 0):
					flr2_coeff = -sqrt_nm * jv(1, self.rho * sqrt_nm) / self.rho
				elif self.FLR[1] == 'pade' and (self.rho != 0):
					flr2_coeff = -(sqrt_nm**2 / 2) / (1 + self.rho**2 * sqrt_nm**2 / 8)
				else:
					flr2_coeff = -sqrt_nm**2 / 2
				self.flr2 = lambda psi: ifft2(fft2(psi) * flr2_coeff)
				self.phi_gc2_0 = self.eta * (2 * self.phi_gc1_1 * self.flr2(self.phi.conj()) - self.flr2(xp.abs(self.phi)**2)).real / 2
				self.phi_gc2_2 = self.eta * (2 * self.phi_gc1_1 * self.flr2(self.phi) - self.flr2(self.phi**2)) / 2
				stack = (*self.derivs(self.phi_gc1_1), *self.derivs(self.phi_gc2_0), *self.derivs(self.phi_gc2_2))
				if self.check_energy:
					self.dim += 1
					stack += (self.pad(self.phi_gc1_1), self.pad(self.phi_gc2_2))
		self.Dphi = xp.moveaxis(xp.stack(stack), 0, -1)

	def chi(self, h:float, y):
		for n in range(1, self.M + 1):
			for m in range(1, self.M + 1):
				if n**2 + m**2 <= self.M**2:
					dy = h * self.A / (n**2 + m**2)**1.5 * xp.cos(n * y[1] + m * y[2] + self.phases[n, m] - y[0])
					y[1] -= m * dy
					y[2] += n * dy
		y[0] += h
		return y

	def chi_star(self, h:float, y):
		y[0] += h
		for m in range(self.M + 1, 1, -1):
			for n in range(self.M + 1, 1, -1):
				if n**2 + m**2 <= self.M**2:
					dy = h * self.A / (n**2 + m**2)**1.5 * xp.cos(n * y[1] + m * y[2] + self.phases[n, m] - y[0])
					y[1] -= m * dy
					y[2] += n * dy
		return y

	def eqn(self, t:float, y:xp.ndarray) -> xp.ndarray:
		vars = xp.split(y, self.dim)
		r = xp.moveaxis(xp.asarray(vars[:2]) % (2 * xp.pi), 0, -1)
		fields = xp.moveaxis(interpn(self.xy_, self.Dphi, r), 0, 1)
		dphidx, dphidy = fields[:2]
		if self.Method.endswith('_gc'):
			dy_gc = xp.concatenate((-(dphidy * xp.exp(-1j * t)).imag, (dphidx * xp.exp(-1j * t)).imag), axis=None)
			if self.GCorder == 1:
				if not self.check_energy:
					return dy_gc
				phi = fields[2]
				dk = (phi * xp.exp(-1j * t)).real
				return xp.concatenate((dy_gc, dk), axis=None)
			dphidx_0, dphidy_0, dphidx_2, dphidy_2 = fields[2:6]
			dy_gc += xp.concatenate((-dphidy_0.real + (dphidy_2 * xp.exp(-2j * t)).real, dphidx_0.real - (dphidx_2 * xp.exp(-2j * t)).real), axis=None)
			if self.GCorder == 2:
				if not self.check_energy:
					return dy_gc
				phi1, phi2 = fields[6:8]
				dk = (phi1 * xp.exp(-1j * t)).real + 2 * (phi2 * xp.exp(-2j * t)).imag
				return xp.concatenate((dy_gc, dk), axis=None)
			raise ValueError(f'GCorder={self.GCorder} not currently implemented')
		elif self.Method.endswith('_fo'):
			if self.eta == 0 or self.rho == 0:
				raise ValueError('Eta or Rho cannot be zero for full orbits')
			vx, vy = vars[2:4]
			dvx = -(dphidx * xp.exp(-1j * t)).imag / self.rho * xp.sign(self.eta) + vy / (2 * self.eta)
			dvy = -(dphidy * xp.exp(-1j * t)).imag / self.rho * xp.sign(self.eta) - vx / (2 * self.eta)
			d_ = xp.concatenate((self.rho / (2 * xp.abs(self.eta)) * vx, self.rho / (2 * xp.abs(self.eta)) * vy, dvx, dvy), axis=None)
			if not self.check_energy:
				return d_
			phi = fields[2]
			dk = (phi * xp.exp(-1j * t)).real / (2 * self.eta)
			return xp.concatenate((d_, dk), axis=None)

	def compute_energy(self, t:float, *sol) -> xp.ndarray:
		r = xp.moveaxis(xp.asarray(sol[:2]) % (2 * xp.pi), 0, -1)
		k = sol[-1]
		if self.dim <= 3:
			phi_1 = interpn(self.xy_, self.pad(self.phi_gc1_1), r)
			h = k + (phi_1 * xp.exp(-1j * t)).imag
			if self.GCorder == 1:
				return h
			elif self.GCorder == 2:
				phi_0 = interpn(self.xy_, self.pad(self.phi_gc2_0), r)
				phi_2 = interpn(self.xy_, self.pad(self.phi_gc2_2), r)
				h += phi_0 - (phi_2 * xp.exp(-2j * t)).real
				return h
			raise ValueError(f'GCorder={self.GCorder} not currently implemented')
		else:
			vx, vy = sol[2:4]
			h = k + self.rho**2 / (8 * self.eta**2) * (vx**2 + vy**2) + (interpn(self.xy_, self.pad(self.phi), r) * xp.exp(-1j * t)).imag / (2 * self.eta)
			return h

	def fo2gc(self, t:float, *sol, order:int=1) -> Tuple[xp.ndarray, xp.ndarray]:
		x, y, vx, vy = sol[:4]
		v = vy + 1j * vx
		theta, rho = xp.pi + xp.angle(v), self.rho * xp.abs(v)
		x_gc, y_gc = x - rho * xp.cos(theta), y + rho * xp.sin(theta)
		if order <= 1:
			return x_gc, y_gc
		grid, s1 = -self.antiderivative(self.phi, rho)
		ds1dx, ds1dy = self.derivs(s1)
		r_gc = xp.moveaxis(xp.asarray((x_gc, y_gc, theta)) % (2 * xp.pi), 0, -1)
		x_gc -= 2 * self.eta * interpn(grid, (ds1dy * xp.exp(-1j * t)).imag, r_gc)
		y_gc += 2 * self.eta * interpn(grid, (ds1dx * xp.exp(-1j * t)).imag, r_gc)
		if order == 2:
			return x_gc, y_gc
		raise ValueError(f'fo2gc not available at order {order}')

	def compute_mu(self, t:float, *sol, order:int=1) -> xp.ndarray:
		x, y, vx, vy = sol[:4]
		r = xp.moveaxis(xp.asarray((x, y)) % (2 * xp.pi), 0, -1)
		mu = self.rho**2 * (vx**2 + vy**2) / 2
		if order == 0:
			return mu
		r_gc = xp.moveaxis(xp.asarray(self.fo2gc(t, *sol, order=1)) % (2 * xp.pi), 0, -1)
		phi_c = interpn(self.xy_, self.pad(self.phi), r)
		mu += 2 * self.eta * ((phi_c - interpn(self.xy_, self.pad(self.phi_gc1_1), r_gc)) * xp.exp(-1j * t)).imag
		if order == 1:
			return mu
		phi_gc = interpn(self.xy_, self.pad(self.flr2(self.phi)), r_gc)
		phi_2_0 = 2 * phi_c * phi_gc.conj() - interpn(self.xy_, self.pad(self.flr2(xp.abs(self.phi)**2)), r_gc)
		phi_2_2 = 2 * phi_c * phi_gc - interpn(self.xy_, self.pad(self.flr2(self.phi**2)), r_gc)
		mu += self.eta**2 * (phi_2_2 * xp.exp(-2j * t) - phi_2_0).real
		if order == 2:
			return mu
		raise ValueError(f'compute_mu not available at order {order}')

	def antiderivative(self, phi:xp.ndarray, rho:float, N:int=2**8) -> xp.ndarray:
		n = xp.arange(1, N//2 + 1)
		jvn = xp.moveaxis(xp.asarray([jv(n_, rho * self.elts_nm[0]) for n_ in n]), 0, -1)
		ja = 1j**n * jvn / (1j * n) * xp.exp(1j * n * self.elts_nm[1].reshape(self.N, self.N, 1))
		ja = xp.concatenate((xp.zeros((self.N, self.N, 1)), ja[:, :, :N//2 - 1], xp.flip((-1)**n * ja.conj(), axis=2)), axis=2)
		return self.xy_ + (xp.linspace(0, 2 * xp.pi, N + 1, dtype=xp.float64),), ifftn(fft2(phi) * ja) * N


class GC2Dk:
	def __repr__(self) -> str:
		return f'{self.__class__.__name__}({self.DictParams})'

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for the KMdCN potential with FLR = {self.FLR} and GC order = {self.GCorder}'

	def __init__(self, dict_:dict) -> None:
		for key in dict_:
			setattr(self, key, dict_[key])
		self.DictParams = dict_
		if self.FLR[0] == 'all':
			flr1_coeff = jv(0, self.rho * xp.sqrt(2))
		elif self.FLR[0] == 'pade':
			flr1_coeff = 1 / (1 + self.rho**2 / 2)
		else:
			flr1_coeff = 1
		self.A1 = self.A * flr1_coeff
		if self.FLR[1] == 'all':
			flr2_coeff20 = -2 * (jv(1, self.rho * 2) - xp.sqrt(2) *  jv(0, self.rho * xp.sqrt(2)) * jv(1, self.rho * xp.sqrt(2))) / self.rho
			flr2_coeff22 = -xp.sqrt(2) * (jv(1, self.rho * 2 * xp.sqrt(2)) - jv(0, self.rho * xp.sqrt(2)) * jv(1, self.rho * xp.sqrt(2))) / self.rho
		else:
			flr2_coeff20, flr2_coeff22 = 0, -1
		self.A20 = -(self.A**2) * self.eta * flr2_coeff20
		self.A22 = -(self.A**2) * self.eta * flr2_coeff22

	def compute_coeffs(self, t:float) -> Tuple[float, float]:
		cheby_coeff = eval_chebyu(self.M-1, xp.cos(t))
		alpha = 0.5 + (xp.cos((self.M+1) * t) + xp.cos(self.M * t)) * cheby_coeff
		beta = 0.5 + (xp.cos((self.M+1) * t) - xp.cos(self.M * t)) * cheby_coeff
		return alpha, beta

	def eqn_gc(self, t, y:xp.ndarray) -> xp.ndarray:
		x_, y_ = xp.split(y, 2)
		alpha, beta = self.compute_coeffs(t)
		smxy, spxy = xp.sin(x_ - y_), xp.sin(x_ + y_)
		dy_gc1 = self.A1 * xp.concatenate((-alpha * smxy + beta * spxy, -alpha * smxy - beta * spxy), axis=None)
		if self.GCorder == 1:
			return dy_gc1
		elif self.GCorder == 2:
			v20 = xp.split(self.A20 * alpha * beta * xp.sin(2 * y), 2)
			v2m2 = self.A22 * (alpha**2) * xp.sin(2 * (x_ - y_))
			v2p2 = self.A22 * (beta**2) * xp.sin(2 * (x_ + y_))
			dy_gc2 = 2 * xp.concatenate((v20[1] + v2p2 - v2m2, -v20[0] - v2p2 - v2m2), axis=None)
			return dy_gc1 + dy_gc2

	def eqn_fo(self, t:float, y:xp.ndarray) -> xp.ndarray:
		if self.eta == 0 or self.rho == 0:
			raise ValueError('Eta or Rho cannot be zero for full orbits')
		x_, y_, vx, vy = xp.split(y, 4)
		alpha, beta = self.compute_coeffs(t)
		smxy, spxy = xp.sin(x_ - y_), xp.sin(x_ + y_)
		dphidx = -alpha * smxy - beta * spxy
		dphidy = alpha * smxy - beta * spxy
		dvx = -self.A * dphidx / self.rho * xp.sign(self.eta) + vy / (2 * self.eta)
		dvy = -self.A * dphidy / self.rho * xp.sign(self.eta) - vx / (2 * self.eta)
		return xp.concatenate((self.rho / (2 * xp.abs(self.eta)) * vx, self.rho / (2 * xp.abs(self.eta)) * vy, dvx, dvy), axis=None)

if __name__ == '__main__':
	main()
