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
from numpy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import interpn
from gc2d_modules import run_method
from gc2d_dict import dict_list
from pyhamsys import SymplecticIntegrator

def run_case(dict) -> None:
	case = GC2Dt(dict)
	run_method(case)

def main() -> None:
	for dict in dict_list:
		run_case(dict)
	plt.show()

class GC2Dt:
	def __repr__(self) -> str:
		return "{self.__class__.__name__}({self.DictParams})".format(self=self)

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for the turbulent potential'

	def __init__(self, dict_: dict) -> None:
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
		fft_phi[1:self.M+1, 1:self.M+1] = (self.A / (n[0]**2 + n[1]**2)** 1.5).astype(xp.complex128) * xp.exp(1j * self.phases)
		fft_phi[sqrt_nm > self.M] = 0
		self.fft_phi_sml = fft_phi[:self.M+1, :self.M+1]
		self.phi = ifft2(fft_phi) * (self.N**2)
		self.pad = lambda psi: xp.pad(psi, ((0, 1),), mode='wrap')
		self.derivs = lambda psi: [self.pad(ifft2(1j * nm[_] * fft2(psi))) for _ in range(2)]
		self.phi_gc1_1 = ifft2(fft_phi) * (self.N**2)
		self.dim = 2
		stack = self.derivs(self.phi_gc1_1)
		if self.check_energy:
			stack = (*stack, self.pad(self.phi_gc1_1))
		self.Dphi = xp.moveaxis(xp.stack(stack), 0, -1)
		if self.solve_method == 'symp':
			self.integrator = lambda step: SymplecticIntegrator(self.ode_solver, step)

	def chi(self, h:float, y) -> xp.ndarray:
		fft_coeffs = h * self.fft_phi_sml
		for n in range(1, self.M + 1):
			for m in range(1, self.M + 1):
				dy = (fft_coeffs[n, m] * xp.exp(1j * (n * y[1] + m * y[2] - y[0]))).real 
				y[1] -= m * dy
				y[2] += n * dy
		y[0] += h
		return y
	
	def chi_star(self, h:float, y) -> xp.ndarray:
		fft_coeffs = h * self.fft_phi_sml
		y[0] += h
		for m in range(self.M + 1, 1, -1):
			for n in range(self.M + 1, 1, -1):
				dy = (fft_coeffs[n, m] * xp.exp(1j * (n * y[1] + m * y[2] - y[0]))).real
				y[1] -= m * dy
				y[2] += n * dy
		return y

	def eqn_interp(self, t:float, y:xp.ndarray) -> xp.ndarray:
		vars = xp.split(y, self.dim)
		r = xp.moveaxis(xp.asarray(vars[:2]) % (2 * xp.pi), 0, -1)
		fields = xp.moveaxis(interpn(self.xy_, self.Dphi, r), 0, 1)
		dphidx, dphidy = fields[:2]
		dy_gc = xp.concatenate((-(dphidy * xp.exp(-1j * t)).imag, (dphidx * xp.exp(-1j * t)).imag), axis=None)
		if not self.check_energy:
			return dy_gc
		phi = fields[2]
		dk = (phi * xp.exp(-1j * t)).real
		return xp.concatenate((dy_gc, dk), axis=None)

	def compute_energy(self, t:float, *sol, method:str='interp') -> xp.ndarray:
		if method == 'interp':
			r = xp.moveaxis(xp.asarray(sol[:2]) % (2 * xp.pi), 0, -1)
			k = sol[-1]
			phi_1 = interpn(self.xy_, self.pad(self.phi_gc1_1), r)
			h = k + (phi_1 * xp.exp(-1j * t)).imag
			return h
		else:
			nm = xp.meshgrid(fftfreq(self.M, d=1/self.M), fftfreq(self.M, d=1/self.M), indexing='ij')
			return xp.sum(self.fft_phi_sml * xp.exp(nm[0] * sol[0].reshape(-1, 1, 1) + nm[1] *sol[1].reshape(-1, 1, 1) - t), (1, 2)).imag

if __name__ == '__main__':
	main()
