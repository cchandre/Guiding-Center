import numpy as xp
from scipy.integrate import solve_ivp
from scipy.special import jv, eval_chebyu
import sympy as sp
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from datetime import date
from dict_gc2d_s import dict_list
from gc2d import save_data, run_method

def main():
	for dict_params in dict_list:
		case = GC2Ds(dict_params)
		run_method(case)


class GC2Ds:
	def __repr__(self):
		return '{self.__class__.__name__}({self.DictParams})'.format(self=self)

	def __str__(self):
		return '2D Guiding Center ({self.__class__.__name__}) for the Kryukov-Martinell-delCastilloNegrete potential with FLR = {self.flr_order} and GC order = {self.gc_order}'.format(self=self)

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
		if self.flr_order[0] in range(2):
			flr1_coeff = 1.0
		elif self.flr_order[0] == 'all':
			flr1_coeff = jv(0, self.rho * xp.sqrt(2.0))
		else:
			x = sp.Symbol('x')
			flr_exp = sp.besselj(0, x).series(x, 0, self.flr_order[0] + 1).removeO()
			flr1_coeff = sp.lambdify(x, flr_exp)(self.rho * xp.sqrt(2.0))
		self.A1 = self.A * flr1_coeff
		if self.flr_order[1] in range(2):
			flr2_coeff20 = 0.0
			flr2_coeff22 = - 0.25
		elif self.flr_order[1] == 'all':
			flr2_coeff20 = - 2.0 * (jv(1, self.rho * 2.0) - xp.sqrt(2.0) *  jv(0, self.rho * xp.sqrt(2.0)) * jv(1, self.rho * xp.sqrt(2.0))) / self.rho
			flr2_coeff22 = - xp.sqrt(2.0) * (jv(1, self.rho * 2.0 * xp.sqrt(2.0)) - jv(0, self.rho * xp.sqrt(2.0)) * jv(1, self.rho * xp.sqrt(2.0))) / self.rho
		else:
			x = sp.Symbol('x')
			flr2_exp20 = - 2 * ((sp.besselj(1, 2 * x) - sp.sqrt(2) * sp.besselj(0, sp.sqrt(2) * x) * sp.besselj(1, sp.sqrt(2) * x)) / x).series(x, 0, self.flr_order[1] + 1).removeO()
			flr2_exp22 = - sp.sqrt(2) * ((sp.besselj(1, 2 * sp.sqrt(2) * x) - sp.besselj(0, sp.sqrt(2) * x) * sp.besselj(1, sp.sqrt(2) * x)) / x).series(x, 0, self.flr_order[1] + 1).removeO()
			flr2_coeff20 = sp.lambdify(x, flr2_exp20)(self.rho)
			flr2_coeff22 = sp.lambdify(x, flr2_exp22)(self.rho)
		self.A20 = - self.A ** 2 * self.eta * flr2_coeff20
		self.A22 = - self.A ** 2 * self.eta * flr2_coeff22

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
