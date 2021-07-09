import numpy as xp
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.integrate import solve_ivp
from scipy.special import jv
from sympy import *
from scipy.io import savemat
import time
from datetime import date

def main():
    dict_params = {
        'N': 2 ** 10,
        'M': 25,
        'A': 0.8}
    dict_params.update({
        'FLR': True,
        'flr_order': 'all',
        'rho': 0.05,
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

    timestr = time.strftime("%Y%m%d_%H%M")
    case = GC2D(dict_params)
    if case.method == 'plot_potentials':
        data = xp.array([case.phi, case.phi_flr, case.phi_gc2_0, case.phi_gc2_2])
        case.save_data('potentials', data, timestr)
        if case.plot_results:
            plt.figure(figsize=(8, 8))
            plt.pcolor(case.xv, case.xv, case.phi.imag, shading='auto')
            plt.colorbar()
            plt.figure(figsize=(8, 8))
            plt.pcolor(case.xv, case.xv, case.phi_flr.imag, shading='auto')
            plt.colorbar()
            plt.figure(figsize=(8, 8))
            plt.pcolor(case.xv, case.xv, case.phi_gc2_0 + case.phi_gc2_2.real, shading='auto')
            plt.colorbar()
            plt.show()
    elif case.method == 'poincare':
        y0 = 2.0 * xp.pi * xp.random.rand(2 * case.Ntraj)
        t_eval = 2.0 * xp.pi * xp.arange(0, case.Tf)
        start = time.time()
        sol = solve_ivp(case.eqn_phi, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.timestep, atol=1, rtol=1)
        print('Computation finished in {} seconds'.format(int(time.time() - start)))
        if case.modulo:
            sol.y = sol.y % (2.0 * xp.pi)
        case.save_data('poincare', xp.array([sol.y[:case.Ntraj, :], sol.y[case.Ntraj:, :]]).transpose(), timestr)
        if case.plot_results:
            plt.figure(figsize=(8, 8))
            plt.plot(sol.y[:case.Ntraj, :], sol.y[case.Ntraj:, :], 'b.', markersize=2)
            plt.show()
    elif case.method == 'diffusion':
        y0 = 2.0 * xp.pi * xp.random.rand(2 * case.Ntraj)
        t_eval = 2.0 * xp.pi * xp.arange(0, case.Tf)
        start = time.time()
        sol = solve_ivp(case.eqn_phi, (0, t_eval.max()), y0, t_eval=t_eval, max_step=case.timestep, atol=1, rtol=1)
        print('Computation finished in {} seconds'.format(int(time.time() - start)))
        r2 = xp.zeros(case.Tf)
        for t in range(case.Tf):
            r2[t] += (xp.abs(sol.y[:, t:] - sol.y[:, :case.Tf-t]) ** 2).sum() / (case.Ntraj * (case.Tf - t))
        max_y = (xp.abs(sol.y[:case.Ntraj, :] - sol.y[:case.Ntraj, 0].reshape(case.Ntraj,1)) ** 2\
        + xp.abs(sol.y[case.Ntraj:, :] - sol.y[case.Ntraj:, 0].reshape(case.Ntraj,1)) ** 2).max(axis=1)
        tra = (max_y <= xp.pi).sum()
        case.save_data('diffusion', [t_eval, r2], timestr, info='trapped particles = {}'.format(tra))
        if case.plot_results:
            plt.figure(figsize=(8, 8))
            plt.plot(t_eval, r2, 'b', linewidth=2)
            plt.show()


class GC2D:
    def __repr__(self):
        return '{self.__class__.name__}({self.DictParams})'.format(self=self)

    def __str__(self):
        return '2D Guiding Center ({self.__class__.name__}) with FLR = {self.FLR} and GC order = {self.gc_order}'.format(self=self)

    def __init__(self, dict_params):
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

        fft_phi_flr = xp.zeros((self.N, self.N), dtype=xp.complex128)
        if self.flr_order == 'all':
            fft_phi_flr = jv(0, self.rho * xp.sqrt(nm[0] ** 2 + nm[1] ** 2)) * fft_phi
        else:
            x = Symbol('x')
            flr_expansion = besselj(0, x).series(x, 0, self.flr_order+1).removeO()
            flr_func = lambdify(x, flr_expansion)
            fft_phi_flr = flr_func(self.rho * xp.sqrt(nm[0] ** 2 + nm[1] ** 2)) * fft_phi

        self.phi = ifft2(fft_phi) * self.N**2
        self.phi_flr = ifft2(fft_phi_flr) * self.N**2
        self.dphidx = ifft2(1j * nm[0] * fft_phi) * self.N**2
        self.dphidy = ifft2(1j * nm[1] * fft_phi) * self.N**2
        self.dphidx_flr = xp.pad(ifft2(1j * nm[0] * fft_phi_flr) * self.N**2, (0, 1), mode='wrap')
        self.dphidy_flr = xp.pad(ifft2(1j * nm[1] * fft_phi_flr) * self.N**2, (0, 1), mode='wrap')
        self.phi_gc2_0 = - self.eta * self.A**2 * (xp.abs(self.dphidx) ** 2 + xp.abs(self.dphidy) ** 2) / 2.0
        self.phi_gc2_2 = - self.eta * self.A**2 * (self.dphidx ** 2 + self.dphidy ** 2) / 2.0
        self.dphidx = xp.pad(self.dphidx, (0, 1), mode='wrap')
        self.dphidy = xp.pad(self.dphidy, (0, 1), mode='wrap')
        self.dphidx_gc2_0 = xp.pad(ifft2(1j * nm[0] * fft2(self.phi_gc2_0)), (0, 1), mode='wrap')
        self.dphidy_gc2_0 = xp.pad(ifft2(1j * nm[1] * fft2(self.phi_gc2_0)), (0, 1), mode='wrap')
        self.dphidx_gc2_2 = xp.pad(ifft2(1j * nm[0] * fft2(self.phi_gc2_2)), (0, 1), mode='wrap')
        self.dphidy_gc2_2 = xp.pad(ifft2(1j * nm[1] * fft2(self.phi_gc2_2)), (0, 1), mode='wrap')

    def eqn_phi(self, t, y):
        yr = xp.array([y[:self.Ntraj], y[self.Ntraj:]]).transpose() % (2.0 * xp.pi)
        if self.FLR:
            dphidx_ = self.dphidx_flr
            dphidy_ = self.dphidy_flr
        else:
            dphidx_ = self.dphidx
            dphidy_ = self.dphidy
        DphiDx = interpn((self.xv_, self.xv_), dphidx_, yr)
        DphiDy = interpn((self.xv_, self.xv_), dphidy_, yr)
        dy = xp.concatenate((- (DphiDy * xp.exp(- 1j * t)).imag, (DphiDx * xp.exp(- 1j *t)).imag), axis=None)
        if self.gc_order == 1:
            return dy
        elif self.gc_order == 2:
            DphiDx_gc2_0 = interpn((self.xv_, self.xv_), self.dphidx_gc2_0, yr)
            DphiDy_gc2_0 = interpn((self.xv_, self.xv_), self.dphidy_gc2_0, yr)
            DphiDx_gc2_2 = interpn((self.xv_, self.xv_), self.dphidx_gc2_2, yr)
            DphiDy_gc2_2 = interpn((self.xv_, self.xv_), self.dphidy_gc2_2, yr)
            dy_gc2 = xp.concatenate((- DphiDy_gc2_0.real + (DphiDy_gc2_2 * xp.exp(- 2j * t)).real,
                                 DphiDx_gc2_0.real - (DphiDx_gc2_2 * xp.exp(- 2j * t)).real), axis=None)
            return dy + dy_gc2

    def save_data(self, name, data, timestr, info=[]):
        if self.save_results:
            mdic = self.DictParams.copy()
            mdic.update({'data': data, 'info': info})
            date_today = date.today().strftime(" %B %d, %Y\n")
            mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
            savemat(type(self).__name__ + '_' + name + '_' + timestr + '.mat', mdic)

if __name__ == "__main__":
	main()
