# Guiding-Center (GC) dynamics in plasma physics

- [`gc2d_dict.py`](https://github.com/cchandre/Guiding-Center/blob/main/gc2d_dict.py): to be edited to change the parameters of the GC computation (see below for a dictionary of parameters)

- [`gc2d.py`](https://github.com/cchandre/Guiding-Center/blob/main/gc2d.py): contains the GC classes and main functions defining the GC dynamics

- [`gc2d_modules.py`](https://github.com/cchandre/Guiding-Center/blob/main/gc2d_modules.py): contains the methods to integrate the GC dynamics

Once [`gc2d_dict.py`](https://github.com/cchandre/Guiding-Center/blob/main/gc2d_dict.py) has been edited with the relevant parameters, run the file as 
```sh
python3 gc2d.py
```
or 
```sh
nohup python3 -u gc2d.py &>gc2d.out < /dev/null &
```
The list of Python packages and their version are specified in [`requirements.txt`](https://github.com/cchandre/Guiding-Center/blob/main/requirements.txt)
___
##  Parameter dictionary

- *Potential*: string; 'KMdCN' or 'turbulent' 
- *Method*: string
  - 'potentials' (only for Potential='turbulent'): plots the electrostatic potential as well as the first and second order guiding-center potentials
  - 'diffusion_ions': computes the diffusion coefficient for the ions
  - 'diffusion_gc': computes the diffusion coefficient for the guiding-center trajectories 
  - 'poincare_ions': plots the trajectories of the ions in the plane (*x*, *y*) for every period of the potential (stroboscopic plot)
  - 'poincare_gc': plots the guiding-center trajectories in the plane (*x*, *y*) for every period of the potential (stroboscopic plot)
####
- *FLR*: tuple of 2 strings; 'all', 'pade' or 'none'; if 'all', FLR to all orders is taken into account; if 'pade', a Padé approximant is considered for the FLR effects; if 'none', no FLR effects are taken into account 
####
- *A*: float or array of floats; amplitude(s) of the electrostatic potential [theory: *A*=&epsilon;<sub>&delta;</sub>/*B*]
- *rho*: float or array of floats; value(s) of the Larmor radius; for ions, this value corresponds to the thermal Larmor radius
- *eta*: float or array of floats; value(s) of the coefficient in front of the GC order 2 potential; &eta;>0 for positive charge, &eta;<0 for negative charge [theory: &eta;=1/(2&Omega;)] 
####
- *Ntraj*: integer; number of trajectories to be integrated
- *Tf*: integer; number of periods for the integration of the trajectories
- *threshold*: float; value used to discriminate between trapped and untrapped trajectories
- *TwoStepIntegration*: boolean; if True, computes trajectories from 0 to 2&pi;*T*<sub>mid</sub>, removes the trapped trajectories, and continues integration from 2&pi;*T*<sub>mid</sub> to 2&pi;*T*<sub>f</sub>
- *Tmid*: integer; number of periods for the integration of trajectories in the first step (if *TwoStepIntegration*=True)
- *TimeStep*: float; time step used by the integrator (recommended: 5x10<sup>-3</sup> for '_gc' and 5x10<sup>-4</sup> for '_ions')
- *check_energy*: boolean; if True, the autonomous system is integrated, and the output (`.mat` file) includes the total energy (only if *SaveData*=True)
- *init*: string; 'random' or 'fixed'; method to generate initial conditions  
####
- *SaveData*: boolean; if True, the results are saved in a `.mat` file; Poincaré sections and diffusion plots *r*<sup>2</sup>(*t*) are saved as `.png` files; NB: the diffusion data are saved in a `.txt` file regardless of the value of *SaveData*
- *PlotResults*: boolean; if True, the results are plotted right after the computation
- *Parallelization*: tuple (boolean, int); True for parallelization, int is the number of cores to be used or int='all' to use all available cores
####
- *modulo*: boolean; if True, *x* and *y* are represented modulo 2&pi; (only for Method='poincare' and PlotResults=True)
- *grid*: boolean; if True, show the grid lines on plots
- *dpi*: integer; dpi value for the figures
- *darkmode*: boolean; if True, plots are done in dark mode
####
- *M*: integer; number of modes (default = 5 for 'KMdCN' and 25 for 'turbulent') 
- *N*: integer; number of points on each axis for 'turbulent' (recommended: 2<sup>12</sup>)

---
For more information: <cristel.chandre@cnrs.fr>

<p align="center">
  <img src="https://github.com/cchandre/Guiding-Center/blob/main/A060_RHO040.gif" alt="Example" width="600"/>
</p>
