# Guiding-Center (GC) theory in plasma physics

- [`gc2d_dict.py`](https://github.com/cchandre/Guiding-Center/blob/main/gc2d_dict.py): to be edited to change the parameters of the GC computation (see below for a dictionary of parameters)

- [`gc2d.py`](https://github.com/cchandre/Guiding-Center/blob/main/gc2d.py): contains the GC classes and main functions defining the GC dynamics

- [`gc2d_modules.py`](https://github.com/cchandre/Guiding-Center/blob/main/gc2d_modules.py): contains the methods to integrate the GC dynamics

Once [`gc2d_dict.py`](https://github.com/cchandre/Guiding-Center/blob/main/gc2d_dict.py) has been edited with the relevant parameters, run the file as 
```sh
python3 gc2d.py
```

___
##  Parameter dictionary

- *Potential*: string; 'KMdCN' or 'turbulent' 
- *Method*: string; 'plot_potentials' (only for 'turbulent'), 'diffusion', 'poincare'
####
- *FLR*: tuple of 2 elements; 'none', 'all' or integer; FLR order for each GC order
- *GCorder*: 1 or 2; order in the guiding-center expansion 
- *A*: float; amplitude of the electrostatic potential [theory: &epsilon;<sub>&delta;</sub>/B]
- *rho*: float; value of the Larmor radius 
- *eta*: float; coefficient in front of the GC order 2 potential; eta>0 for positive charge, eta<0 for negative charge [theory: 1/(2&Omega;)] 
- *M*: integer; number of modes (default = 5 for 'KMdCN' and 25 for 'turbulent') 
####
- *Ntraj*: integer; number of trajectories to be integrated
- *Tf*: integer; number of periods for the integration of the trajectories
- *init*: boolean; 'random' or 'fixed'; method to generate initial conditions  
- *modulo*: boolean; only for Method='poincare'; if True, *x* and *y* are represented modulo 2&pi;
- *threshold*: float; value used to discriminate between trapped and untrapped trajectories
- *N*: integer; number of points on each axis for 'turbulent' (default = 2 ** 10)
- *TimeStep*: float; time step used by the integrator
####
- *SaveData*: boolean; if True, the results are saved in a `.mat` file; Poincaré section saved as a `.png` figure
- *PlotResults*: boolean; if True, the results are plotted right after the computation
- *Parallelization*: tuple (boolean, int); True for parallelization, int is the number of cores to be used or int='all' to use all available cores

- *darkmode*: boolean; if True, plots are done in dark mode

---
For more information: <cristel.chandre@cnrs.fr>

<p align="center">
  <img src="https://github.com/cchandre/Guiding-Center/blob/main/A060_RHO040.gif" alt="Example" width="600"/>
</p>
