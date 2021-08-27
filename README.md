# Guiding-Center (GC) theory in plasma physics

- **gc2d_dict.py**: to be edited to change the parameters of the GC computation (see below for a dictionary of parameters)

- **gc2d.py**: contains the GC classes and main functions defining the GC dynamics

- **gc2d_modules**: contains the methods to integrate the GC dynamics

Once gc2d_dict.py has been edited with the relevant parameters, run the file as 
> `python3.8 gc2d.py`

___
##  Parameter dictionary

- *Potential*: string; 'KMdCN' or 'turbulent' 
- *Method*: string; 'plot_potentials' (only for 'turbulent'), 'diffusion', 'poincare'
- *FLR*: array of length 2; 'none', 'all' or integer; FLR order for each GC order
- *GCorder*: 1 or 2; order in the guiding-center expansion 
- *A*: float; amplitude of the electrostatic potential 
- *rho*: float; value of the Larmor radius 
- *eta*: float; amplitude of the GC order 2 potential
- *M*: integer; number of modes (default = 5 for 'KMdCN' and 25 for 'turbulent')
- *N*: integer; number of points on each axis for 'turbulent' (default = 2 ** 10) 
- *Ntraj*: integer; number of trajectories to be integrated
- *Tf*: integer; number of periods for the integration of the trajectories
- *init*: boolean; 'random' or 'fixed'; method to generate initial conditions  
- *modulo*: boolean; only for method='poincare'; if True, *x* and *y* are taken modulo 2 x pi
- *TimeStep*: float; time step used by the integrator
- *SaveData*: boolean; if True, the results are saved in a `.mat` file
- *PlotResults*: boolean; if True, the results are plotted right after the computation
- *Parallelization*: 2d array [boolean, int]; True for parallelization, int is the number of processors to be used or int='all' to use all available processors

---
For more information: <cristel.chandre@univ-amu.fr>
