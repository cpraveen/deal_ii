# Steady viscous Burger's equation

Solves

```
u u_x - u_xx = s in (0,1)
```

where `s = s (x)` is some source term.

Choose s so that we have exact solution, e.g.,

if exact solution `u = 10x(1-x)sin(ax)` then get `s = u u_x - u_xx`

Choose exact solution and dx so that cell Peclet condition is satisfied

```
|u dx / mu| <= 2
```

(Here mu=1)

Numerical scheme: (TODO: CHECK)

* Galerkin method, with time marching to steady state
* Diffusion treated implicitly
* CG for linear solver

After running code, plot solution in gnuplot

```
gnuplot> load 'plot.gnu'
```
