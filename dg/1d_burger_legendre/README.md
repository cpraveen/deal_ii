# 1-D Burger equation using DG

```bash
cmake .
make release && make
rm -f *.gpl
./dg
gnuplot anim.gnu
```

## Order of accuracy test

We solve with initial condition `u(x,0) = 0.2 * sin(x)` upto `t=2` where the solution is still smooth. We find the exact solution by solving the equation `u(x,t) = u(x - t u(x,t), 0)` using Newton method.

Modify the `main` function like this

```c++
       param.degree = 1;
       param.n_cells = 50;
       param.nstep = 5;
       param.output_step = 10;
       param.test_case = exact1;
       param.cfl = 0.98/(2.0*param.degree+1.0);
       param.final_time = 2;
       param.flux_type = godunov;
       param.limiter_type = none
```

we observe optimal convergence rates

```bash
step cells dofs       L2             H1
   0    50  100 4.440e-04    - 2.003e-02    -
   1   100  200 1.141e-04 1.96 1.014e-02 0.98
   2   200  400 2.896e-05 1.98 5.101e-03 0.99
   3   400  800 7.301e-06 1.99 2.559e-03 1.00
   4   800 1600 1.833e-06 1.99 1.282e-03 1.00
```

You can generate a Latex file with the error table

```bash
pdflatex error.tex
open error.pdf
```