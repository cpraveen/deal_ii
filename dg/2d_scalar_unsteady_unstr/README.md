# 2d linear advection equation

Set `grid` in `run` function.

For `grid = 2`, generate grid

```
gmsh -2 annulus.geo
```

Compile

```
make release
make
```

Run

```
mpirun -np 4 ./dg
```

See solution

```
visit -o solution.visit
```

or

```
paraview solution.pvd
```

You can save in high order format by setting

```
flags.write_higher_order_cells = true
```

but this only works in paraview.
