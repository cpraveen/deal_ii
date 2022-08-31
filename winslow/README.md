# Winslow equation based high order order meshes

The code in "serial" is bit old. It is recommended to use the code in "parallel". The parallel code requires Trilinos, so your deal.II must have been compiled with Trilinos.

```
cd parallel
cmake .
make release
make
```

Now run the code in "run" directory.

```
cd ../run
../parallel/main 0
```

Plot using gnuplot

```
gnuplot> load 'plot.gnu'
```

You can also run in parallel 
```
cd ../run
mpirun -np 2 ../parallel/main 0
```

## Different test cases

```
../parallel/main 0

../parallel/main 1

gmsh -2 annulus.geo
../parallel/main 2

gmsh -2 naca_struct.geo
../parallel/main 3
gnuplot naca.gnu
open gridq1.eps
open gridqk.eps
```
