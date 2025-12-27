# Test mapping

## Run one of the examples

Circle (`hyper_ball`)

```shell
../parallel/main 0
```

Annulus (`hyper_shell`)

```shell
../parallel/main 1
```

Annulus

```shell
gmsh -2 annulus.geo
../parallel/main 2
```

NACA0012 airfoil

```shell
gmsh -2 naca_struct.geo
../parallel/main 3
```

## Visualize

Start gnuplot

```shell
gnuplot> load 'plot.gnu'
```
