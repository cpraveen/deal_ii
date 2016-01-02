# Example of OCC

This example takes a grid around an RAE2822 airfoil and refines it using the IGES file.

Inside ```occ/rae2822``` run
```
gmsh -2 rae2822.geo
```
This generates the ```rae2822.msh``` file. Inside ```occ/refine``` do
```
cmake .
make
./main
```
See the files ```grid0.vtk``` and ```grid1.vtk```.
