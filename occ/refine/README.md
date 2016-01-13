# Example of OCC

This example takes a grid around an RAE2822 airfoil and refines it using the IGES file.

Inside ```occ/rae2822``` run
```
gmsh -2 rae2822.geo
```
This makes use of the iges file and generates the ```rae2822.msh``` file. Inside ```occ/refine``` do
```
cmake .
make
./main
```
See the initial grid in ```grid0.vtk``` and the refined grid in ```grid1.vtk```. The new points during refinement are added on the true boundary as represented by the iges file.
