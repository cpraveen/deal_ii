# A manifold to describe the surface of sphere

For a description of this manifold, see

http://onlinelibrary.wiley.com/doi/10.1002/qj.49712253209/abstract

The formulae can also be found here

https://www.image.ucar.edu/staff/rnair/mwr05_dg_adv.pdf

The `CubedSphere` manifold can be used with `MappingManifold` which gives exact mapping. A comparison of the grid obtained from `SphericalManifold` and `CubedSphere` is shown below.

(grid)[cubedsphere.png)

The red grid is obtained with `SphericalManifold` and the black grid is from `CubedSphere`.

The use of this is shown in the following code.

```C++
const double R = 1.0; // Radius of sphere
Point<spacedim> center(0.0, 0.0, 0.0);
GridGenerator::hyper_sphere(triangulation, center, R);

static CubedSphere sphere1(1);
static CubedSphere sphere2(2);
static CubedSphere sphere3(3);
static CubedSphere sphere4(4);
static CubedSphere sphere5(5);
static CubedSphere sphere6(6);

triangulation.set_manifold (1, sphere1);
triangulation.set_manifold (2, sphere2);
triangulation.set_manifold (3, sphere3);
triangulation.set_manifold (4, sphere4);
triangulation.set_manifold (5, sphere5);
triangulation.set_manifold (6, sphere6);

triangulation.set_all_manifold_ids(0);

// 2*a is length of each side of cube inscribed in sphere
const double a = R/std::sqrt(3.0);

for (typename Triangulation<dim,spacedim>::active_cell_iterator
   cell=triangulation.begin_active();
   cell!=triangulation.end(); ++cell)
{
   const Point<spacedim> p = cell->center();
   if(std::fabs(p[0]-a) < 1.0e-13) // x plus
   {
      cell->set_all_manifold_ids(1);
      std::cout << "Setting CubedSphere(1)\n";
   }
   else if(std::fabs(p[0]+a) < 1.0e-13) // x minus
   {
      cell->set_all_manifold_ids(3);
      std::cout << "Setting CubedSphere(3)\n";
   }
   else if(std::fabs(p[1]-a) < 1.0e-13) // y plus
   {
      cell->set_all_manifold_ids(2);
      std::cout << "Setting CubedSphere(2)\n";
   }
   else if(std::fabs(p[1]+a) < 1.0e-13) // y minus
   {
      cell->set_all_manifold_ids(4);
      std::cout << "Setting CubedSphere(4)\n";
   }
   else if(std::fabs(p[2]-a) < 1.0e-13) // z plus
   {
      cell->set_all_manifold_ids(5);
      std::cout << "Setting CubedSphere(5)\n";
   }
   else if(std::fabs(p[2]+a) < 1.0e-13) // z minus
   {
      cell->set_all_manifold_ids(6);
      std::cout << "Setting CubedSphere(6)\n";
   }
   else
   {
      AssertThrow(false, ExcMessage("Error: Did not match any face of cube !!!"));
   }
}
// Now you can refine the grid
triangulation.refine_global(3);
```
