/* ---------------------------------------------------------------------
Solve the Winslow equations using Picard iterations as in
   Meire Fortunato, Per-Olof Persson
   High-order unstructured curved mesh generation using the Winslow equations
   Journal of Computational Physics 307 (2016) 1–14
   http://dx.doi.org/10.1016/j.jcp.2015.11.020
--------------------------------------------------------------------- */


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>

#include <iostream>
#include <fstream>

#include "winslow.h"


//------------------------------------------------------------------------------
int main ()
{
   using namespace dealii;
   using namespace Winslow;
   
   // Setup the triangulation
   Triangulation<2> triangulation;
   
   GridGenerator::hyper_ball (triangulation);
   //GridGenerator::hyper_shell (triangulation, Point<dim>(0.0,0.0), 0.5, 1.0);
   //GridIn<2> grid_in;
   //grid_in.attach_triangulation(triangulation);
   //std::ifstream input_file("annulus.msh");
   //grid_in.read_msh(input_file);
   
   // Attach manifold to boundaries
   static const SphericalManifold<2> boundary;
   triangulation.set_all_manifold_ids_on_boundary(0);
   triangulation.set_manifold (0, boundary);
   
   // Do some refinement
   triangulation.refine_global(2);
   
   // Compute the euler vector
   unsigned int degree = 4;
   compute_mapping (degree, triangulation);
}
