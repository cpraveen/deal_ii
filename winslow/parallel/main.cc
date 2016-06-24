/* ---------------------------------------------------------------------
 Solve the Winslow equations using Picard iterations as in
 Meire Fortunato, Per-Olof Persson
 High-order unstructured curved mesh generation using the Winslow equations
 Journal of Computational Physics 307 (2016) 1–14
 http://dx.doi.org/10.1016/j.jcp.2015.11.020
 --------------------------------------------------------------------- */


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>

#include <iostream>
#include <fstream>

#include "winslow.h"
#include "naca.h"


//------------------------------------------------------------------------------
int main (int argc, char *argv[])
{
   try
   {
      using namespace dealii;
      using namespace Winslow;
      
      Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
      
      // Setup the triangulation
      parallel::distributed::Triangulation<2> triangulation (MPI_COMM_WORLD);
      
      unsigned int test_case = 1;
      unsigned int n_refine = 0;
      
      if(test_case==0)
      {
         GridGenerator::hyper_ball (triangulation);
         n_refine = 2;
      }
      else if(test_case==1)
      {
         GridGenerator::hyper_shell (triangulation, Point<2>(0.0,0.0), 0.5, 1.0);
         n_refine = 2;
      }
      else if(test_case==2)
      {
         GridIn<2> grid_in;
         grid_in.attach_triangulation(triangulation);
         std::ifstream input_file("annulus.msh");
         grid_in.read_msh(input_file);
         n_refine = 0;
      }
      else if(test_case==3)
      {
         GridIn<2> grid_in;
         grid_in.attach_triangulation(triangulation);
         std::ifstream input_file("naca_struct.msh");
         grid_in.read_msh(input_file);
         n_refine = 0;
      }
      
      // Attach manifold to boundaries
      if(test_case==0 || test_case==1 || test_case==2)
      {
         static const SphericalManifold<2> boundary;
         triangulation.set_all_manifold_ids_on_boundary(0);
         triangulation.set_manifold (0, boundary);
      }
      else  if(test_case==3)
      {
         NACA::set_curved_boundaries (triangulation);
         n_refine = 0;
      }
      
      // Do some refinement
      if(n_refine > 0) triangulation.refine_global(n_refine);
      
      // Compute the euler vector
      unsigned int degree = 4;
      compute_mapping (degree, triangulation);
   }
   catch (std::exception &exc)
   {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
   }
   catch (...)
   {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
   };
   
   return 0;
}
