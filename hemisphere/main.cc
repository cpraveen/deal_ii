#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>

using namespace dealii;

void set_boundary_id(Triangulation<3>& triangulation,
                     const Point<3>&   center,
                     const double      inner_radius,
                     const double      outer_radius)
{
   // We want to use a standard boundary description where
   // the boundary is not curved. Hence set boundary id 2 to
   // to all faces in a first step.
   Triangulation<3>::cell_iterator cell = triangulation.begin();
   for (; cell!=triangulation.end(); ++cell)
      for (unsigned int i=0; i<GeometryInfo<3>::faces_per_cell; ++i)
         if (cell->at_boundary(i))
            cell->face(i)->set_all_boundary_ids(2);

   // Next look for the curved boundaries. If the x value of the
   // center of the face is not equal to center(0), we're on a curved
   // boundary. Then decide whether the center is nearer to the inner
   // or outer boundary to set the correct boundary id.
   for (cell=triangulation.begin(); cell!=triangulation.end(); ++cell)
      for (unsigned int i=0; i<GeometryInfo<3>::faces_per_cell; ++i)
         if (cell->at_boundary(i))
         {
            const Triangulation<3>::face_iterator face
            = cell->face(i);

            unsigned int n_inner = 0, n_outer = 0;

            for(unsigned int j=0; j<GeometryInfo<3>::vertices_per_face; ++j)
            {
               double dr = (face->vertex(j) - center).norm();
               if(fabs(dr-inner_radius) < 1.0e-10*inner_radius)
               {
                  ++n_inner;
               }
               else if(fabs(dr-outer_radius) < 1.0e-10*outer_radius)
               {
                  ++n_outer;
               }

               if(n_inner == 4)
                  face->set_all_boundary_ids(3);
               else if(n_outer == 4)
                  face->set_all_boundary_ids(1);
            }
         }
}

int main()
{
   Triangulation<3> triangulation;
   Point<3> center(0,0,0);
   double inner_radius = 1.0;
   double outer_radius = 2.0;
   GridGenerator::half_hyper_shell(triangulation,
                                   center,
                                   inner_radius,
                                   outer_radius,
                                   0,
                                   true);

   set_boundary_id(triangulation, center, inner_radius, outer_radius);
   triangulation.refine_global(4);

   std::cout << "Number of cells    = " << triangulation.n_active_cells() << std::endl;
   std::cout << "Number of vertices = " << triangulation.n_used_vertices() << std::endl;

   GridOut grid_out;
   GridOutFlags::Msh msh_flags(true, false);
   std::ofstream msh_file("hemisphere.msh");
   grid_out.set_flags(msh_flags);
   grid_out.write_msh(triangulation, msh_file);
}
