/*
Mark cells which are cut by an embedded curve.
Uses ideas from step-60 of deal.II tutorials.

WARNING: Does not mark all cells properly.

We make a grid on the curve and find in which cells these grid points lie and
mark them. But this can miss some cells if the curve-grid is coarser than the 
space-grid.
*/
#include <deal.II/base/function_parser.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/vector.h>

#include <fstream>

const int dim = 1;
const int spacedim = 2;

using namespace dealii;
namespace bgi = boost::geometry::index;

int main()
{
   Triangulation<spacedim> space_grid;
   GridGenerator::hyper_cube(space_grid, 0, 1, true);
   space_grid.refine_global(6);

   Triangulation<dim,spacedim> embedded_grid;
   GridGenerator::hyper_cube(embedded_grid);
   embedded_grid.refine_global(6);

   FESystem<dim,spacedim> embedded_configuration_fe(FE_Q<dim, spacedim>(2),
                                                    spacedim);
   DoFHandler<dim, spacedim>  embedded_configuration_dh(embedded_grid);
   embedded_configuration_dh.distribute_dofs(embedded_configuration_fe);

   Vector<double> embedded_configuration;
   embedded_configuration.reinit(embedded_configuration_dh.n_dofs());

   std::string variables = "x,y";
   std::vector<std::string> expressions(2);
   expressions[0] = "R*cos(2*pi*x)+Cx";
   expressions[1] = "R*sin(2*pi*x)+Cy";
   std::map<std::string, double> constants;
   constants["R"] = 0.3;
   constants["Cx"] = 0.4;
   constants["Cy"] = 0.4;
   constants["pi"] = M_PI;
   FunctionParser<spacedim> embedded_configuration_function(2);
   embedded_configuration_function.initialize(variables, expressions, constants);

   VectorTools::interpolate(embedded_configuration_dh,
                            embedded_configuration_function,
                            embedded_configuration);

   MappingFEField<dim, spacedim, Vector<double>> 
      embedded_mapping(embedded_configuration_dh, embedded_configuration);

   std::vector<Point<spacedim>> support_points(embedded_configuration_dh.n_dofs());
   DoFTools::map_dofs_to_support_points(embedded_mapping,
                                        embedded_configuration_dh,
                                        support_points);

   FE_DGQ<spacedim> fe_cell(0);
   DoFHandler<spacedim>  cell_dh(space_grid);
   cell_dh.distribute_dofs(fe_cell);
   std::vector<types::global_dof_index> global_indices(fe_cell.dofs_per_cell);
   Vector<double> marker(cell_dh.n_dofs());
   marker = 0.0;

   GridTools::Cache<spacedim, spacedim> space_grid_tools_cache(space_grid);
   const auto &tree =
       space_grid_tools_cache.get_locally_owned_cell_bounding_boxes_rtree();
   GridTools::Cache<dim, spacedim> embedded_cache(embedded_grid);
   const auto &embedded_tree = embedded_cache.get_cell_bounding_boxes_rtree();

   for (const auto &[embedded_box, embedded_cell] : embedded_tree)
   {
      for (const auto &[space_box, space_cell] :
           tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
      {
         // do what you need here
         typename DoFHandler<spacedim>::active_cell_iterator
             u_cell(&space_grid,
                    space_cell->level(),
                    space_cell->index(),
                    &cell_dh);
         u_cell->get_dof_indices(global_indices);
         marker(global_indices[0]) = 1.0;
      }
   }

   {
      std::ofstream output_file("space_grid.gnu");
      GridOut().write_gnuplot (space_grid, output_file);
   }

   {
      std::ofstream output_file("embedded_grid.gnu");
      GridOut().write_gnuplot (embedded_grid, output_file, &embedded_mapping);
   }

   {
      DataOut<dim,spacedim> data_out;
      data_out.attach_dof_handler(embedded_configuration_dh);
      data_out.build_patches(embedded_mapping, embedded_configuration_fe.degree+1);
      std::ofstream output_file("embedded_grid.vtk");
      data_out.write_vtk (output_file);
   }

   {
      DataOut<spacedim> data_out;
      data_out.attach_dof_handler(cell_dh);
      data_out.add_data_vector(marker, "marker", DataOut<spacedim>::type_cell_data);
      data_out.build_patches();
      std::ofstream output("marker.vtk");
      data_out.write_vtk(output);
   }
}
