#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

#include "elasticity.h"

using namespace dealii;

const int dim = 2;

int main(int argc, char** argv)
{
   typedef PETScWrappers::MPI::Vector VecType;
   Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

   parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
   GridGenerator::subdivided_hyper_rectangle (triangulation,
                                              {50,25},
                                              Point<dim>(0.0,0.0),
                                              Point<dim>(1.0,0.5),
                                              true);

   const int mapping_degree = 2;
   const double lambda = 1.0;
   const double mu = 1.0;
   ElasticityModel<dim> model(triangulation, mapping_degree, lambda, mu);
   VecType euler_vector;
   model.reinit(euler_vector);

   // Apply displacement on top boundary
   const std::string constants = "pi=3.141592653589793";
   const FunctionParser<dim> top_displacement("0; 0.1*sin(2*pi*x)",
                                              constants);
   std::map<types::global_dof_index,double> boundary_values;
   VectorTools::interpolate_boundary_values(model.get_dof_handler(),
                                            types::boundary_id(3),
                                            top_displacement,
                                            boundary_values);

   // euler_vector has ghosts, we cannot write into it. Create a temporary
   // vector without ghosts and then copy into euler_vector.
   VecType distributed_euler_vector(model.get_dof_handler().locally_owned_dofs(),
                                    MPI_COMM_WORLD);
   for(auto [i,v] : boundary_values)
   {
      if(distributed_euler_vector.in_local_range(i))
         distributed_euler_vector(i) = v;
   }
   distributed_euler_vector.compress(VectorOperation::insert);
   euler_vector = distributed_euler_vector;

   model.solve(euler_vector, 2);

   MappingQEulerian<dim,VecType> mapping(mapping_degree,
                                         model.get_dof_handler(),
                                         euler_vector);

   DataOut<dim> data_out;
   data_out.attach_dof_handler(model.get_dof_handler());
   data_out.add_data_vector(euler_vector, "euler");
   data_out.build_patches(mapping, mapping_degree);
   data_out.write_vtu_with_pvtu_record("./", "euler", 0, MPI_COMM_WORLD);

   return 0;
}
