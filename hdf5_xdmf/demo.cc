#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/dofs/dof_tools.h>

#include <random>
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace dealii;

static const int dim = 3;
static const int degree = 4;

typedef LinearAlgebra::distributed::Vector<double> PVector;
typedef parallel::distributed::Triangulation<dim> PTriangulation;

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  PTriangulation triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  triangulation.refine_global(5); // 2^7 x degree(4) approx 512
  DoFHandler<dim>  dof_handler(triangulation);
  const FESystem<dim> fe(FE_Q<dim>(degree), dim+2);
  dof_handler.clear();
  dof_handler.distribute_dofs(fe);
  const auto& locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  PVector solution;
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  // Fill random values into solution
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for(auto i : solution.locally_owned_elements())
    solution[i] = dist(rng);
  solution.compress(VectorOperation::insert);
  solution.update_ghost_values();
  solution *= 0.1234;

  const bool write_mesh_file = true;
  std::string mesh_filename = "mesh.h5";
  std::string solution_filename = "vars.h5";

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.build_patches(fe.degree);

  DataOutBase::DataOutFilter data_filter(DataOutBase::DataOutFilterFlags(true, true));
  data_out.write_filtered_data(data_filter);

  data_out.write_hdf5_parallel(data_filter,
                               write_mesh_file,
                               mesh_filename,
                               solution_filename,
                               MPI_COMM_WORLD);

  XDMFEntry new_xdmf_entry = data_out.create_xdmf_entry(data_filter,
                                                        mesh_filename,
                                                        solution_filename,
                                                        0.0,
                                                        MPI_COMM_WORLD);

  std::vector<XDMFEntry>        xdmf_entries;
  xdmf_entries.push_back(new_xdmf_entry);
  data_out.write_xdmf_file(xdmf_entries, "solution.xdmf", MPI_COMM_WORLD);

  return 0;
}