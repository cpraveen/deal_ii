#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/meshworker/copy_data.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>

#include <iostream>

using namespace dealii;

const int dim = 2;

const std::vector<unsigned int> mesh_size = {3, 3};

typedef parallel::distributed::Triangulation<dim> PTriangulation;

int main(int argc, char* argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    ConditionalOStream  pcout(std::cout,
                              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
    PTriangulation triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::smoothing_on_refinement,
      parallel::distributed::Triangulation<dim>::mesh_reconstruction_after_repartitioning);

    const Point<dim> point1(0, 0);
    const Point<dim> point2(1, 1);
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              mesh_size,
                                              point1,
                                              point2,
                                              true);

  typedef typename PTriangulation::cell_iterator Iter;
  std::vector<GridTools::PeriodicFacePair<Iter>> periodicity_vector;

  // Periodic along x
  {
    pcout << "Collect periodic faces along x\n";
    GridTools::collect_periodic_faces(triangulation,
                                      0,
                                      1,
                                      0,
                                      periodicity_vector);
  }

  // Periodic along y
  {
    pcout << "Collect periodic faces along y\n";
    GridTools::collect_periodic_faces(triangulation,
                                      2,
                                      3,
                                      1,
                                      periodicity_vector);
  }

  {
    pcout << "Applying periodicity\n";
    triangulation.add_periodicity(periodicity_vector);
  }

  int count = 0;
  for(auto& cell : triangulation.active_cell_iterators())
  {
    cell->set_user_index(count++);
  }

  int no_boundary_faces = 0;
  for(auto& cell : triangulation.active_cell_iterators())
  {
    pcout << "----- Cell no = " << cell->user_index() << "----- \n";
    for(auto f : cell->face_indices())
    if(cell->face(f)->at_boundary())
    {
      ++no_boundary_faces;
      pcout << "\t Periodic neigbor cell = "
            << cell->periodic_neighbor(f)->user_index() << "\n";
    }
  }

  pcout << "No of boundary faces = " << no_boundary_faces << "\n";

  auto assemble_flags = MeshWorker::assemble_own_interior_faces_once;
  assemble_flags |= MeshWorker::assemble_ghost_faces_once;
  assemble_flags |= MeshWorker::assemble_boundary_faces;

  const auto iterator_range =
      filter_iterators(triangulation.active_cell_iterators(),
                       IteratorFilters::LocallyOwnedCell());

  using ScratchData = MeshWorker::ScratchData<dim, dim>;
  using CopyData = MeshWorker::CopyData<1, 1, 1>;
  using Iterator = decltype(triangulation.begin_active());

  const auto fe = FE_Q<dim>(1);
  ScratchData scratch_data(fe, QGauss<dim>(1), update_values);
  CopyData copy_data(fe.dofs_per_cell);

  auto face_worker =
      [](const Iterator &cell,
          const unsigned int f,
          const unsigned int sf,
          const Iterator &ncell,
          const unsigned int nf,
          const unsigned int nsf,
          ScratchData &scratch_data,
          CopyData &copy_data)
  {
    std::cout << "Interior face: (cell,neigh) = ("
              << cell->user_index() << ","
              << ncell->user_index()
              << ")" << std::endl;
  };

  auto boundary_worker =
      [](const Iterator &cell,
          const unsigned int f,
          ScratchData &scratch_data,
          CopyData &copy_data)
  {
    std::cout << "Boundary face: cell,f: "
              << cell->user_index() << " "
              << f << std::endl;
  };

  auto copier = [](const CopyData &cd)
  {
  };

  pcout << "Beginning of mesh_loop:\n\n";
  MeshWorker::mesh_loop(iterator_range,
                        nullptr,
                        copier,
                        scratch_data,
                        copy_data,
                        assemble_flags,
                        boundary_worker,
                        face_worker);
}
