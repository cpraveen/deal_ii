#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <iostream>

using namespace dealii;

const int dim = 3;

const std::vector<unsigned int> mesh_size = {128, 128, 128};
const unsigned int nrefine = 0;

//const std::vector<unsigned int> mesh_size = {0, 0, 0};
//const unsigned int nrefine = 7;

typedef parallel::distributed::Triangulation<dim> PTriangulation;

int main(int argc, char* argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    ConditionalOStream  pcout(std::cout, 
                              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
    TimerOutput  computing_timer(pcout,
                                 TimerOutput::never,
                                 TimerOutput::wall_times);
    PTriangulation triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::smoothing_on_refinement,
      parallel::distributed::Triangulation<dim>::mesh_reconstruction_after_repartitioning);

    const Point<3> point1(0, 0, 0);
    const Point<3> point2(1, 1, 1);
    if(mesh_size[0] == 0)
      GridGenerator::hyper_rectangle(triangulation, point1, point2, true);
    else
      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                mesh_size,
                                                point1,
                                                point2,
                                                true);

  typedef typename PTriangulation::cell_iterator Iter;
  std::vector<GridTools::PeriodicFacePair<Iter>> periodicity_vector;

  // Periodic along x
  {
    TimerOutput::Scope t(computing_timer, "Collect faces x");
    pcout << "Collect periodic faces along x\n";
    GridTools::collect_periodic_faces(triangulation,
                                      0,
                                      1,
                                      0,
                                      periodicity_vector);
  }

  // Periodic along y
  {
    TimerOutput::Scope t(computing_timer, "Collect faces y");
    pcout << "Collect periodic faces along y\n";
    GridTools::collect_periodic_faces(triangulation,
                                      2,
                                      3,
                                      1,
                                      periodicity_vector);
  }

  // Periodic along z
  {
    TimerOutput::Scope t(computing_timer, "Collect faces z");
    pcout << "Collect periodic faces along z\n";
    GridTools::collect_periodic_faces(triangulation,
                                      4,
                                      5,
                                      2,
                                      periodicity_vector);
  }

  {
    TimerOutput::Scope t(computing_timer, "Add periodicity");
    pcout << "Applying periodicity\n";
    triangulation.add_periodicity(periodicity_vector);
  }

  if(nrefine > 0)
  {
    TimerOutput::Scope t(computing_timer, "Global refine");
    pcout << "Global refine\n";
    triangulation.refine_global(nrefine);
  }

  computing_timer.print_summary();
}
