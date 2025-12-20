#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <iostream>

using namespace dealii;

const int dim = 3;

const std::vector<unsigned int> mesh_size = {128, 128, 128};
const unsigned int nrefine = 0;

//const std::vector<unsigned int> mesh_size = {0, 0, 0};
//const unsigned int nrefine = 7;

typedef parallel::fullydistributed::Triangulation<dim> PTriangulation;

int main(int argc, char* argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    ConditionalOStream  pcout(std::cout, 
                              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
    TimerOutput  computing_timer(pcout,
                                 TimerOutput::never,
                                 TimerOutput::wall_times);
    PTriangulation triangulation(MPI_COMM_WORLD);

    const Point<3> point1(0, 0, 0);
    const Point<3> point2(1, 1, 1);
    Triangulation<dim> basetria;
    if(mesh_size[0] == 0)
      GridGenerator::hyper_rectangle(basetria, point1, point2, true);
    else
      GridGenerator::subdivided_hyper_rectangle(basetria,
                                                mesh_size,
                                                point1,
                                                point2,
                                                true);
   GridTools::partition_triangulation_zorder(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), basetria);

  // extract relevant information form serial triangulation
  auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      basetria, MPI_COMM_WORLD);

  // actually create triangulation
  triangulation.create_triangulation(construction_data);

  auto add_periodicity = [&](dealii::Triangulation<dim> &tria) 
  {
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
         periodic_faces;

    {
      TimerOutput::Scope t(computing_timer, "Collect faces x");
      pcout << "Collect periodic faces along x\n";
      GridTools::collect_periodic_faces(tria, 0, 1, 0, periodic_faces);
    }

    {
      TimerOutput::Scope t(computing_timer, "Collect faces y");
      pcout << "Collect periodic faces along y\n";
      GridTools::collect_periodic_faces(tria, 2, 3, 1, periodic_faces);
    }

    {
      TimerOutput::Scope t(computing_timer, "Collect faces z");
      pcout << "Collect periodic faces along z\n";
      GridTools::collect_periodic_faces(tria, 4, 5, 2, periodic_faces);
    }

    {
      TimerOutput::Scope t(computing_timer, "Add periodicity");
      pcout << "Applying periodicity\n";
      tria.add_periodicity(periodic_faces);
    }
  };

  add_periodicity(triangulation);

  if(nrefine > 0)
  {
    TimerOutput::Scope t(computing_timer, "Global refine");
    pcout << "Global refine\n";
    triangulation.refine_global(nrefine);
  }

  computing_timer.print_summary();
}
