#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <iostream>

using namespace dealii;

int main() {
  constexpr unsigned int dim = 2;

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.close();

  MappingQ1<dim> mapping;

  MatrixFree<dim, double> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, QGauss<1>(2));

  unsigned int n_lanes = matrix_free.n_active_entries_per_cell_batch(0);
  std::cout << n_lanes << " lanes mf\n";

  return 0;
}

