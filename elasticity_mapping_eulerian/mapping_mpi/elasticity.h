#ifndef __ELASTICITY_MODEL_H__
#define __ELASTICITY_MODEL_H__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/numerics/matrix_tools.h>

using namespace dealii;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <int dim>
class ElasticityModel
{
   public:
      template <class TriaType>
      ElasticityModel(const TriaType& triangulation,
                      const int       degree,
                      const double    lambda,
                      const double    mu);
      template <class VecType>
      void reinit(VecType&  solution);

      template <class VecType>
      void solve(VecType&   solution,
                 const bool restart=false,
                 const int  verbosity=0);

      types::global_dof_index n_dofs() const
      {
         return dof_handler.n_dofs();
      }

      const DoFHandler<dim>& get_dof_handler()
      {
         return dof_handler;
      }

   private:
      void setup_system();
      void assemble_system_matrix();

      typedef PETScWrappers::MPI::SparseMatrix MatType;

      const MPI_Comm            mpi_comm;
      const ConditionalOStream  pcout;
      DoFHandler<dim>           dof_handler;
      const FESystem<dim>       fe;
      AffineConstraints<double> constraints;
      MatType                   matrix;
      MatType                   system_matrix;

      const double              lambda;
      const double              mu;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <int dim>
template <class TriaType>
ElasticityModel<dim>::ElasticityModel(const TriaType &triangulation,
                                      const int degree,
                                      const double lambda,
                                      const double mu)
    : mpi_comm(MPI_COMM_WORLD),
      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
      dof_handler(triangulation),
      fe(FE_Q<dim>(degree), dim),
      lambda(lambda),
      mu(mu)
{
   pcout << "Elasticity model for mapping\n";
   pcout << "   lambda = " << lambda << std::endl;
   pcout << "   mu     = " << mu     << std::endl;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <int dim>
template <class VecType>
void ElasticityModel<dim>::reinit(VecType& solution)
{
   pcout << "Setting up elasticity system ...\n";

   dof_handler.distribute_dofs(fe);

   const auto& locally_owned_dofs = dof_handler.locally_owned_dofs();
   const auto locally_relevant_dofs =
       DoFTools::extract_locally_relevant_dofs(dof_handler);

   constraints.clear();
   constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
   DoFTools::make_hanging_node_constraints(dof_handler, constraints);
   constraints.close();

   DynamicSparsityPattern sparsity_pattern(locally_relevant_dofs);
   DoFTools::make_sparsity_pattern(dof_handler,
                                   sparsity_pattern,
                                   constraints,
                                   /*keep_constrained_dofs = */ false);
   SparsityTools::distribute_sparsity_pattern(sparsity_pattern,
                                              locally_owned_dofs,
                                              mpi_comm,
                                              locally_relevant_dofs);

   matrix.reinit(locally_owned_dofs,
                 locally_owned_dofs,
                 sparsity_pattern,
                 mpi_comm);
   system_matrix.reinit(matrix);

   solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

   assemble_system_matrix();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <int dim>
void ElasticityModel<dim>::assemble_system_matrix()
{
   pcout << "Assembling elasticity matrix ...\n";
   const QGauss<dim> quadrature_formula(fe.degree + 1);
   FEValues<dim> fe_values(fe,
                           quadrature_formula,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

   const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
   const unsigned int n_q_points = quadrature_formula.size();

   FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
   std::vector<double> lambda_values(n_q_points);
   std::vector<double> mu_values(n_q_points);
   Functions::ConstantFunction<dim> lambda(this->lambda), mu(this->mu);

   for (const auto &cell : dof_handler.active_cell_iterators())
   if(cell->is_locally_owned())
   {
      fe_values.reinit(cell);

      cell_matrix = 0;

      lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
      mu.value_list(fe_values.get_quadrature_points(), mu_values);

      for (const unsigned int i : fe_values.dof_indices())
      {
         const unsigned int component_i =
             fe.system_to_component_index(i).first;

         for (const unsigned int j : fe_values.dof_indices())
         {
            const unsigned int component_j =
                fe.system_to_component_index(j).first;

            for (const unsigned int q_point :
                 fe_values.quadrature_point_indices())
            {
               cell_matrix(i, j) +=
                   (
                       (fe_values.shape_grad(i, q_point)[component_i] *
                        fe_values.shape_grad(j, q_point)[component_j] *
                        lambda_values[q_point])
                       +
                       (fe_values.shape_grad(i, q_point)[component_j] *
                        fe_values.shape_grad(j, q_point)[component_i] *
                        mu_values[q_point])
                       +
                       ((component_i == component_j) ?
                            (fe_values.shape_grad(i, q_point) *
                             fe_values.shape_grad(j, q_point) *
                             mu_values[q_point]) : 0)
                   ) *
                   fe_values.JxW(q_point);
            }
         }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, 
                                             local_dof_indices, 
                                             matrix);
   }

   matrix.compress(VectorOperation::add);
}

//------------------------------------------------------------------------------
// We assume "solution" already has boundary values filled in.
//------------------------------------------------------------------------------
template <int dim>
template <class VecType>
void ElasticityModel<dim>::solve(VecType&   solution,
                                 const bool restart,
                                 const int  verbosity)
{
   pcout << "Solving\n";

   // Apply boundary condition
   const auto boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler);
   std::map<types::global_dof_index, double> boundary_values;
   for(auto i : boundary_dofs)
   {
      if(solution.in_local_range(i))
         boundary_values[i] = solution(i);
   }

   system_matrix.copy_from(matrix);
   VecType distributed_solution(dof_handler.locally_owned_dofs(), mpi_comm);
   if(restart)
      distributed_solution = solution;
   VecType system_rhs(distributed_solution);
   system_rhs = 0.0;
   MatrixTools::apply_boundary_values(boundary_values,
                                      system_matrix,
                                      distributed_solution,
                                      system_rhs,
                                      false);

   const double tol = 1.0e-6 * system_rhs.l2_norm();
   if(tol < 1.0e-14)
   {
      solution = 0.0;
      return;
   }

   SolverControl solver_control(1000, tol, true, true);
   PETScWrappers::SolverCG cg(solver_control);
   PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

   deallog.depth_console(verbosity);
   cg.solve(system_matrix, distributed_solution, system_rhs, preconditioner);

   constraints.distribute(distributed_solution);
   solution = distributed_solution;
}

#endif
