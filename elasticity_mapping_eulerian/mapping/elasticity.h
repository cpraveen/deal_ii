#ifndef __ELASTICITY_MODEL_H__
#define __ELASTICITY_MODEL_H__

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/matrix_tools.h>

using namespace dealii;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <int dim>
class ElasticityModel
{
   public:
      ElasticityModel(const Triangulation<dim>& triangulation,
                      const int    degree,
                      const double lambda,
                      const double mu,
                      const std::set<types::boundary_id>& noslip_boundaries,
                      const std::set<types::boundary_id>& slip_boundaries);
      void solve(Vector<double>& solution,
                 const bool restart,
                 const int verbosity=0);

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

      DoFHandler<dim>           dof_handler;
      const FESystem<dim>       fe;
      AffineConstraints<double> constraints;
      SparsityPattern           sparsity_pattern;
      SparseMatrix<double>      matrix;
      SparseMatrix<double>      system_matrix;
      Vector<double>            system_rhs;

      const double              lambda;
      const double              mu;

      const std::set<types::boundary_id> noslip_boundaries;
      const std::set<types::boundary_id> slip_boundaries;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <int dim>
ElasticityModel<dim>::ElasticityModel(const Triangulation<dim>& triangulation, 
                                      const int degree,
                                      const double lambda,
                                      const double mu,
                                      const std::set<types::boundary_id>& noslip_boundaries,
                                      const std::set<types::boundary_id>& slip_boundaries)
    : dof_handler(triangulation),
      fe(FE_Q<dim>(degree), dim),
      lambda(lambda),
      mu(mu),
      noslip_boundaries(noslip_boundaries),
      slip_boundaries(slip_boundaries)
{
   std::cout << "Elasticity model for mapping\n";
   std::cout << "   lambda = " << lambda << std::endl;
   std::cout << "   mu     = " << mu     << std::endl;

   setup_system();
   assemble_system_matrix();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <int dim>
void ElasticityModel<dim>::setup_system()
{
   std::cout << "Setting up elasticity system ...\n";

   dof_handler.distribute_dofs(fe);
   system_rhs.reinit(dof_handler.n_dofs());

   constraints.clear();
   DoFTools::make_hanging_node_constraints(dof_handler, constraints);
   VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                   0,
                                                   slip_boundaries,
                                                   constraints);
   constraints.close();

   DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
   DoFTools::make_sparsity_pattern(dof_handler,
                                   dsp,
                                   constraints,
                                   /*keep_constrained_dofs = */ false);
   sparsity_pattern.copy_from(dsp);

   matrix.reinit(sparsity_pattern);
   system_matrix.reinit(sparsity_pattern);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <int dim>
void ElasticityModel<dim>::assemble_system_matrix()
{
   std::cout << "Assembling elasticity matrix ...\n";
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
}

//------------------------------------------------------------------------------
// We assume "solution" already has boundary values filled in.
//------------------------------------------------------------------------------
template <int dim>
void ElasticityModel<dim>::solve(Vector<double>& solution,
                                 const bool restart,
                                 const int verbosity)
{
   system_rhs = 0.0;
   system_matrix.copy_from(matrix);

   // Apply boundary condition
   const ComponentMask component_mask = {};
   const auto boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler, 
                                                              component_mask,
                                                              noslip_boundaries);
   std::map<types::global_dof_index, double> boundary_values;
   for(auto i : boundary_dofs)
   {
      boundary_values[i] = solution(i);
   }
   if(restart == false) solution = 0.0;
   MatrixTools::apply_boundary_values(boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);

   const double tol = 1.0e-6 * system_rhs.l2_norm();
   if(tol < 1.0e-14)
   {
      solution = 0.0;
      return;
   }

   SolverControl solver_control(1000, tol, true, true);
   SolverCG<Vector<double>> cg(solver_control);

   PreconditionSSOR<SparseMatrix<double>> preconditioner;
   preconditioner.initialize(system_matrix, 1.2);

   cg.solve(system_matrix, solution, system_rhs, preconditioner);
   if(verbosity > 0)
      std::cout << "CG iters = " << solver_control.last_step()
                << ", res norm = " << solver_control.last_value() << std::endl;

   constraints.distribute(solution);
}

#endif
