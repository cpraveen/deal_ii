/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 */


#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/matrix_tools.h>


#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template <int dim>
void g_matrix (const Tensor<1,dim>& Dx, const Tensor<1,dim>& Dy, Tensor<2,dim>& g)
{
   g[0][0] = Dx[0] * Dx[0] + Dy[0] * Dy[0];
   g[0][1] = Dx[0] * Dx[1] + Dy[0] * Dy[1];
   g[1][0] = g[0][1];
   g[1][1] = Dx[1] * Dx[1] + Dy[1] * Dy[1];
}

template <int dim>
class Winslow
{
public:
   Winslow (unsigned int degree);
   void run ();
   
private:
   void setup_system ();
   void assemble_mass_matrix ();
   void assemble_system_matrix_rhs ();
   void set_initial_condition ();
   
   Triangulation<dim> triangulation;
   FE_Q<dim> fe;
   DoFHandler<dim> dof_handler;
   
   Vector<double> x, y;
   Vector<double> ax, ay;
   
   SparsityPattern       sparsity_pattern;
   SparseMatrix<double>  mass_matrix;
   SparseMatrix<double>  system_matrix;
   
   Vector<double> rhs_x;
   Vector<double> rhs_y;
   Vector<double> rhs_ax;
   Vector<double> rhs_ay;
};

//------------------------------------------------------------------------------
template <int dim>
Winslow<dim>::Winslow(unsigned int degree)
:
fe (QGaussLobatto<1>(degree+1)),
dof_handler (triangulation)
{
   
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::setup_system ()
{
   dof_handler.distribute_dofs (fe);
   
   x.reinit (dof_handler.n_dofs());
   y.reinit (dof_handler.n_dofs());
   ax.reinit (dof_handler.n_dofs());
   ay.reinit (dof_handler.n_dofs());
   
   DynamicSparsityPattern dsp(dof_handler.n_dofs());
   DoFTools::make_sparsity_pattern(dof_handler,
                                   dsp);
   sparsity_pattern.copy_from(dsp);
   system_matrix.reinit (sparsity_pattern);
   mass_matrix.reinit (sparsity_pattern);
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::assemble_mass_matrix ()
{
   MatrixCreator::create_mass_matrix (dof_handler,
                                      QGauss<dim>(fe.degree+1),
                                      mass_matrix);
}

//------------------------------------------------------------------------------
// Initial coordinates are set equal to support points on the Q1 mesh
//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::set_initial_condition ()
{
   std::vector<Point<dim>> support_points (dof_handler.n_dofs());
   DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
                                         dof_handler,
                                         support_points);
   
   for(unsigned int i=0; i<dof_handler.n_dofs(); ++i)
   {
      x(i) = support_points[i][0];
      y(i) = support_points[i][1];
   }
   
   ax = 0;
   ay = 0;
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::assemble_system_matrix_rhs ()
{
   system_matrix = 0;
   rhs_x         = 0;
   rhs_y         = 0;
   rhs_ax        = 0;
   rhs_ay        = 0;
   
   const QGauss<dim>  quadrature_formula(fe.degree+1);
   FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values | update_gradients | update_JxW_values);
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   const unsigned int   n_q_points    = quadrature_formula.size();
   FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
   Vector<double>  cell_rhs_ax (dofs_per_cell);
   Vector<double>  cell_rhs_ay (dofs_per_cell);
   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
   
   std::vector<double> ax_values (n_q_points);
   std::vector<double> ay_values (n_q_points);
   std::vector<Tensor<1,dim>> Dx_values (n_q_points, Tensor<1,dim>());
   std::vector<Tensor<1,dim>> Dy_values (n_q_points, Tensor<1,dim>());
   
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   for (; cell!=endc; ++cell)
   {
      cell_matrix = 0;
      cell_rhs_ax = 0;
      cell_rhs_ay = 0;
      
      fe_values.reinit (cell);
      
      fe_values.get_function_values (ax, ax_values);
      fe_values.get_function_values (ay, ay_values);
      fe_values.get_function_gradients (x, Dx_values);
      fe_values.get_function_gradients (y, Dy_values);
      
      for(unsigned int q=0; q<n_q_points; ++q)
      {
         Tensor<2,dim> g;
         g_matrix (Dx_values[q], Dy_values[q], g);
         const Tensor<2,dim> gi = invert(g);

         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            for(unsigned int j=0; j<dofs_per_cell; ++j)
            {
               cell_matrix(i,j) += ((gi * fe_values.shape_grad(j,q)) * fe_values.shape_grad(i,q)
                                   +
                                    (ax_values[q] * fe_values.shape_grad(j,q)[0] +
                                     ay_values[q] * fe_values.shape_grad(j,q)[1]) *
                                   fe_values.shape_value(i,q)) * fe_values.JxW(q);
            }
            
            cell_rhs_ax(i) += (gi[0][0] * fe_values.shape_grad(i,q)[0] +
                               gi[1][0] * fe_values.shape_grad(i,q)[1]) * fe_values.JxW(q);
            cell_rhs_ay(i) += (gi[0][1] * fe_values.shape_grad(i,q)[0] +
                               gi[1][1] * fe_values.shape_grad(i,q)[1]) * fe_values.JxW(q);
         }
      }
      
      // Boundary terms
      
      // Add cell_matrix to system_matrix
   }
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::run()
{
   assemble_mass_matrix ();
   
   // Set initial condition
   set_initial_condition ();
   
   // start Picard iteration
   assemble_system_matrix_rhs ();
}

//------------------------------------------------------------------------------
int main ()
{
   Winslow<2> winslow (2);
   winslow.run ();
}
