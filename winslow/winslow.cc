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
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/filtered_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>


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
   void initialize_grid ();
   void setup_system ();
   void assemble_mass_matrix ();
   void assemble_system_matrix_rhs ();
   void set_initial_condition ();
   void map_boundary_values ();
   void solve ();
   void solve_direct ();
   double compute_change ();
   void output ();
   
   Triangulation<dim> triangulation;
   FE_Q<dim> fe;
   DoFHandler<dim> dof_handler;
   
   Vector<double> x, y;
   Vector<double> x_old, y_old;
   Vector<double> ax, ay;
   
   SparsityPattern       sparsity_pattern;
   SparseMatrix<double>  mass_matrix;
   SparseMatrix<double>  system_matrix;
   SparseMatrix<double>  system_matrix_x;
   SparseMatrix<double>  system_matrix_y;
   
   Vector<double> rhs_x;
   Vector<double> rhs_y;
   Vector<double> rhs_ax;
   Vector<double> rhs_ay;
   
   std::map<types::global_dof_index,double> boundary_values_x;
   std::map<types::global_dof_index,double> boundary_values_y;
   
   SparseDirectUMFPACK  solver_mass_matrix;
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
   std::cout << "Number of dofs = " << dof_handler.n_dofs() << std::endl;
   std::cout << "Dofs per cell  = " << fe.dofs_per_cell << std::endl;
   std::cout << "Dofs per face  = " << fe.dofs_per_face << std::endl;

   x.reinit (dof_handler.n_dofs());
   y.reinit (dof_handler.n_dofs());
   x_old.reinit (dof_handler.n_dofs());
   y_old.reinit (dof_handler.n_dofs());
   ax.reinit (dof_handler.n_dofs());
   ay.reinit (dof_handler.n_dofs());
   
   rhs_x.reinit (dof_handler.n_dofs());
   rhs_y.reinit (dof_handler.n_dofs());
   rhs_ax.reinit (dof_handler.n_dofs());
   rhs_ay.reinit (dof_handler.n_dofs());
   
   DynamicSparsityPattern dsp(dof_handler.n_dofs());
   DoFTools::make_sparsity_pattern(dof_handler,
                                   dsp);
   sparsity_pattern.copy_from(dsp);
   system_matrix.reinit (sparsity_pattern);
   system_matrix_x.reinit (sparsity_pattern);
   system_matrix_y.reinit (sparsity_pattern);
   mass_matrix.reinit (sparsity_pattern);
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::initialize_grid ()
{
   GridGenerator::hyper_ball (triangulation);
   static const SphericalManifold<dim> boundary;
   triangulation.set_all_manifold_ids_on_boundary(0);
   triangulation.set_manifold (0, boundary);
   
   std::cout << "Number of cell = " << triangulation.n_active_cells();
   std::cout << std::endl;
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::assemble_mass_matrix ()
{
   std::cout << "Creating mass matrix\n";
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
   std::cout << "Setting initial condition\n";
   std::vector<Point<dim>> support_points (dof_handler.n_dofs());
   DoFTools::map_dofs_to_support_points (MappingQ<dim,dim>(fe.degree),
                                         dof_handler,
                                         support_points);
   
   for(unsigned int i=0; i<dof_handler.n_dofs(); ++i)
   {
      x(i) = support_points[i][0];
      y(i) = support_points[i][1];
   }
   
   x_old = x;
   y_old = y;
   
   ax = 0;
   ay = 0;
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::map_boundary_values()
{
   std::cout << "Creating boundary condition list\n";
   const unsigned int dofs_per_face = fe.dofs_per_face;
   std::vector<types::global_dof_index> dof_indices(dofs_per_face);

   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end();
   for (; cell!=endc; ++cell)
   {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
         if (cell->face(f)->at_boundary())
         {
            cell->face(f)->get_dof_indices(dof_indices);
            for(unsigned int i=0; i<dofs_per_face; ++i)
            {
               // search if index exists in boundary map
               const unsigned int global_i = dof_indices[i];
               std::map<types::global_dof_index,double>::iterator it = boundary_values_x.find(global_i);
               if(it == boundary_values_x.end())
               {
                  boundary_values_x.insert(std::pair<types::global_dof_index,double>(global_i,x(global_i)));
                  boundary_values_y.insert(std::pair<types::global_dof_index,double>(global_i,y(global_i)));
               }
            }
            // Remove the manifold
            cell->face(f)->set_manifold_id (numbers::flat_manifold_id);
         }
   }
   
//   for (const auto &pair : boundary_values_x)
//   {
//      double x0 = pair.second;
//      double y0 = boundary_values_y[pair.first];
//      std::cout << x0 << "  " << y0 << std::endl;
//   }
//   exit(0);
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
   
   // Needed for face assembly
   const QGauss<dim-1>  face_quadrature_formula(fe.degree+1);
   FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                     update_values | update_gradients |
                                     update_normal_vectors | update_JxW_values);
   const unsigned int   n_face_q_points    = face_quadrature_formula.size();
   std::vector<Tensor<1,dim>> Dx_face_values (n_face_q_points, Tensor<1,dim>());
   std::vector<Tensor<1,dim>> Dy_face_values (n_face_q_points, Tensor<1,dim>());

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
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
         if (cell->face(f)->at_boundary())
         {
            fe_face_values.reinit(cell, f);
            fe_face_values.get_function_gradients (x, Dx_face_values);
            fe_face_values.get_function_gradients (y, Dy_face_values);
            
            for(unsigned int q=0; q<n_face_q_points; ++q)
            {
               Tensor<2,dim> g;
               g_matrix (Dx_face_values[q], Dy_face_values[q], g);
               const Tensor<2,dim> gi = invert(g);
               
               for(unsigned int i=0; i<dofs_per_cell; ++i)
               {
                  cell_rhs_ax(i) -= (  gi[0][0]*fe_face_values.normal_vector(q)[0]
                                     + gi[1][0]*fe_face_values.normal_vector(q)[1]) *
                                    fe_face_values.shape_value(i,q) * fe_face_values.JxW(q);
                  cell_rhs_ay(i) -= (  gi[0][1]*fe_face_values.normal_vector(q)[0]
                                     + gi[1][1]*fe_face_values.normal_vector(q)[1]) *
                                    fe_face_values.shape_value(i,q) * fe_face_values.JxW(q);
               }
            }
         }
      
      // Add cell_matrix to system_matrix
      cell->get_dof_indices(local_dof_indices);
      for(unsigned int i=0; i<dofs_per_cell; ++i)
      {
         rhs_ax(local_dof_indices[i]) += cell_rhs_ax(i);
         rhs_ay(local_dof_indices[i]) += cell_rhs_ay(i);
         
         for(unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i,j));
      }
   }
   
   system_matrix_x.copy_from(system_matrix);
   system_matrix_y.copy_from(system_matrix);
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::solve_direct ()
{
   static int first_time = 1;

   // solve for ax, ay
   if(first_time)
   {
      solver_mass_matrix.initialize(mass_matrix);
      first_time = 0;
   }
   solver_mass_matrix.vmult (ax, rhs_ax);
   solver_mass_matrix.vmult (ay, rhs_ay);
   
   // solve x
   {
      MatrixTools::apply_boundary_values (boundary_values_x,
                                          system_matrix_x,
                                          x,
                                          rhs_x);
      SparseDirectUMFPACK solver_x;
      solver_x.initialize (system_matrix_x);
      solver_x.vmult (x, rhs_x);
   }
   
   // solve y
   {
      MatrixTools::apply_boundary_values (boundary_values_y,
                                          system_matrix_y,
                                          y,
                                          rhs_y);
      SparseDirectUMFPACK solver_y;
      solver_y.initialize (system_matrix_y);
      solver_y.vmult (y, rhs_y);
   }
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::solve ()
{
   // solve for ax, ay
   {
      SolverControl           solver_control (1000, 1e-12);
      SolverCG<>              solver (solver_control);
      PreconditionSSOR<> preconditioner;
      preconditioner.initialize(mass_matrix, 1.2);
      solver.solve (mass_matrix,
                    ax,
                    rhs_ax,
                    preconditioner);
      solver.solve (mass_matrix,
                    ay,
                    rhs_ay,
                    preconditioner);
   }
   
   // solve for x, y
   {
      FilteredMatrix<Vector<double>> system_matrix_x (system_matrix);
      system_matrix_x.add_constraints (boundary_values_x);

      FilteredMatrix<Vector<double>> system_matrix_y (system_matrix);
      system_matrix_y.add_constraints (boundary_values_y);
      
      // set up a linear solver
      SolverControl control (1000, 1.e-10, false, false);
      SolverCG<Vector<double>> solver (control);
      
      // set up a preconditioner object
      PreconditionJacobi<SparseMatrix<double> > prec;
      prec.initialize (system_matrix, 1.2);
      
      FilteredMatrix<Vector<double>> prec_x (prec);
      prec_x.add_constraints (boundary_values_x);
      // compute modification of right hand side
      prec_x.apply_constraints (rhs_x);
      
      FilteredMatrix<Vector<double>> prec_y (prec);
      prec_y.add_constraints (boundary_values_y);
      // compute modification of right hand side
      prec_y.apply_constraints (rhs_y);
      
      // solve for solution vector x
      solver.solve (system_matrix_x, x, rhs_x, prec_x);
      solver.solve (system_matrix_y, y, rhs_y, prec_y);
   }
}

//------------------------------------------------------------------------------
template <int dim>
double Winslow<dim>::compute_change ()
{
   Vector<double> dx (x);
   dx -= x_old;
   double res_norm_x = dx.l2_norm();
   res_norm_x = std::sqrt( std::pow(res_norm_x,2) / dx.size() );

   Vector<double> dy (y);
   dy -= y_old;
   double res_norm_y = dy.l2_norm();
   res_norm_y = std::sqrt( std::pow(res_norm_y,2) / dy.size() );
   
   return res_norm_x + res_norm_y;
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::output ()
{
   static int count = 0;
   
   DataOut<dim> data_out;
   data_out.attach_dof_handler (dof_handler);
   data_out.add_data_vector(x, "x");
   data_out.add_data_vector(y, "y");
   data_out.build_patches (fe.degree);
   
   std::string filename = "sol-" + Utilities::int_to_string(count, 2) + ".vtk";
   std::ofstream output (filename.c_str());
   data_out.write_vtk (output);
   
   ++count;
}

//------------------------------------------------------------------------------
template <int dim>
void Winslow<dim>::run()
{
   initialize_grid ();
   setup_system ();
   set_initial_condition ();
   map_boundary_values ();
   assemble_mass_matrix ();
   
   output ();
   
   // start Picard iteration
   const double RESTOL = 1.0e-8;
   double res_norm = RESTOL + 1;
   unsigned int iter = 0, max_iter = 10;
   while(res_norm > RESTOL && iter < max_iter)
   {
      assemble_system_matrix_rhs ();
      solve_direct ();
      res_norm = compute_change ();
      ++iter;
      std::cout << iter << "  " << res_norm << std::endl;
      x_old = x;
      y_old = y;
      output ();
   }
}

//------------------------------------------------------------------------------
int main ()
{
   Winslow<2> winslow (2);
   winslow.run ();
}
