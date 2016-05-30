/* ---------------------------------------------------------------------
 Solve the Winslow equations using Picard iterations as in
 Meire Fortunato, Per-Olof Persson
 High-order unstructured curved mesh generation using the Winslow equations
 Journal of Computational Physics 307 (2016) 1–14
 http://dx.doi.org/10.1016/j.jcp.2015.11.020
 --------------------------------------------------------------------- */


#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
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

#include "winslow.h"

using namespace dealii;

namespace Winslow
{
   
   //------------------------------------------------------------------------------
   template <int dim>
   Winslow<dim>::Winslow(const unsigned int  degree,
                         Triangulation<dim> &triangulation)
   :
   triangulation (&triangulation),
   fe (QGaussLobatto<1>(degree+1)),
   dof_handler (triangulation),
   cell_quadrature (2*fe.degree+1),
   face_quadrature (2*fe.degree+1)
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
      system_matrix_x.reinit (sparsity_pattern);
      system_matrix_y.reinit (sparsity_pattern);
      mass_matrix.reinit (sparsity_pattern);
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::initialize_grid ()
   {
      std::cout << "Number of cell = " << triangulation->n_active_cells();
      std::cout << std::endl;
      
      {
         std::ofstream out ("gridq1.vtk");
         GridOut grid_out;
         grid_out.write_vtk (*triangulation, out);
         std::cout << "Grid written to gridq1.vtk" << std::endl;
      }
      
      {
         std::ofstream out ("gridq1.gnu");
         GridOut grid_out;
         grid_out.write_gnuplot (*triangulation, out);
         std::cout << "Grid written to gridq1.gnu" << std::endl;
      }
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::assemble_mass_matrix ()
   {
      std::cout << "Creating mass matrix\n";
      MatrixCreator::create_mass_matrix (dof_handler,
                                         cell_quadrature,
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
      
      for(typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          cell!=endc; ++cell)
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
                  if(it == boundary_values_x.end()) // Did not find index, so add it
                  {
                     boundary_values_x.insert(std::pair<types::global_dof_index,double>(global_i,x(global_i)));
                     boundary_values_y.insert(std::pair<types::global_dof_index,double>(global_i,y(global_i)));
                  }
               }
            }
      }
      
      // set all boundaries to be flat
      static const FlatManifold<dim> flat_boundary;
      triangulation->set_all_manifold_ids_on_boundary(0);
      triangulation->set_manifold (0, flat_boundary);
      
      // save boundary points to file
      std::cout << "Number of boundary points = " << boundary_values_x.size() << std::endl;
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::output_grids()
   {
      std::cout << "Saving grid\n";
      
      QTrapez<dim-1> trapezoidal_rule;
      QIterated<dim-1> quadrature (trapezoidal_rule, fe.degree+1);
      unsigned int n_face_q_points = quadrature.size();
      FEFaceValues<dim> fe_face_values (fe, quadrature, update_values);
      std::vector<double> x_values(n_face_q_points);
      std::vector<double> y_values(n_face_q_points);
      
      std::ofstream bdpts ("bd.gnu");
      std::ofstream gridq ("gridqk.gnu");
      std::cout << "Boundary points saved into bd.gnu\n";
      std::cout << "High order grid saved into gridqk.gnu\n";
      
      for(typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          cell!=endc; ++cell)
      {
         for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
         {
            fe_face_values.reinit(cell, f);
            fe_face_values.get_function_values (x, x_values);
            fe_face_values.get_function_values (y, y_values);
            for(unsigned int q=0; q<n_face_q_points; ++q)
               gridq << x_values[q] << "  " << y_values[q] << std::endl;
            gridq << std::endl;
            if(cell->face(f)->at_boundary())
            {
               for(unsigned int q=0; q<n_face_q_points; ++q)
                  bdpts << x_values[q] << "  " << y_values[q] << std::endl;
               bdpts << std::endl;
            }
         }
      }
      
      bdpts.close();
      gridq.close();
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::assemble_alpha_rhs ()
   {
      rhs_ax        = 0;
      rhs_ay        = 0;
      
      // Needed for cell assembly
      FEValues<dim> fe_values (fe, cell_quadrature,
                               update_gradients | update_JxW_values);
      const unsigned int   dofs_per_cell = fe.dofs_per_cell;
      const unsigned int   n_q_points    = cell_quadrature.size();
      Vector<double>  cell_rhs_ax (dofs_per_cell);
      Vector<double>  cell_rhs_ay (dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      
      std::vector<Tensor<1,dim>> Dx_values (n_q_points, Tensor<1,dim>());
      std::vector<Tensor<1,dim>> Dy_values (n_q_points, Tensor<1,dim>());
      
      // Needed for face assembly
      FEFaceValues<dim> fe_face_values (fe, face_quadrature,
                                        update_values |
                                        update_gradients |
                                        update_normal_vectors |
                                        update_JxW_values);
      const unsigned int   n_face_q_points    = face_quadrature.size();
      std::vector<Tensor<1,dim>> Dx_face_values (n_face_q_points, Tensor<1,dim>());
      std::vector<Tensor<1,dim>> Dy_face_values (n_face_q_points, Tensor<1,dim>());
      
      for(typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          cell!=endc; ++cell)
      {
         cell_rhs_ax = 0;
         cell_rhs_ay = 0;
         
         fe_values.reinit (cell);
         
         fe_values.get_function_gradients (x, Dx_values);
         fe_values.get_function_gradients (y, Dy_values);
         
         for(unsigned int q=0; q<n_q_points; ++q)
         {
            Tensor<2,dim> g;
            g_matrix (Dx_values[q], Dy_values[q], g);
            const Tensor<2,dim> gi = ginvert(g);
            
            for(unsigned int i=0; i<dofs_per_cell; ++i)
            {
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
                  const Tensor<2,dim> gi = ginvert(g);
                  
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
         }
      }
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::solve_alpha ()
   {
      static int first_time = 1;
      
      // solve for ax, ay
      if(first_time)
      {
         std::cout << "LU decomposition of mass matrix\n";
         solver_mass_matrix.initialize(mass_matrix);
         first_time = 0;
      }
      solver_mass_matrix.vmult (ax, rhs_ax);
      solver_mass_matrix.vmult (ay, rhs_ay);
      
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::assemble_system_matrix_rhs ()
   {
      system_matrix_x = 0;
      rhs_x           = 0;
      rhs_y           = 0;
      
      FEValues<dim> fe_values (fe, cell_quadrature,
                               update_values | update_gradients | update_JxW_values);
      const unsigned int   dofs_per_cell = fe.dofs_per_cell;
      const unsigned int   n_q_points    = cell_quadrature.size();
      FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      
      std::vector<double> ax_values (n_q_points);
      std::vector<double> ay_values (n_q_points);
      std::vector<Tensor<1,dim>> Dx_values (n_q_points, Tensor<1,dim>());
      std::vector<Tensor<1,dim>> Dy_values (n_q_points, Tensor<1,dim>());
      
      for(typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          cell!=endc; ++cell)
      {
         cell_matrix = 0;
         
         fe_values.reinit (cell);
         
         fe_values.get_function_values (ax, ax_values);
         fe_values.get_function_values (ay, ay_values);
         fe_values.get_function_gradients (x, Dx_values);
         fe_values.get_function_gradients (y, Dy_values);
         
         for(unsigned int q=0; q<n_q_points; ++q)
         {
            Tensor<2,dim> g;
            g_matrix (Dx_values[q], Dy_values[q], g);
            const Tensor<2,dim> gi = ginvert(g);
            
            for(unsigned int i=0; i<dofs_per_cell; ++i)
            {
               for(unsigned int j=0; j<dofs_per_cell; ++j)
               {
                  cell_matrix(i,j) += ((gi * fe_values.shape_grad(j,q)) * fe_values.shape_grad(i,q)
                                       -
                                       (ax_values[q] * fe_values.shape_grad(j,q)[0] +
                                        ay_values[q] * fe_values.shape_grad(j,q)[1]) *
                                       fe_values.shape_value(i,q)) * fe_values.JxW(q);
               }
            }
         }
         
         // Add cell_matrix to system_matrix
         cell->get_dof_indices(local_dof_indices);
         system_matrix_x.add(local_dof_indices, cell_matrix);
      }
      
      system_matrix_y.copy_from(system_matrix_x);
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::solve_direct ()
   {
      // solve x
      {
         x = 0;
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
         y = 0;
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
      data_out.add_data_vector(ax, "ax");
      data_out.add_data_vector(ay, "ay");
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
      const double RESTOL = 1.0e-12;
      double res_norm = RESTOL + 1;
      unsigned int iter = 0, max_iter = 20;
      while(res_norm > RESTOL && iter < max_iter)
      {
         // solve ax, ay
         assemble_alpha_rhs ();
         solve_alpha ();
         // solve x, y
         assemble_system_matrix_rhs ();
         solve_direct ();
         
         res_norm = compute_change ();
         ++iter;
         std::cout << iter << "  " << res_norm << std::endl;
         x_old = x;
         y_old = y;
         output ();
      }
      
      output_grids ();
   }
   
   
}

// Instantiations
template class Winslow::Winslow<2>;
