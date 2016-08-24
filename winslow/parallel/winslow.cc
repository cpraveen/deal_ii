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
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

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
   Winslow<dim>::Winslow(const unsigned int   degree,
                         PDTriangulation     &tria)
   :
   mpi_communicator (tria.get_communicator()),
   triangulation (&tria),
   fe (QGaussLobatto<1>(degree+1)),
   dof_handler (tria),
   cell_quadrature (2*fe.degree+1),
   face_quadrature (2*fe.degree+1),
   pcout (std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)==0))
   {
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::setup_system ()
   {
      dof_handler.distribute_dofs (fe);
      locally_owned_dofs = dof_handler.locally_owned_dofs ();
      DoFTools::extract_locally_relevant_dofs (dof_handler,
                                               locally_relevant_dofs);
      pcout << "Number of dofs = " << dof_handler.n_dofs() << std::endl;
      pcout << "Dofs per cell  = " << fe.dofs_per_cell << std::endl;
      pcout << "Dofs per face  = " << fe.dofs_per_face << std::endl;
      
      x.reinit (locally_relevant_dofs, mpi_communicator);
      y.reinit (x);
      x_old.reinit (locally_owned_dofs, mpi_communicator);
      y_old.reinit (locally_owned_dofs, mpi_communicator);
      ax.reinit (x);
      ay.reinit (x);
      
      rhs_x.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator, true);
      rhs_y.reinit (rhs_x);
      rhs_ax.reinit (rhs_x);
      rhs_ay.reinit (rhs_x);
      
      // Create hanging node constraints.
      // This is used for ax, ay and there are no boundary conditions.
      constraints.clear();
      constraints.reinit (locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler, constraints);
      constraints.close();
      
      // These are used for x, y. We add dirichlet bc later and close this.
      constraints_x.clear();
      constraints_x.reinit (locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler, constraints_x);

      constraints_y.clear();
      constraints_y.reinit (locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints (dof_handler, constraints_y);

      // Create sparsity pattern and allocate memory for matrix
      // NOTE: We use "constraints" for all the matrices, this should be ok.
      {
         TrilinosWrappers::SparsityPattern sp (locally_owned_dofs,
                                               locally_owned_dofs,
                                               locally_relevant_dofs,
                                               mpi_communicator);
         DoFTools::make_sparsity_pattern (dof_handler,
                                          sp,
                                          constraints,
                                          false,
                                          Utilities::MPI::this_mpi_process(mpi_communicator));
         sp.compress ();
         system_matrix_x.reinit (sp);
         system_matrix_y.reinit (sp);
         mass_matrix.reinit (sp);
         
         // Uncomment these lines to save sparsity pattern
         // Run in serial mode. Plot in gnuplot> p 'sparsity.gnu' w d
         //std::ofstream spfile ("sparsity.gnu");
         //sp.print_gnuplot(spfile);
      }
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::initialize_grid ()
   {
      pcout << "Number of cell = " << triangulation->n_active_cells();
      pcout << std::endl;
      
      if(Utilities::MPI::n_mpi_processes(mpi_communicator) > 1) return;
      
      {
         std::ofstream out ("gridq1.vtk");
         GridOut grid_out;
         grid_out.write_vtk (*triangulation, out);
         pcout << "Grid written to gridq1.vtk" << std::endl;
      }
      
      {
         std::ofstream out ("gridq1.gnu");
         GridOut grid_out;
         grid_out.write_gnuplot (*triangulation, out);
         pcout << "Grid written to gridq1.gnu" << std::endl;
      }
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::assemble_mass_matrix ()
   {
      pcout << "Creating mass matrix\n";
      
      FEValues<dim> fe_values (fe, cell_quadrature,
                               update_values | update_JxW_values);
      const unsigned int   dofs_per_cell = fe.dofs_per_cell;
      const unsigned int   n_q_points    = cell_quadrature.size();
      FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      mass_matrix = 0;
      
      for(typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          cell!=endc; ++cell)
         if(cell->is_locally_owned())
         {
            fe_values.reinit (cell);
            cell_matrix = 0;
            
            for(unsigned int i=0; i<dofs_per_cell; ++i)
               for(unsigned int j=0; j<dofs_per_cell; ++j)
                  for(unsigned int q=0; q<n_q_points; ++q)
                     cell_matrix(i,j) += fe_values.shape_value (i, q) *
                                         fe_values.shape_value (j, q) *
                                         fe_values.JxW (q);
            
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global (cell_matrix,
                                                    local_dof_indices,
                                                    mass_matrix);
         }
      
      mass_matrix.compress (VectorOperation::add);
   }
   
   //------------------------------------------------------------------------------
   // Initial coordinates are set equal to support points on the Q1 mesh
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::set_initial_condition ()
   {
      pcout << "Setting initial condition\n";
      std::map<types::global_dof_index, Point<dim>> support_points;
      DoFTools::map_dofs_to_support_points (MappingQ<dim,dim>(fe.degree),
                                            dof_handler,
                                            support_points);
      TrilinosWrappers::MPI::Vector x_tmp (locally_owned_dofs, mpi_communicator);
      TrilinosWrappers::MPI::Vector y_tmp (locally_owned_dofs, mpi_communicator);
      
      for(const auto &pair : support_points)
      {
         const Point<dim>& p = pair.second;
         x_tmp (pair.first) = p[0];
         y_tmp (pair.first) = p[1];
      }
      
      x     = x_tmp;
      y     = y_tmp;
      x_old = x_tmp;
      y_old = y_tmp;
      
      ax = 0;
      ay = 0;
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::map_boundary_values()
   {
      pcout << "Creating boundary condition list\n";
      const unsigned int dofs_per_face = fe.dofs_per_face;
      std::vector<types::global_dof_index> dof_indices(dofs_per_face);
      
      for(typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          cell!=endc; ++cell)
         if(cell->is_locally_owned())
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
      unsigned int n = boundary_values_x.size();
      n = Utilities::MPI::sum (n, mpi_communicator);
      pcout << "Number of boundary dofs = " << n << std::endl;
      
      add_dirichlet_constraints (boundary_values_x, constraints_x); constraints_x.close();
      add_dirichlet_constraints (boundary_values_y, constraints_y); constraints_y.close();
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::output_grids()
   {
      if(Utilities::MPI::n_mpi_processes(mpi_communicator) > 1) return;

      pcout << "Saving grid for visualization\n";
      
      QTrapez<dim-1> trapezoidal_rule;
      QIterated<dim-1> quadrature (trapezoidal_rule, fe.degree+1);
      unsigned int n_face_q_points = quadrature.size();
      FEFaceValues<dim> fe_face_values (fe, quadrature, update_values);
      std::vector<double> x_values(n_face_q_points);
      std::vector<double> y_values(n_face_q_points);
      
      std::ofstream bdpts ("bd.gnu");
      std::ofstream gridq ("gridqk.gnu");
      pcout << "Boundary points saved into bd.gnu\n";
      pcout << "High order grid saved into gridqk.gnu\n";
      
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
         if(cell->is_locally_owned())
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
            
            // Add cell rhs to global vector
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global (cell_rhs_ax, local_dof_indices, rhs_ax);
            constraints.distribute_local_to_global (cell_rhs_ay, local_dof_indices, rhs_ay);
         }
      
      rhs_ax.compress (VectorOperation::add);
      rhs_ay.compress (VectorOperation::add);
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::solve_alpha ()
   {
      static TrilinosWrappers::SolverDirect::AdditionalData data (false, "Amesos_Mumps");
      static SolverControl solver_control (1, 0);
      
      // If it is first time, compute LU decomposition
      if(!mumps_solver)
      {
         pcout << "Performing LU decomposition\n";
         mumps_solver = std_cxx11::shared_ptr<TrilinosWrappers::SolverDirect>
                        (new TrilinosWrappers::SolverDirect(solver_control, data));
         mumps_solver->initialize (mass_matrix);
      }
      
      // solve for ax
      {
         TrilinosWrappers::MPI::Vector tmp (locally_owned_dofs, mpi_communicator);
         mumps_solver->solve (tmp, rhs_ax);
         constraints.distribute (tmp);
         ax = tmp;
      }
      
      // solve for ay
      {
         TrilinosWrappers::MPI::Vector tmp (locally_owned_dofs, mpi_communicator);
         mumps_solver->solve (tmp, rhs_ay);
         constraints.distribute (tmp);
         ay = tmp;
      }
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::assemble_system_matrix_rhs ()
   {
      system_matrix_x = 0;
      system_matrix_y = 0;
      rhs_x           = 0;
      rhs_y           = 0;
      
      FEValues<dim> fe_values (fe, cell_quadrature,
                               update_values | update_gradients | update_JxW_values);
      const unsigned int   dofs_per_cell = fe.dofs_per_cell;
      const unsigned int   n_q_points    = cell_quadrature.size();
      FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
      Vector<double>       cell_rhs_x(dofs_per_cell), cell_rhs_y(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      
      std::vector<double> ax_values (n_q_points);
      std::vector<double> ay_values (n_q_points);
      std::vector<Tensor<1,dim>> Dx_values (n_q_points, Tensor<1,dim>());
      std::vector<Tensor<1,dim>> Dy_values (n_q_points, Tensor<1,dim>());
      
      for(typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          cell!=endc; ++cell)
         if(cell->is_locally_owned())
         {
            cell_matrix = 0;
            cell_rhs_x = 0;
            cell_rhs_y = 0;
            
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
            constraints_x.distribute_local_to_global(cell_matrix,
                                                     cell_rhs_x,
                                                     local_dof_indices,
                                                     system_matrix_x,
                                                     rhs_x);
            constraints_y.distribute_local_to_global(cell_matrix,
                                                     cell_rhs_y,
                                                     local_dof_indices,
                                                     system_matrix_y,
                                                     rhs_y);
         }
      
      rhs_x.compress (VectorOperation::add);
      rhs_y.compress (VectorOperation::add);
      system_matrix_x.compress (VectorOperation::add);
      system_matrix_y.compress (VectorOperation::add);
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::solve_xy ()
   {
      static TrilinosWrappers::SolverDirect::AdditionalData data (false, "Amesos_Mumps");
      static SolverControl solver_control (1, 0);

      // solve for x
      {
         TrilinosWrappers::MPI::Vector tmp (locally_owned_dofs, mpi_communicator);
         TrilinosWrappers::SolverDirect direct_x (solver_control, data);
         direct_x.solve (system_matrix_x, tmp, rhs_x);
         constraints_x.distribute (tmp);
         x = tmp;
      }
      
      // solve for y
      {
         TrilinosWrappers::MPI::Vector tmp (locally_owned_dofs, mpi_communicator);
         TrilinosWrappers::SolverDirect direct_y (solver_control, data);
         direct_y.solve (system_matrix_y, tmp, rhs_y);
         constraints_y.distribute (tmp);
         y = tmp;
      }
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   double Winslow<dim>::compute_change ()
   {
      TrilinosWrappers::MPI::Vector dx (locally_owned_dofs, mpi_communicator);
      dx  = x;
      dx -= x_old;
      double res_norm_x = dx.l2_norm();
      res_norm_x = std::sqrt( std::pow(res_norm_x,2) / dx.size() );
      
      TrilinosWrappers::MPI::Vector dy (locally_owned_dofs, mpi_communicator);
      dy  = y;
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
   void Winslow<dim>::fill_euler_vector (DoFHandler<dim>               &dh_euler,
                                         TrilinosWrappers::MPI::Vector &euler_vector)
   {
      unsigned int dofs_per_cell = fe.dofs_per_cell;
      unsigned int dofs_per_cell_euler = dh_euler.get_fe().dofs_per_cell;
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      std::vector<types::global_dof_index> euler_dof_indices (dofs_per_cell_euler);
      
      for(typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          cell!=endc; ++cell)
         if(cell->is_locally_owned())
         {
            typename DoFHandler<dim>::active_cell_iterator
            euler_cell (triangulation,
                        cell->level(),
                        cell->index(),
                        &dh_euler);
            cell->get_dof_indices (local_dof_indices);
            euler_cell->get_dof_indices (euler_dof_indices);
            
            for(unsigned int i=0; i<dofs_per_cell_euler; ++i)
            {
               unsigned int comp_i = dh_euler.get_fe().system_to_component_index(i).first;
               unsigned int indx_i = dh_euler.get_fe().system_to_component_index(i).second;
               if(comp_i == 0)
               {
                  euler_vector(euler_dof_indices[i]) = x(local_dof_indices[indx_i]);
               }
               else if(comp_i == 1)
               {
                  euler_vector(euler_dof_indices[i]) = y(local_dof_indices[indx_i]);
               }
               else
               {
                  AssertThrow(false, ExcMessage("Unknown component"));
               }

            }
         }
   }
   
   //------------------------------------------------------------------------------
   template <int dim>
   void Winslow<dim>::run(DoFHandler<dim>               &dh_euler,
                          TrilinosWrappers::MPI::Vector &euler_vector)
   {
      initialize_grid ();
      setup_system ();
      set_initial_condition ();
      map_boundary_values ();
      assemble_mass_matrix ();

      //output ();
      
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
         solve_xy ();
         
         res_norm = compute_change ();
         ++iter;
         pcout << iter << "  " << res_norm << std::endl;
         x_old = x;
         y_old = y;
         //output ();
      }
      
      if(res_norm > RESTOL)
      {
         pcout << "****************************************\n";
         pcout << "|  Picard iterations did not converge  |\n";
         pcout << "****************************************\n";
      }

      output_grids ();
      fill_euler_vector (dh_euler, euler_vector);
   }
   
   
} // end of namespace Winslow

// Instantiations
template class Winslow::Winslow<2>;
