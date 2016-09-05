/* ---------------------------------------------------------------------
 Solve the Winslow equations using Picard iterations as in
 Meire Fortunato, Per-Olof Persson
 High-order unstructured curved mesh generation using the Winslow equations
 Journal of Computational Physics 307 (2016) 1–14
 http://dx.doi.org/10.1016/j.jcp.2015.11.020
 --------------------------------------------------------------------- */


#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/distributed/tria.h>

using namespace dealii;

namespace Winslow
{
   inline
   void add_dirichlet_constraints (const std::map<types::global_dof_index,double> &values,
                                   ConstraintMatrix                               &constraints)
   {
      for (const auto &pair : values)
      {
         Assert(constraints.is_constrained(pair.first)==false,
                ExcMessage("dof already constrained"));
         constraints.add_line (pair.first);
         constraints.set_inhomogeneity (pair.first, pair.second);
      }
   }
   
   template <int dim>
   inline
   void g_matrix (const Tensor<1,dim>& Dx, const Tensor<1,dim>& Dy, Tensor<2,dim>& g)
   {
      g[0][0] = Dx[0] * Dx[0] + Dy[0] * Dy[0];
      g[0][1] = Dx[0] * Dx[1] + Dy[0] * Dy[1];
      g[1][0] = g[0][1];
      g[1][1] = Dx[1] * Dx[1] + Dy[1] * Dy[1];
   }
   
   template <int dim>
   inline
   Tensor<2,dim> ginvert (const Tensor<2,dim>& g)
   {
      Tensor<2,dim> gi;
      double c = 1.0;
      gi[0][0] =  c * g[1][1];
      gi[1][1] =  c * g[0][0];
      gi[0][1] = -c * g[0][1];
      gi[1][0] = -c * g[1][0];
      return gi;
   }
   
   template <int dim>
   inline
   void sort_points (std::vector<Point<dim>> &points)
   {
      std::vector<Point<dim>> tmp (points.size(), Point<dim>());
      
      tmp[0] = points[0];
      for(unsigned int i=2; i<points.size(); ++i)
         tmp[i-1] = points[i];
      tmp[points.size()-1] = points[1];
      
      for(unsigned int i=0; i<points.size(); ++i)
         points[i] = tmp[i];
   }
   
   template <int dim>
   class Winslow
   {
   public:
      Winslow (const unsigned int                         degree,
               parallel::distributed::Triangulation<dim> &triangulation);
      void run (DoFHandler<dim>                          &dh_euler,
                TrilinosWrappers::MPI::Vector            &euler_vector);
      
   private:
      void initialize_grid ();
      void setup_system ();
      void assemble_mass_matrix ();
      void assemble_alpha_rhs ();
      void assemble_system_matrix_rhs ();
      void set_initial_condition ();
      void map_boundary_values ();
      void solve_xy ();
      void solve_alpha ();
      double compute_change ();
      void output ();
      void output_grids ();
      void fill_euler_vector (DoFHandler<dim>               &dh_euler,
                              TrilinosWrappers::MPI::Vector &euler_vector);
      
      typedef parallel::distributed::Triangulation<dim> PDTriangulation;
      
      MPI_Comm                          mpi_communicator;
      PDTriangulation*                  triangulation;
      IndexSet                          locally_owned_dofs;
      IndexSet                          locally_relevant_dofs;
      FE_Q<dim>                         fe;
      DoFHandler<dim>                   dof_handler;
      
      TrilinosWrappers::MPI::Vector     x, y;
      TrilinosWrappers::MPI::Vector     x_old, y_old;
      TrilinosWrappers::MPI::Vector     ax, ay;
      
      ConstraintMatrix                  constraints;
      ConstraintMatrix                  constraints_x;
      ConstraintMatrix                  constraints_y;

      TrilinosWrappers::SparseMatrix    mass_matrix;
      TrilinosWrappers::SparseMatrix    system_matrix_x;
      TrilinosWrappers::SparseMatrix    system_matrix_y;
      
      TrilinosWrappers::MPI::Vector     rhs_x;
      TrilinosWrappers::MPI::Vector     rhs_y;
      TrilinosWrappers::MPI::Vector     rhs_ax;
      TrilinosWrappers::MPI::Vector     rhs_ay;
      
      std::map<types::global_dof_index,double> boundary_values_x;
      std::map<types::global_dof_index,double> boundary_values_y;

      const QGauss<dim>                 cell_quadrature;
      const QGauss<dim-1>               face_quadrature;
      
      ConditionalOStream                pcout;
      
      std_cxx11::shared_ptr<TrilinosWrappers::SolverDirect> mumps_solver;
   };
   
   template <int dim>
   inline
   void compute_mapping (const unsigned int                         degree,
                         parallel::distributed::Triangulation<dim> &triangulation,
                         DoFHandler<dim>                           &dh_euler,
                         TrilinosWrappers::MPI::Vector             &euler_vector)
   {
      IndexSet locally_owned_dofs;
      IndexSet locally_relevant_dofs;
      locally_owned_dofs = dh_euler.locally_owned_dofs ();
      DoFTools::extract_locally_relevant_dofs (dh_euler,
                                               locally_relevant_dofs);
      TrilinosWrappers::MPI::Vector distributed_euler_vector (locally_owned_dofs,
                                                              triangulation.get_communicator());
      
      Winslow<dim> winslow (degree, triangulation);
      winslow.run (dh_euler, distributed_euler_vector);
      euler_vector.reinit (locally_relevant_dofs, triangulation.get_communicator());
      euler_vector = distributed_euler_vector;
   }
   
}