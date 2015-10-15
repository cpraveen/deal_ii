/* 1d DG code for 10 moment gaussian closure model. 
   * This is not particularly efficient code.
   * Legendre basis functions
   * TVD/TVB limiter
   * moment limiter (BDF)
   * Modified Moment limiter (BSB)
   * Characteristic based limiter
   * Positivity preserving limiter
   * Numerical fluxes: Lax-Friedrich
   *
   * Authors: 
   *     Praveen. C, http://praveen.tifrbng.res.in
   *     Asha Meena
   *     Harish Kumar
*/
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/logstream.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Number of variables: mass, momentum and energy
const unsigned int n_var = 6;
double gas_gamma;
double gas_const;
double d_left, u1_left, u2_left, p11_left, p12_left, p22_left;
double d_right, u1_right, u2_right, p11_right, p12_right, p22_right;
double xmin, xmax, xmid;
double Mdx2; // for TVB limiter

// Coefficients for 3-stage SSP RK scheme of Shu-Osher
std::vector<double> a_rk, b_rk;

// Numerical flux functions
enum FluxType {lxf, roe, hllc};
enum TestCase {sod};


//------------------------------------------------------------------------------
// minmod of three numbers
//------------------------------------------------------------------------------
double minmod (const double& a, const double& b, const double& c)
{
   if(std::fabs(a) < Mdx2) return a;
   
   double result;
   if( a*b > 0.0 && b*c > 0.0)
   {
      result  = std::min( std::fabs(a), std::min(std::fabs(b), std::fabs(c)));
      result *= ((a>0.0) ? 1.0 : -1.0);
   }
   else 
   {
      result = 0.0;
   }
   
   return result;
}
//------------------------------------------------------------------------------
// maxmod of two numbers
// Author: Sudarshan Kumar K
//------------------------------------------------------------------------------
double maxmod (const double& a, const double& b)
{   
   double result;
   if( a*b >= 0.0)
      result  = (std::fabs(a) > std::fabs(b)) ? a : b;
   else 
      result = 0.0;
   
   return result;
}

//------------------------------------------------------------------------------
// Compute matrix of eigenvectors = R
// and its inverse matrix = Ri
//------------------------------------------------------------------------------
void EigMat(const double& d,
            const double& m1,
            const double& m2,
            const double& e11,
            const double& e12,
            const double& e22,
            double R[][n_var], double Ri[][n_var])
{
   std::cout << "EigMat not implemented !!!\n";
   AssertThrow(false, ExcMessage("EigMat not implemented !!!"));
}

//------------------------------------------------------------------------------
// U = R*U
//------------------------------------------------------------------------------
void Multi(double R[][n_var], std::vector<double>& U)
{
   std::vector<double> Ut(U);

   for(unsigned int i = 0; i < n_var; i++) 
   {
      U[i] = 0.0;
      for(unsigned int j = 0; j < n_var; j++)
         U[i] += R[i][j] * Ut[j];
   }
}
//------------------------------------------------------------------------------
// Exact solution
//------------------------------------------------------------------------------
template <int dim>
class ExactSolution : public Function<dim>
{
public:
   ExactSolution () : Function<dim>() {}
   
   virtual double value (const Point<dim>   &p, const unsigned int component = 0) const;
   virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                   const unsigned int  component = 0) const;
};

template<int dim>
double ExactSolution<dim>::value (const Point<dim>   &p, const unsigned int) const
{
   double x = p[0] - 2.0;
   return 1.0 + 0.2*sin(M_PI*x);
}

template <int dim>
Tensor<1,dim> ExactSolution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
   Tensor<1,dim> return_value(0.0);
   double x = p[0] - 2.0;
   return_value[0] = 0.2 * (M_PI) * cos(M_PI*x);
   return return_value;
}
//------------------------------------------------------------------------------
// Initial condition
// Returns primitive variables: (rho, u1, u2, p11, p12, p22)
//------------------------------------------------------------------------------
template <int dim>
class InitialCondition : public Function<dim>
{
public:
   InitialCondition () : Function<dim>() {}
   
   virtual void vector_value (const Point<dim>   &p,
                              Vector<double>& values) const;
   std::string test_case;
};

// Initial condition for density, velocity, pressure
template<int dim>
void InitialCondition<dim>::vector_value (const Point<dim>   &p,
                                          Vector<double>& values) const
{
   if(test_case == "sod")
   {
      if(p[0] < xmid)
      {
         values(0) = d_left;
         values(1) = u1_left;
         values(2) = u2_left;
         values(3) = p11_left;
         values(4) = p12_left;
         values(5) = p22_left;
      }
      else
      {
         values(0) = d_right;
         values(1) = u1_right;
         values(2) = u2_right;
         values(3) = p11_right;
         values(4) = p12_right;
         values(5) = p22_right;
      }
   }
   else
   {
      std::cout << "Unknown test case\n";
      exit(0);
   }

}

//------------------------------------------------------------------------------
// Main class of the problem
//------------------------------------------------------------------------------
template <int dim>
class Gauss10
{
public:
   Gauss10 (unsigned int degree, const ParameterHandler& prm);
   void run (double& h, int& ndof, double& L2_error, double& H1_error, double& Linf_error);
   
private:
   void make_grid_and_dofs ();
   void initialize ();
   void assemble_mass_matrix ();
   void assemble_rhs ();
   void compute_averages ();
   void compute_dt ();
   void apply_limiter ();
   void apply_limiter_TVB ();
   void apply_limiter_BDF ();
   void apply_limiter_BSB ();
   void apply_positivity_limiter ();
   void update (const unsigned int rk_stage);
   void output_results () const;
   void compute_errors (double& L2_error, double& H1_error, double& Linf_error) const;
   
   unsigned int         n_cells;
   std::string          test_case;
   double               dt;
   double               dx;
   double               cfl;
   double               beta;
   double               final_time;
   double               min_residue;
   unsigned int         max_iter;
   unsigned int         n_rk_stages;
   FluxType             flux_type;
   std::string          limiter;
   bool                 lim_char, lim_pos;
   bool                 lbc_reflect, rbc_reflect, periodic;
   unsigned int         save_freq;
   
   Triangulation<dim>   triangulation;
   FE_DGP<dim>          fe;
   DoFHandler<dim>      dof_handler;
   
   std::vector< Vector<double> > inv_mass_matrix;
   
   Vector<double>       density;
   Vector<double>       momentum1;
   Vector<double>       momentum2;
   Vector<double>       energy11;
   Vector<double>       energy12;
   Vector<double>       energy22;
   Vector<double>       density_old;
   Vector<double>       momentum1_old;
   Vector<double>       momentum2_old;
   Vector<double>       energy11_old;
   Vector<double>       energy12_old;
   Vector<double>       energy22_old;
   Vector<double>       rhs_density;
   Vector<double>       rhs_momentum1;
   Vector<double>       rhs_momentum2;
   Vector<double>       rhs_energy11;
   Vector<double>       rhs_energy12;
   Vector<double>       rhs_energy22;
   
   std::vector<double>  density_average;
   std::vector<double>  momentum1_average;
   std::vector<double>  momentum2_average;
   std::vector<double>  energy11_average;
   std::vector<double>  energy12_average;
   std::vector<double>  energy22_average;
   
   std::vector<double>  residual;
   std::vector<double>  residual0;

   typename DoFHandler<dim>::active_cell_iterator firstc, lastc;
   std::vector<typename DoFHandler<dim>::active_cell_iterator> lcell, rcell;
   
};

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <int dim>
Gauss10<dim>::Gauss10 (unsigned int degree,
                       const ParameterHandler& prm
                      ) :
   fe (degree),
   dof_handler (triangulation)
{
   Assert (dim==1, ExcIndexRange(dim, 0, 1));

   n_cells  = prm.get_integer("ncells");
   test_case= prm.get("test case");
   lim_char = prm.get_bool("characteristic limiter");
   lim_pos  = prm.get_bool("positivity limiter");
   cfl      = prm.get_double("cfl");
   double M = prm.get_double("M");
   beta     = prm.get_double("beta");
   save_freq= prm.get_integer("save frequency");
   limiter  = prm.get("limiter");
   std::string flux  = prm.get("flux");
   max_iter = prm.get_integer("max iter");
   
   if(limiter == "BDF" || limiter == "BSB") M = 0.0;
   
   n_rk_stages = std::min(degree,2u) + 1;
   a_rk.resize(n_rk_stages);
   b_rk.resize(n_rk_stages);
   if(n_rk_stages==1)
   {
      a_rk[0] = 0.0;
      b_rk[0] = 1.0;
   }
   else if(n_rk_stages==2)
   {
      a_rk[0] = 0.0; a_rk[1] = 0.5;
      b_rk[0] = 1.0; b_rk[1] = 0.5;
   }
   else if(n_rk_stages==3)
   {
      a_rk[0] = 0.0; a_rk[1] = 3.0/4.0; a_rk[2] = 1.0/3.0;
      b_rk[0] = 1.0; b_rk[1] = 1.0/4.0; b_rk[2] = 2.0/3.0;
   }
   else
   {
      std::cout << "Number of RK stages not set correctly.\n";
      exit(0);
   }

   // Set flux enum type
   if(flux == "lxf")
      flux_type = lxf;
   else
   {
      std::cout << "Numerical flux type is not set\n";
      exit(0);
   }
   
   lbc_reflect = rbc_reflect = periodic = false;
   min_residue= 1.0e20;
   
   if(test_case == "sod")
   {
      xmin    = 0.0;
      xmax    = 1.0;
      xmid    = 0.5;
      final_time = 0.2;
      
      d_left  = 1.0;
      d_right = 0.125;
      
      u1_left  = 0.0;
      u1_right = 0.0;
      
      u2_left  = 0.0;
      u2_right = 0.0;
      
      p11_left  = 1.0;
      p11_right = 0.1;

      p12_left  = 0.0;
      p12_right = 0.0;

      p22_left  = 1.0;
      p22_right = 0.1;
   }
   else
   {
      std::cout << "Unknown test case " << test_case << "\n";
   }
   
   cfl *= 1.0/(2.0*fe.degree + 1.0);
   dx   = (xmax - xmin) / n_cells;
   Mdx2 = M * dx * dx;

}

//------------------------------------------------------------------------------
// Make grid and allocate memory for solution variables
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::make_grid_and_dofs ()
{
    GridGenerator::subdivided_hyper_cube (triangulation, n_cells, xmin, xmax);

    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Total number of cells: "
              << triangulation.n_cells()
              << std::endl;

    dof_handler.distribute_dofs (fe);

    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

   // allocate memory for inverse mass matrix. We store only diagonals.
   inv_mass_matrix.resize(n_cells);
   for (unsigned int c=0; c<n_cells; ++c)
      inv_mass_matrix[c].reinit(fe.degree+1);
   
    // Solution variables
    density.reinit (dof_handler.n_dofs());
    density_old.reinit (dof_handler.n_dofs());
    rhs_density.reinit (dof_handler.n_dofs());
   
    momentum1.reinit (dof_handler.n_dofs());
    momentum1_old.reinit (dof_handler.n_dofs());
    rhs_momentum1.reinit (dof_handler.n_dofs());
   
    momentum2.reinit (dof_handler.n_dofs());
    momentum2_old.reinit (dof_handler.n_dofs());
    rhs_momentum2.reinit (dof_handler.n_dofs());
   
    energy11.reinit (dof_handler.n_dofs());
    energy11_old.reinit (dof_handler.n_dofs());
    rhs_energy11.reinit (dof_handler.n_dofs());   
   
    energy12.reinit (dof_handler.n_dofs());
    energy12_old.reinit (dof_handler.n_dofs());
    rhs_energy12.reinit (dof_handler.n_dofs());   
   
    energy22.reinit (dof_handler.n_dofs());
    energy22_old.reinit (dof_handler.n_dofs());
    rhs_energy22.reinit (dof_handler.n_dofs());   
   
    density_average.resize (triangulation.n_cells());
    momentum1_average.resize (triangulation.n_cells());
    momentum2_average.resize (triangulation.n_cells());
    energy11_average.resize (triangulation.n_cells());
    energy12_average.resize (triangulation.n_cells());
    energy22_average.resize (triangulation.n_cells());
   
    residual.resize(n_var, 1.0);
    residual0.resize(n_var);
   
   // Find first and last cell
   // We need these for periodic bc
   // WARNING: This could be wrong with adaptive refinement.
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   firstc = dof_handler.begin_active();
   for (unsigned int c=0; cell!=endc; ++cell, ++c)
   {
      if(c == triangulation.n_active_cells()-1)
         lastc = cell;
   }
   
   // for each cell find left cell and right cell
   lcell.resize(n_cells);
   rcell.resize(n_cells);
   cell = dof_handler.begin_active();
   for (unsigned int c=0; cell!=endc; ++cell, ++c)
   {
      if(c==0)
      {
         rcell[n_cells-1] = cell;
         lcell[c+1] = cell;
      }
      else if(c==n_cells-1)
      {
         rcell[c-1] = cell;
         lcell[0] = cell;
      }
      else
      {
         rcell[c-1] = cell;
         lcell[c+1] = cell;
      }
      
   }
}

//------------------------------------------------------------------------------
// Set initial conditions
// L2 projection of initial condition onto dofs
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::initialize ()
{
   std::cout << "Projecting initial condition ...\n";
   
   QGauss<dim>  quadrature_formula(fe.degree+1);
   
   FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values   |
                            update_quadrature_points | 
                            update_JxW_values);
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   const unsigned int   n_q_points    = quadrature_formula.size();
   
   Vector<double>       cell_rhs_density  (dofs_per_cell);
   Vector<double>       cell_rhs_momentum1 (dofs_per_cell);
   Vector<double>       cell_rhs_momentum2 (dofs_per_cell);
   Vector<double>       cell_rhs_energy11 (dofs_per_cell);
   Vector<double>       cell_rhs_energy12 (dofs_per_cell);
   Vector<double>       cell_rhs_energy22 (dofs_per_cell);
   
   
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   
   InitialCondition<dim> initial_condition;
   initial_condition.test_case = test_case;
   
   Vector<double> initial_value(n_var);
   double initial_density;
   double initial_momentum1;
   double initial_momentum2;
   double initial_energy11;
   double initial_energy12;
   double initial_energy22;

   typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                  endc = dof_handler.end();
   for (unsigned int c=0; cell!=endc; ++cell, ++c)
   {
      fe_values.reinit (cell);
      
      cell_rhs_density   = 0.0;
      cell_rhs_momentum1 = 0.0;
      cell_rhs_momentum2 = 0.0;
      cell_rhs_energy11  = 0.0;
      cell_rhs_energy12  = 0.0;
      cell_rhs_energy22  = 0.0;
      
      
      // Flux integral over cell
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
         // Get primitive variable at quadrature point
         initial_condition.vector_value(fe_values.quadrature_point(q_point),
                                        initial_value);
         // Convert primitive to conserved
         initial_density  = initial_value(0);
         initial_momentum1= initial_value(0) * initial_value(1);
         initial_momentum2= initial_value(0) * initial_value(2);
         initial_energy11 = 0.5 * initial_value(3) + 
                            0.5 * initial_value(0) * pow(initial_value(1),2);
         initial_energy12 = 0.5 * initial_value(4) + 
                            0.5 * initial_value(0) * initial_value(1) * initial_value(2);
         initial_energy22 = 0.5 * initial_value(5) + 
                            0.5 * initial_value(0) * pow(initial_value(2),2);
         for (unsigned int i=0; i<dofs_per_cell; ++i)
         {
            cell_rhs_density(i) += (fe_values.shape_value (i, q_point) *
                                    initial_density *
                                    fe_values.JxW (q_point));
            cell_rhs_momentum1(i)+= (fe_values.shape_value (i, q_point) *
                                    initial_momentum1 *
                                    fe_values.JxW (q_point));
            cell_rhs_momentum2(i)+= (fe_values.shape_value (i, q_point) *
                                    initial_momentum2 *
                                    fe_values.JxW (q_point));
            cell_rhs_energy11(i)  += (fe_values.shape_value (i, q_point) *
                                     initial_energy11 *
                                     fe_values.JxW (q_point));
            cell_rhs_energy12(i)  += (fe_values.shape_value (i, q_point) *
                                     initial_energy12 *
                                     fe_values.JxW (q_point));
            cell_rhs_energy22(i)  += (fe_values.shape_value (i, q_point) *
                                     initial_energy22 *
                                     fe_values.JxW (q_point));
         }
      }
      
      
      // Multiply by inverse mass matrix and add to rhs
      cell->get_dof_indices (local_dof_indices);
      unsigned int ig;
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
         ig = local_dof_indices[i];
         
         density(ig)  = inv_mass_matrix[c](i) * cell_rhs_density(i);
         momentum1(ig) = inv_mass_matrix[c](i) * cell_rhs_momentum1(i);
         momentum2(ig) = inv_mass_matrix[c](i) * cell_rhs_momentum2(i);
         energy11(ig)  = inv_mass_matrix[c](i) * cell_rhs_energy11(i);
         energy12(ig)  = inv_mass_matrix[c](i) * cell_rhs_energy12(i);
         energy22(ig)  = inv_mass_matrix[c](i) * cell_rhs_energy22(i);
      }
   }
}

//------------------------------------------------------------------------------
// Assemble mass matrix for each cell
// Invert it and store
// For Legendre basis, mass matrix is diagonal
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::assemble_mass_matrix ()
{
   std::cout << "Constructing mass matrix ...\n";
   std::cout << "  Quadrature using " << fe.degree+1 << " points\n";
   
   QGauss<dim>  quadrature_formula(fe.degree+1);
   
   FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values | update_JxW_values);
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   const unsigned int   n_q_points    = quadrature_formula.size();
   
   Vector<double>   cell_matrix (dofs_per_cell);
   
   // Cell iterator
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   for (unsigned int c=0; cell!=endc; ++cell,++c)
   {
      fe_values.reinit (cell);
      cell_matrix = 0.0;
      
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         for (unsigned int i=0; i<dofs_per_cell; ++i)
            cell_matrix(i) += fe_values.shape_value (i, q_point) *
                              fe_values.shape_value (i, q_point) *
                              fe_values.JxW (q_point);
      
      for (unsigned int i=0; i<dofs_per_cell; ++i)
         inv_mass_matrix[c](i) = 1.0/cell_matrix(i);
   }
}


//------------------------------------------------------------------------------
// Flux for Euler equation
//------------------------------------------------------------------------------
void euler_flux (const double& density,
                 const double& momentum1,
                 const double& momentum2,
                 const double& energy11,
                 const double& energy12,
                 const double& energy22,
                 Vector<double>& flux)
{   
   double velocity1 = momentum1 / density;
   double velocity2 = momentum2 / density;
   double pressure11 = 2.0 * (energy11 - 0.5 * density * velocity1 * velocity1);
   double pressure12 = 2.0 * (energy12 - 0.5 * density * velocity1 * velocity2);

   flux(0) = momentum1;
   flux(1) = pressure11 + density * velocity1 * velocity1;
   flux(2) = pressure12 + density * velocity1 * velocity2;
   flux(3) = (energy11 + pressure11) * velocity1;
   flux(4) = energy12 * velocity1 + 0.5*(pressure11 * velocity2 + pressure12 * velocity1);
   flux(5) = energy22 * velocity1 + pressure12 * velocity2;
}

//------------------------------------------------------------------------------
// Lax-Friedrichs flux
//------------------------------------------------------------------------------
void LaxFlux (const Vector<double>& left_state,
              const Vector<double>& right_state,
              Vector<double>& flux)
{
   // Left state
   double left_velocity1 = left_state(1) / left_state(0);
   double left_velocity2 = left_state(2) / left_state(0);
   double left_pressure11 =  2.0 * left_state(3) -
                             left_state(1) * left_velocity1;
   double left_pressure12 =  2.0 * left_state(4) -
                             left_state(1) * left_velocity2;
   
   double left_sonic  = sqrt( 3.0 * left_pressure11 / left_state(0) );
   double left_eig_max= fabs(left_velocity1) + left_sonic;
   
   // Left flux
   Vector<double> left_flux(n_var);
   left_flux(0) = left_state(1);
   left_flux(1) = left_pressure11 + left_state(1) * left_velocity1;
   left_flux(2) = left_pressure12 + left_state(1) * left_velocity2;
   left_flux(3) = (left_state(3) + left_pressure11) * left_velocity1;
   left_flux(4) = left_state(4) * left_velocity1 +
                   0.5 * (left_pressure11 * left_velocity2 + left_pressure12 * left_velocity1);
   left_flux(5) = left_state(5) * left_velocity1 + left_pressure12 * left_velocity2;
   
   // Right state
   double right_velocity1 = right_state(1) / right_state(0);
   double right_velocity2 = right_state(2) / right_state(0);
   double right_pressure11 = 2.0 * right_state(3) -
                              right_state(1) * right_velocity1;
   double right_pressure12 = 2.0 * right_state(4) -
                              right_state(1) * right_velocity2;
   
   double right_sonic  = sqrt( 3.0 * right_pressure11 / right_state(0) );
   double right_eig_max= fabs(right_velocity1) + right_sonic;
   
   // Right flux
   Vector<double> right_flux(n_var);
   right_flux(0) = right_state(1);
   right_flux(1) = right_pressure11 + right_state(1) * right_velocity1;
   right_flux(2) = right_pressure12 + right_state(1) * right_velocity2;
   right_flux(3) = (right_state(3) + right_pressure11) * right_velocity1;
   right_flux(4) = right_state(4) * right_velocity1 +
                   0.5 * (right_pressure11 * right_velocity2 + right_pressure12 * right_velocity1);
   right_flux(5) = right_state(5) * right_velocity1 + right_pressure12 * right_velocity2;
   
   // Maximum local wave speed at face
   double lambda = std::max ( left_eig_max, right_eig_max );
   
   for(unsigned int i=0; i<n_var; ++i)
      flux(i) = 0.5 * ( left_flux(i) + right_flux(i) ) -
                0.5 * lambda * ( right_state(i) - left_state(i) );
}

//------------------------------------------------------------------------------
// Compute flux across cell faces
//------------------------------------------------------------------------------
void numerical_flux (const FluxType& flux_type,
                     Vector<double>& left_state,
                     Vector<double>& right_state,
                     Vector<double>& flux)
{
   switch (flux_type) 
   {
      case lxf: // Lax-Friedrich flux
         LaxFlux (left_state, right_state, flux);
         break;
         
      default:
         std::cout << "Unknown flux_type !!!\n";
         abort ();
   }
}

//------------------------------------------------------------------------------
// Assemble system rhs
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::assemble_rhs ()
{
    QGaussLobatto<dim>  quadrature_formula(fe.degree+2);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | 
                             update_JxW_values);

   // for getting neighbour cell solutions to compute intercell flux
   QTrapez<dim> quadrature_dummy;
   FEValues<dim> fe_values_neighbor (fe, quadrature_dummy,
                            update_values   | update_gradients);
   
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    std::vector<double>  density_values  (n_q_points);
    std::vector<double>  momentum1_values (n_q_points);
    std::vector<double>  momentum2_values (n_q_points);
    std::vector<double>  energy11_values  (n_q_points);
    std::vector<double>  energy12_values  (n_q_points);
    std::vector<double>  energy22_values  (n_q_points);
   
   // for getting neighbor cell solution using trapezoidal rule
   std::vector<double>  density_values_n   (2);
   std::vector<double>  momentum1_values_n (2);
   std::vector<double>  momentum2_values_n (2);
   std::vector<double>  energy11_values_n  (2);
   std::vector<double>  energy12_values_n  (2);
   std::vector<double>  energy22_values_n  (2);

    Vector<double>       cell_rhs_density  (dofs_per_cell);
    Vector<double>       cell_rhs_momentum1 (dofs_per_cell);
    Vector<double>       cell_rhs_momentum2 (dofs_per_cell);
    Vector<double>       cell_rhs_energy11 (dofs_per_cell);
    Vector<double>       cell_rhs_energy12 (dofs_per_cell);
    Vector<double>       cell_rhs_energy22 (dofs_per_cell);
   
    Vector<double>       flux(n_var);

    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   for(unsigned int i=0; i<n_var; ++i)
      residual[i] = 0.0;
   
    for (unsigned int c=0; cell!=endc; ++cell, ++c)
    {
        fe_values.reinit (cell);
       
        cell_rhs_density  = 0.0;
        cell_rhs_momentum1 = 0.0;
        cell_rhs_momentum2 = 0.0;
        cell_rhs_energy11 = 0.0;
        cell_rhs_energy12 = 0.0;
        cell_rhs_energy22 = 0.0;

        // Compute conserved variables at quadrature points
        fe_values.get_function_values (density,   density_values);
        fe_values.get_function_values (momentum1, momentum1_values);
        fe_values.get_function_values (momentum2, momentum2_values);
        fe_values.get_function_values (energy11,  energy11_values);
        fe_values.get_function_values (energy12,  energy12_values);
        fe_values.get_function_values (energy22,  energy22_values);

        // Flux integral over cell
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
            euler_flux(density_values[q_point], 
                       momentum1_values[q_point],
                       momentum2_values[q_point],
                       energy11_values[q_point], 
                       energy12_values[q_point], 
                       energy22_values[q_point], 
                       flux);

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                cell_rhs_density(i) += (fe_values.shape_grad (i, q_point)[0] *
                                        flux(0) *
                                        fe_values.JxW (q_point));
                cell_rhs_momentum1(i)+= (fe_values.shape_grad (i, q_point)[0] *
                                        flux(1) *
                                        fe_values.JxW (q_point));
                cell_rhs_momentum2(i)+= (fe_values.shape_grad (i, q_point)[0] *
                                        flux(2) *
                                        fe_values.JxW (q_point));
                cell_rhs_energy11(i)  += (fe_values.shape_grad (i, q_point)[0] *
                                         flux(3) *
                                         fe_values.JxW (q_point));
                cell_rhs_energy12(i)  += (fe_values.shape_grad (i, q_point)[0] *
                                         flux(4) *
                                         fe_values.JxW (q_point));
                cell_rhs_energy22(i)  += (fe_values.shape_grad (i, q_point)[0] *
                                         flux(5) *
                                         fe_values.JxW (q_point));
            }
        }
       
       // Computation of flux at cell boundaries
       Vector<double> lf_left_state(n_var), lf_right_state(n_var);
       
        // left face flux
        // right state is from current cell
       lf_right_state(0) = density_values  [0];
       lf_right_state(1) = momentum1_values[0];
       lf_right_state(2) = momentum2_values[0];
       lf_right_state(3) = energy11_values [0];
       lf_right_state(4) = energy12_values [0];
       lf_right_state(5) = energy22_values [0];
       
       if(c==0 && periodic==false)
       {
          if(lbc_reflect)
          {
             lf_left_state(0) = lf_right_state(0);
             lf_left_state(1) =-lf_right_state(1);
             lf_left_state(2) = lf_right_state(2);
             lf_left_state(3) = lf_right_state(3);
             lf_left_state(4) = lf_right_state(4);
             lf_left_state(5) = lf_right_state(5);
          }
          else
          {
             lf_left_state(0) = lf_right_state(0);
             lf_left_state(1) = lf_right_state(1);
             lf_left_state(2) = lf_right_state(2);
             lf_left_state(3) = lf_right_state(3);
             lf_left_state(4) = lf_right_state(4);
             lf_left_state(5) = lf_right_state(5);
          }
       }
       else
       {
          // get left cell dof indices
          //fe_values_neighbor.reinit (cell->neighbor(0));
          fe_values_neighbor.reinit (lcell[c]);
          
          fe_values_neighbor.get_function_values (density,   density_values_n);
          fe_values_neighbor.get_function_values (momentum1, momentum1_values_n);
          fe_values_neighbor.get_function_values (momentum2, momentum2_values_n);
          fe_values_neighbor.get_function_values (energy11,  energy11_values_n);
          fe_values_neighbor.get_function_values (energy12,  energy12_values_n);
          fe_values_neighbor.get_function_values (energy22,  energy22_values_n);
          
          lf_left_state(0) = density_values_n  [1];
          lf_left_state(1) = momentum1_values_n[1];
          lf_left_state(2) = momentum2_values_n[1];
          lf_left_state(3) = energy11_values_n [1];
          lf_left_state(4) = energy12_values_n [1];
          lf_left_state(5) = energy22_values_n [1];
       }
       
       Vector<double> left_flux(n_var);
       numerical_flux (flux_type, lf_left_state, lf_right_state, left_flux);
       
       // right face flux
       Vector<double> rf_left_state(n_var), rf_right_state(n_var);

       // left state is from current cell
       rf_left_state(0) = density_values  [n_q_points-1];
       rf_left_state(1) = momentum1_values[n_q_points-1];
       rf_left_state(2) = momentum2_values[n_q_points-1];
       rf_left_state(3) = energy11_values [n_q_points-1];
       rf_left_state(4) = energy12_values [n_q_points-1];
       rf_left_state(5) = energy22_values [n_q_points-1];
       
       if(c==triangulation.n_cells()-1 && periodic==false)
       {
          if(rbc_reflect)
          {
             rf_right_state(0) = rf_left_state(0);
             rf_right_state(1) =-rf_left_state(1);
             rf_right_state(2) = rf_left_state(2);
             rf_right_state(3) = rf_left_state(3);
             rf_right_state(4) = rf_left_state(4);
             rf_right_state(5) = rf_left_state(5);
          }
          else
          {
             rf_right_state(0) = rf_left_state(0);
             rf_right_state(1) = rf_left_state(1);
             rf_right_state(2) = rf_left_state(2);
             rf_right_state(3) = rf_left_state(3);
             rf_right_state(4) = rf_left_state(4);
             rf_right_state(5) = rf_left_state(5);
          }
       }
       else
       {          
          // get right cell to right face
          //fe_values_neighbor.reinit (cell->neighbor(1));
          fe_values_neighbor.reinit (rcell[c]);
          
          fe_values_neighbor.get_function_values (density,   density_values_n);
          fe_values_neighbor.get_function_values (momentum1, momentum1_values_n);
          fe_values_neighbor.get_function_values (momentum2, momentum2_values_n);
          fe_values_neighbor.get_function_values (energy11,  energy11_values_n);
          fe_values_neighbor.get_function_values (energy12,  energy12_values_n);
          fe_values_neighbor.get_function_values (energy22,  energy22_values_n);
          
          rf_right_state(0) = density_values_n  [0];
          rf_right_state(1) = momentum1_values_n[0];
          rf_right_state(2) = momentum2_values_n[0];
          rf_right_state(3) = energy11_values_n [0];
          rf_right_state(4) = energy12_values_n [0];
          rf_right_state(5) = energy22_values_n [0];
       }
       
       Vector<double> right_flux(n_var);
       numerical_flux (flux_type, rf_left_state, rf_right_state, right_flux);
       
        // Add flux at cell boundaries
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
           // Left face flux
           cell_rhs_density(i) += fe_values.shape_value (i, 0) *
                                  left_flux(0);
           cell_rhs_momentum1(i)+= fe_values.shape_value (i, 0) *
                                   left_flux(1);
           cell_rhs_momentum2(i)+= fe_values.shape_value (i, 0) *
                                   left_flux(2);
           cell_rhs_energy11(i) += fe_values.shape_value (i, 0) *
                                   left_flux(3);
           cell_rhs_energy12(i) += fe_values.shape_value (i, 0) *
                                   left_flux(4);
           cell_rhs_energy22(i) += fe_values.shape_value (i, 0) *
                                   left_flux(5);
           
           // Right face flux
           cell_rhs_density(i) -= fe_values.shape_value (i, n_q_points-1) *
                                  right_flux(0);
           cell_rhs_momentum1(i)-= fe_values.shape_value (i, n_q_points-1) *
                                   right_flux(1);
           cell_rhs_momentum2(i)-= fe_values.shape_value (i, n_q_points-1) *
                                   right_flux(2);
           cell_rhs_energy11(i) -= fe_values.shape_value (i, n_q_points-1) *
                                   right_flux(3);
           cell_rhs_energy12(i) -= fe_values.shape_value (i, n_q_points-1) *
                                   right_flux(4);
           cell_rhs_energy22(i) -= fe_values.shape_value (i, n_q_points-1) *
                                   right_flux(5);
           
        }

        // Multiply by inverse mass matrix and add to rhs
        cell->get_dof_indices (local_dof_indices);
        unsigned int ig;
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
           ig = local_dof_indices[i];
           
           rhs_density(ig)   = inv_mass_matrix[c](i) * cell_rhs_density(i);
           rhs_momentum1(ig) = inv_mass_matrix[c](i) * cell_rhs_momentum1(i);
           rhs_momentum2(ig) = inv_mass_matrix[c](i) * cell_rhs_momentum2(i);
           rhs_energy11(ig)  = inv_mass_matrix[c](i) * cell_rhs_energy11(i);
           rhs_energy12(ig)  = inv_mass_matrix[c](i) * cell_rhs_energy12(i);
           rhs_energy22(ig)  = inv_mass_matrix[c](i) * cell_rhs_energy22(i);
           
           residual[0] += std::pow (rhs_density (ig), 2);
           residual[1] += std::pow (rhs_momentum1 (ig), 2);
           residual[2] += std::pow (rhs_momentum2 (ig), 2);
           residual[3] += std::pow (rhs_energy11 (ig), 2);
           residual[4] += std::pow (rhs_energy12 (ig), 2);
           residual[5] += std::pow (rhs_energy22 (ig), 2);
        }
       
    }

}

//------------------------------------------------------------------------------
// Compute cell average values
// For Legendre, first dof is cell average value
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::compute_averages ()
{
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   
   typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   for (unsigned int c=0; cell!=endc; ++c, ++cell)
   {
      cell->get_dof_indices (local_dof_indices);
      
      density_average[c]   = density  (local_dof_indices[0]);
      momentum1_average[c] = momentum1 (local_dof_indices[0]);
      momentum2_average[c] = momentum2 (local_dof_indices[0]);
      energy11_average[c]  = energy11 (local_dof_indices[0]);
      energy12_average[c]  = energy12 (local_dof_indices[0]);
      energy22_average[c]  = energy22 (local_dof_indices[0]);
   }
}

//------------------------------------------------------------------------------
// Apply chosen limiter
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::apply_limiter ()
{
   
   if(limiter == "TVB")
      apply_limiter_TVB ();
   /* else if(limiter == "BDF")
      apply_limiter_BDF ();
   else if(limiter == "BSB")
      apply_limiter_BSB ();
   else if(limiter == "None" || limiter == "visc")
      return; */
   else
   {
      std::cout << "Unknown limiter\n";
      exit(0);
   }
   
}

//------------------------------------------------------------------------------
// Apply TVD limiter
//------------------------------------------------------------------------------

template <int dim>
void Gauss10<dim>::apply_limiter_TVB ()
{
   if(fe.degree == 0) return;
   
   QTrapez<dim>  quadrature_formula;
   
   FEValues<dim> fe_values (fe, quadrature_formula, update_values);
   std::vector<double> density_face_values(2), momentum1_face_values(2),
                       momentum2_face_values(2), energy11_face_values(2),
                       energy12_face_values(2), energy22_face_values(2);
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;   
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   
   typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   std::vector<double> db(n_var), df(n_var), DB(n_var), DF(n_var);
   std::vector<double> dl(n_var), dr(n_var);
   double density_left, density_right;
   double momentum1_left, momentum1_right;
   double momentum2_left, momentum2_right;
   double energy11_left, energy11_right;
   double energy12_left, energy12_right;
   double energy22_left, energy22_right; 
   
   for (unsigned int c=0; c<n_cells; ++c, ++cell)
   {
      fe_values.reinit(cell);
      cell->get_dof_indices (local_dof_indices);
      fe_values.get_function_values(density, density_face_values);
      fe_values.get_function_values(momentum1, momentum1_face_values);
      fe_values.get_function_values(momentum2, momentum2_face_values);
      fe_values.get_function_values(energy11, energy11_face_values);
      fe_values.get_function_values(energy12, energy12_face_values);
      fe_values.get_function_values(energy22, energy22_face_values);  
      
      unsigned int lc = (c==0) ? n_cells-1 : c-1;
      unsigned int rc = (c==n_cells-1) ? 0 : c+1;
      
      if(c==0 && !periodic)
      {
         density_left = density_average[c];
         if(lbc_reflect)
            momentum1_left = -momentum1_average[c];
         else
            momentum1_left = momentum1_average[c];
         momentum2_left = momentum2_average[c];
         energy11_left = energy11_average[c];
         energy12_left = energy12_average[c];
         energy22_left = energy22_average[c];
         
         density_right = density_average[c+1];
         momentum1_right = momentum1_average[c+1];
         momentum2_right = momentum2_average[c+1];
         energy11_right = energy11_average[c+1];
         energy12_right = energy12_average[c+1];
         energy22_right = energy22_average[c+1];
      }
      else if(c == n_cells-1 && !periodic)
      {
         density_left = density_average[c-1];
         momentum1_left = momentum1_average[c-1];
         momentum2_left = momentum2_average[c-1];
         energy11_left = energy11_average[c-1];
         energy12_left = energy12_average[c-1];
         energy22_left = energy22_average[c-1]; 
         
         density_right = density_average[c];
         if(rbc_reflect)
            momentum1_right = -momentum1_average[c];
         else
            momentum1_right = momentum1_average[c];
         momentum2_right = momentum2_average[c]; 
         energy11_right = energy11_average[c];
         energy12_right = energy12_average[c];
         energy22_right = energy22_average[c];
      }
      else
      {
         density_left = density_average[lc];
         momentum1_left = momentum1_average[lc];
         momentum2_left = momentum2_average[lc];
         energy11_left = energy11_average[lc];
         energy12_left = energy12_average[lc];
         energy22_left = energy22_average[lc]; 
         
         density_right = density_average[rc];
         momentum1_right = momentum1_average[rc];
         momentum2_right = momentum2_average[rc];
         energy11_right = energy11_average[rc];
         energy12_right = energy12_average[rc];
         energy22_right = energy22_average[rc];
      }
      
      // density
      db[0] = density_average[c] - density_left;
      df[0] = density_right - density_average[c];
      DB[0] = density_average[c] - density_face_values[0];
      DF[0] = density_face_values[1] - density_average[c];
      
      // momentum1
      db[1] = momentum1_average[c] - momentum1_left;
      df[1] = momentum1_right - momentum1_average[c];
      DB[1] = momentum1_average[c] - momentum1_face_values[0];
      DF[1] = momentum1_face_values[1] - momentum1_average[c];

      // momentum2
      db[2] = momentum2_average[c] - momentum2_left;
      df[2] = momentum2_right - momentum2_average[c];
      DB[2] = momentum2_average[c] - momentum2_face_values[0];
      DF[2] = momentum2_face_values[1] - momentum2_average[c];
      
      // energy11
      db[3] = energy11_average[c] - energy11_left;
      df[3] = energy11_right - energy11_average[c];
      DB[3] = energy11_average[c] - energy11_face_values[0];
      DF[3] = energy11_face_values[1] - energy11_average[c];

      // energy12
      db[4] = energy12_average[c] - energy12_left;
      df[4] = energy12_right - energy12_average[c];
      DB[4] = energy12_average[c] - energy12_face_values[0];
      DF[4] = energy12_face_values[1] - energy12_average[c];

      // energy22
      db[5] = energy22_average[c] - energy22_left;
      df[5] = energy22_right - energy22_average[c];
      DB[5] = energy22_average[c] - energy22_face_values[0];
      DF[5] = energy22_face_values[1] - energy22_average[c];

       double R[n_var][n_var], Ri[n_var][n_var];
      if(lim_char)
      {
         EigMat(density_average[c], 
                momentum1_average[c],
                momentum2_average[c],
                energy11_average[c],
                energy12_average[c],
                energy22_average[c],
                R, Ri);
         Multi(Ri, db);
         Multi(Ri, df);
         Multi(Ri, DB);
         Multi(Ri, DF);
      }

      double diff = 0;
      for(unsigned int i=0; i<n_var; ++i)
      {
         dl[i] = minmod (DB[i], beta*db[i], beta*df[i]);
         dr[i] = minmod (DF[i], beta*db[i], beta*df[i]);
         diff += std::fabs(dl[i] - DB[i]) + std::fabs(dr[i]-DF[i]);
      }
      diff /= (2*n_var);

      // If diff is nonzero, then limiter is active.
      // Then we keep only linear part
      if(diff > 1.0e-10)
      {
         if(lim_char) 
         {
            Multi(R, dl);
            Multi(R, dr);
         }
         density(local_dof_indices[1])  = 0.5*(dl[0] + dr[0]) / fe_values.shape_value(1,1);
         momentum1(local_dof_indices[1]) = 0.5*(dl[1] + dr[1]) / fe_values.shape_value(1,1);
         momentum2(local_dof_indices[1]) = 0.5*(dl[2] + dr[2]) / fe_values.shape_value(1,1);
         energy11(local_dof_indices[1])  = 0.5*(dl[3] + dr[3]) / fe_values.shape_value(1,1);
         energy12(local_dof_indices[1])  = 0.5*(dl[4] + dr[4]) / fe_values.shape_value(1,1);
         energy22(local_dof_indices[1])  = 0.5*(dl[5] + dr[5]) / fe_values.shape_value(1,1);

         // Higher dofs are set to zero
         for(unsigned int i=2; i<dofs_per_cell; ++i)
         {
            density(local_dof_indices[i])   = 0.0;
            momentum1(local_dof_indices[i]) = 0.0;
            momentum2(local_dof_indices[i]) = 0.0;
            energy11(local_dof_indices[i])  = 0.0;
            energy12(local_dof_indices[i])  = 0.0; 
            energy22(local_dof_indices[i])  = 0.0;
         }
      }
      
   }
}
/*
//------------------------------------------------------------------------------
// Apply moment limiter of Biswas, Devine, Flaherty
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::apply_limiter_BDF ()
{
   if(fe.degree == 0) return;
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;   
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   std::vector<unsigned int> left_dof_indices (dofs_per_cell);
   std::vector<unsigned int> right_dof_indices (dofs_per_cell);
   
   std::vector< std::vector<double> > db(dofs_per_cell, std::vector<double>(n_var));
   std::vector< std::vector<double> > df(dofs_per_cell, std::vector<double>(n_var));
   std::vector< std::vector<double> > DC(dofs_per_cell, std::vector<double>(n_var));

   // Temporary storage
   Vector<double> density_n(density);
   Vector<double> momentum1_n(momentum1);
   Vector<double> energy_n(energy);

   typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   for (unsigned int c=0; c<n_cells; ++c, ++cell)
   {
      cell->get_dof_indices (local_dof_indices);
      
      if(c==0 && !periodic)
      {
         rcell[c]->get_dof_indices (right_dof_indices);
         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            db[i][0] = 0.0;
            if(lbc_reflect)
               db[i][1] = 2.0*momentum1(local_dof_indices[i]);
            else
               db[i][1] = 0.0;
            db[i][2] = 0.0;

            df[i][0] = density(right_dof_indices[i]) - density(local_dof_indices[i]);
            df[i][1] = momentum1(right_dof_indices[i]) - momentum1(local_dof_indices[i]);
            df[i][2] = energy(right_dof_indices[i]) - energy(local_dof_indices[i]);
         }
      }
      else if(c == n_cells-1 && !periodic)
      {
         lcell[c]->get_dof_indices (left_dof_indices);
         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            db[i][0] = density(local_dof_indices[i]) - density(left_dof_indices[i]);
            db[i][1] = momentum1(local_dof_indices[i]) - momentum1(left_dof_indices[i]);
            db[i][2] = energy(local_dof_indices[i]) - energy(left_dof_indices[i]);

            df[i][0] = 0.0;
            if(rbc_reflect)
               df[i][1] = 2.0*momentum1(local_dof_indices[i]);
            else
               df[i][1] = 0.0;
            df[i][2] = 0.0;
         }
      }
      else
      {
         lcell[c]->get_dof_indices (left_dof_indices);
         rcell[c]->get_dof_indices (right_dof_indices);
         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            db[i][0] = density(local_dof_indices[i]) - density(left_dof_indices[i]);
            db[i][1] = momentum1(local_dof_indices[i]) - momentum1(left_dof_indices[i]);
            db[i][2] = energy(local_dof_indices[i]) - energy(left_dof_indices[i]);

            df[i][0] = density(right_dof_indices[i]) - density(local_dof_indices[i]);
            df[i][1] = momentum1(right_dof_indices[i]) - momentum1(local_dof_indices[i]);
            df[i][2] = energy(right_dof_indices[i]) - energy(local_dof_indices[i]);
         }
      }
      
      for(unsigned int i=0; i<dofs_per_cell; ++i)
      {
         DC[i][0] = density(local_dof_indices[i]);
         DC[i][1] = momentum1(local_dof_indices[i]);
         DC[i][2] = energy(local_dof_indices[i]);
      }

      double R[n_var][n_var], Ri[n_var][n_var];
      if(lim_char)
      {
         EigMat(density_average[c], 
                momentum1_average[c], 
                energy_average[c], R, Ri);
         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            Multi(Ri, db[i]);
            Multi(Ri, df[i]);
            Multi(Ri, DC[i]);
         }
      }

      // Legendre in deal.ii is normalized. Moment limiter is BDF paper is
      // given for non-normalized basis functions. We apply correct
      // transformation here to account for this difference
      // cell average value is unchanged
      bool to_limit = true;
      for(unsigned int i=dofs_per_cell-1; i>=1; --i)
      {
         if(to_limit)
         {
            double l = (2*i - 1)*std::sqrt(2*i+1);
            double s = std::sqrt(2*i-1);
            std::vector<double> dcn(n_var);
            dcn[0] = minmod(l*DC[i][0], s*db[i-1][0], s*df[i-1][0])/l;
            dcn[1] = minmod(l*DC[i][1], s*db[i-1][1], s*df[i-1][1])/l;
            dcn[2] = minmod(l*DC[i][2], s*db[i-1][2], s*df[i-1][2])/l;
            double diff = std::fabs(dcn[0]-DC[i][0]) 
                        + std::fabs(dcn[1]-DC[i][1])
                        + std::fabs(dcn[2]-DC[i][2]);
            if(lim_char) Multi(R, dcn);
            density_n(local_dof_indices[i])  = dcn[0];
            momentum1_n(local_dof_indices[i]) = dcn[1];
            energy_n(local_dof_indices[i])   = dcn[2];
            if(diff < 1.0e-10) to_limit = false; // Remaining dofs will not be limited
         }
         else
         {
            density_n(local_dof_indices[i])  = density(local_dof_indices[i]); 
            momentum1_n(local_dof_indices[i]) = momentum1(local_dof_indices[i]);
            energy_n(local_dof_indices[i])   = energy(local_dof_indices[i]);
         }
      }

   }

   // Now copy to main arrays
   density = density_n;
   momentum1= momentum1_n;
   energy  = energy_n;
}

//------------------------------------------------------------------------------
// Moment limiter of Burbeau, Sagaut, Bruneau
// Author: Sudarshan Kumar K
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::apply_limiter_BSB ()
{
   if(fe.degree == 0) return;
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;   
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   std::vector<unsigned int> left_dof_indices (dofs_per_cell);
   std::vector<unsigned int> right_dof_indices (dofs_per_cell);
   
   std::vector< std::vector<double> > db(dofs_per_cell, std::vector<double>(n_var));
   std::vector< std::vector<double> > df(dofs_per_cell, std::vector<double>(n_var));
   std::vector< std::vector<double> > DC(dofs_per_cell, std::vector<double>(n_var));
   std::vector< std::vector<double> > DC_l(dofs_per_cell, std::vector<double>(n_var));
   std::vector< std::vector<double> > DC_r(dofs_per_cell, std::vector<double>(n_var));

   // Temporary storage
   Vector<double> density_n(density);
   Vector<double> momentum1_n(momentum1);
   Vector<double> energy_n(energy);

   typename DoFHandler<dim>::active_cell_iterator 
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   for (unsigned int c=0; c<n_cells; ++c, ++cell)
   {
      cell->get_dof_indices (local_dof_indices);
      
      if(c==0 && !periodic)
      {
         rcell[c]->get_dof_indices (right_dof_indices);
         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            db[i][0] = 0.0;
            DC_l[i][0]=density(local_dof_indices[i]);
            if(lbc_reflect)
            {
               db[i][1] = 2.0*momentum1(local_dof_indices[i]);
               DC_l[i][1]=-momentum1(local_dof_indices[i]);
            }
            else
            {
               db[i][1] = 0.0;
               DC_l[i][1]=momentum1(local_dof_indices[i]);
            }
            db[i][2] = 0.0;
            DC_l[i][2]=energy(local_dof_indices[i]);
            
            df[i][0] = density(right_dof_indices[i]) - density(local_dof_indices[i]);
            df[i][1] = momentum1(right_dof_indices[i]) - momentum1(local_dof_indices[i]);
            df[i][2] = energy(right_dof_indices[i]) - energy(local_dof_indices[i]);
            
            DC_r[i][0] = density(right_dof_indices[i]);
            DC_r[i][1] = momentum1(right_dof_indices[i]);
            DC_r[i][2] = energy(right_dof_indices[i]);
         }
      }
      else if(c == n_cells-1 && !periodic)
      {
         lcell[c]->get_dof_indices (left_dof_indices);
         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            db[i][0] = density(local_dof_indices[i]) - density(left_dof_indices[i]);
            db[i][1] = momentum1(local_dof_indices[i]) - momentum1(left_dof_indices[i]);
            db[i][2] = energy(local_dof_indices[i]) - energy(left_dof_indices[i]);
            
            DC_l[i][0] = density(left_dof_indices[i]);
            DC_l[i][1] = momentum1(left_dof_indices[i]);
            DC_l[i][2] = energy(left_dof_indices[i]);

            df[i][0] = 0.0;
            DC_r[i][0]=density(local_dof_indices[i]);
            if(rbc_reflect)
            {
               df[i][1] = 2.0*momentum1(local_dof_indices[i]);
               DC_r[i][1]=-momentum1(local_dof_indices[i]);
            }
            else
            {
               df[i][1] = 0.0;
               DC_r[i][1]=momentum1(local_dof_indices[i]);
            }
            df[i][2] = 0.0;
            DC_r[i][2]=energy(local_dof_indices[i]);
         }
      }
      else
      {
         lcell[c]->get_dof_indices (left_dof_indices);
         rcell[c]->get_dof_indices (right_dof_indices);
         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            db[i][0] = density(local_dof_indices[i]) - density(left_dof_indices[i]);
            db[i][1] = momentum1(local_dof_indices[i]) - momentum1(left_dof_indices[i]);
            db[i][2] = energy(local_dof_indices[i]) - energy(left_dof_indices[i]);
            
            df[i][0] = density(right_dof_indices[i]) - density(local_dof_indices[i]);
            df[i][1] = momentum1(right_dof_indices[i]) - momentum1(local_dof_indices[i]);
            df[i][2] = energy(right_dof_indices[i]) - energy(local_dof_indices[i]);
            
            DC_l[i][0] = density(left_dof_indices[i]);
            DC_l[i][1] = momentum1(left_dof_indices[i]);
            DC_l[i][2] = energy(left_dof_indices[i]);
            
            DC_r[i][0] = density(right_dof_indices[i]);
            DC_r[i][1] = momentum1(right_dof_indices[i]);
            DC_r[i][2] = energy(right_dof_indices[i]);
         }
      }
      
      for(unsigned int i=0; i<dofs_per_cell; ++i)
      {
         DC[i][0] = density(local_dof_indices[i]);
         DC[i][1] = momentum1(local_dof_indices[i]);
         DC[i][2] = energy(local_dof_indices[i]);
      }

      double R[n_var][n_var], Ri[n_var][n_var];
      if(lim_char)
      {
         EigMat(density_average[c], 
                momentum1_average[c], 
                energy_average[c], R, Ri);
         for(unsigned int i=0; i<dofs_per_cell; ++i)
         {
            Multi(Ri, db[i]);
            Multi(Ri, df[i]);
            Multi(Ri, DC[i]);
            Multi(Ri, DC_l[i]);
            Multi(Ri, DC_r[i]);
         }
      }

      // Legendre in deal.ii is normalized. Moment limiter is BDF & BSB paper is
      // given for non-normalized basis functions. We apply correct
      // transformation here to account for this difference
      // cell average value is unchanged
      bool to_limit = true;
      for(unsigned int i=dofs_per_cell-1; i>=1; --i)
      {
         if(to_limit)
         {
            double l = (2*i - 1)*std::sqrt(2*i+1);
            double s = std::sqrt(2*i-1);
            double t = std::sqrt(2*i+1);
            std::vector<double> dcn(n_var),dcn_m(n_var);
            std::vector<double> ur(n_var),ul(n_var),u_max(n_var);
            dcn[0] = minmod(l*DC[i][0], s*db[i-1][0], s*df[i-1][0])/l;
            dcn[1] = minmod(l*DC[i][1], s*db[i-1][1], s*df[i-1][1])/l;
            dcn[2] = minmod(l*DC[i][2], s*db[i-1][2], s*df[i-1][2])/l;

           double diff = std::fabs(dcn[0]-DC[i][0]) 
                       + std::fabs(dcn[1]-DC[i][1])
                       + std::fabs(dcn[2]-DC[i][2]);
            if(diff>1.0e-10)
            {
               ur[0] = s*DC_r[i-1][0] - (2*i-1)*t*DC_r[i][0];
               ur[1] = s*DC_r[i-1][1] - (2*i-1)*t*DC_r[i][1];
               ur[2] = s*DC_r[i-1][2] - (2*i-1)*t*DC_r[i][2];
               
               ul[0] = s*DC_l[i-1][0] + (2*i-1)*t*DC_l[i][0];
               ul[1] = s*DC_l[i-1][1] + (2*i-1)*t*DC_l[i][1];
               ul[2] = s*DC_l[i-1][2] + (2*i-1)*t*DC_l[i][2];
               
               u_max[0] = minmod(l*DC[i][0], (ur[0]-s*DC[i-1][0]), (s*DC[i-1][0]-ul[0]) ) /l;
               u_max[1] = minmod(l*DC[i][1], (ur[1]-s*DC[i-1][1]), (s*DC[i-1][1]-ul[1]) ) /l;
               u_max[2] = minmod(l*DC[i][2], (ur[2]-s*DC[i-1][2]), (s*DC[i-1][2]-ul[2]) ) /l;
               
               dcn_m[0] = maxmod(dcn[0],u_max[0]);
               dcn_m[1] = maxmod(dcn[1],u_max[1]);
               dcn_m[2] = maxmod(dcn[2],u_max[2]);
               if(lim_char) Multi(R, dcn_m);
               density_n(local_dof_indices[i])  = dcn_m[0];
               momentum1_n(local_dof_indices[i]) = dcn_m[1];
               energy_n(local_dof_indices[i])   = dcn_m[2];
            }
            else
            {
               density_n(local_dof_indices[i])  = density(local_dof_indices[i]); 
               momentum1_n(local_dof_indices[i]) = momentum1(local_dof_indices[i]);
               energy_n(local_dof_indices[i])   = energy(local_dof_indices[i]);
               to_limit = false; // Remaining dofs will not be limited
            }
         }
         else
         {
            density_n(local_dof_indices[i])  = density(local_dof_indices[i]); 
            momentum1_n(local_dof_indices[i]) = momentum1(local_dof_indices[i]);
            energy_n(local_dof_indices[i])   = energy(local_dof_indices[i]);
         }
      }

   }

   // Now copy to main arrays
   density = density_n;
   momentum1= momentum1_n;
   energy  = energy_n;
}

*/

//------------------------------------------------------------------------------
// Apply positivity limiter
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::apply_positivity_limiter ()
{
   if(fe.degree == 0) return;
   
   /*
   // Need 2N - 3 >= degree for the quadrature to be exact.
   unsigned int N = (fe.degree + 3)/2;
   if((fe.degree+3)%2 != 0) N += 1;
   QGaussLobatto<dim>  quadrature_formula(N);
   const unsigned int n_q_points = quadrature_formula.size();
   FEValues<dim> fe_values (fe, quadrature_formula, update_values);
   std::vector<double> density_values(n_q_points), momentum1_values(n_q_points),
                       energy_values(n_q_points);
   
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;
   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   
   double eps = 1.0e-13;
   for (unsigned int c=0; c<n_cells; ++c)
   {
      double velocity = momentum1_average[c] / density_average[c];
      double pressure = (gas_gamma-1.0) * ( energy_average[c] -
                                           0.5 * momentum1_average[c] * velocity );
      eps = std::min(eps, density_average[c]);
      eps = std::min(eps, pressure);
   }
   if(eps < 0.0)
   {
      std::cout << "Fatal: Negative states\n";
      exit(0);
   }

   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

   for (unsigned int c=0; cell!=endc; ++cell, ++c)
   {
      fe_values.reinit(cell);
      cell->get_dof_indices (local_dof_indices);
      
      // First limit density
      fe_values.get_function_values(density, density_values);
      
      // find minimum density at GLL points
      double rho_min = 1.0e20;
      for(unsigned int q=0; q<n_q_points; ++q)
         rho_min = std::min(rho_min, density_values[q]);
      
      double rat = std::fabs(density_average[c] - eps) /
                   (std::fabs(density_average[c] - rho_min) + 1.0e-13);
      double theta1 = std::min(rat, 1.0);
      
      for(unsigned int i=1; i<dofs_per_cell; ++i)
         density(local_dof_indices[i]) *= theta1;
      
      // now limit pressure
      fe_values.get_function_values(density, density_values);
      fe_values.get_function_values(momentum1, momentum1_values);
      fe_values.get_function_values(energy, energy_values);
      
      double theta2 = 1.0;
      for(unsigned int q=0; q<n_q_points; ++q)
      {
         double pressure = (gas_gamma-1.0)*(energy_values[q] -
                              0.5*std::pow(momentum1_values[q],2)/density_values[q]);
         if(pressure < eps)
         {
            double drho = density_values[q] - density_average[c];
            double dm = momentum1_values[q] - momentum1_average[c];
            double dE = energy_values[q] - energy_average[c];
            double a1 = 2.0*drho*dE - dm*dm;
            double b1 = 2.0*drho*(energy_average[c] - eps/(gas_gamma-1.0))
                        + 2.0*density_average[c]*dE
                        - 2.0*momentum1_average[c]*dm;
            double c1 = 2.0*density_average[c]*energy_average[c]
                        - momentum1_average[c]*momentum1_average[c]
                        - 2.0*eps*density_average[c]/(gas_gamma-1.0);
            // Divide by a1 to avoid round-off error
            b1 /= a1; c1 /= a1;
            double D = std::sqrt( std::fabs(b1*b1 - 4.0*c1) );
            double t1 = 0.5*(-b1 - D);
            double t2 = 0.5*(-b1 + D);
            double t;
            if(t1 > -1.0e-12 && t1 < 1.0 + 1.0e-12)
               t = t1;
            else if(t2 > -1.0e-12 && t2 < 1.0 + 1.0e-12)
                  t = t2;
            else
            {
               std::cout << "Problem in positivity limiter\n";
               std::cout << "\t a1, b1, c1 = " << a1 << " " << b1 << " " << c1 << "\n";
               std::cout << "\t t1, t2 = " << t1 << " " << t2 << "\n";
               std::cout << "\t eps, rho_min = " << eps << " " << rho_min << "\n";
               std::cout << "\t theta1 = " << theta1 << "\n";
               std::cout << "\t pressure = " << pressure << "\n";
               exit(0);
            }
            // t should strictly lie in [0,1]
            t = std::min(1.0, t);
            t = std::max(0.0, t);
            // Need t < 1.0. If t==1 upto machine precision
            // then we are suffering from round off error.
            // In this case we take the cell average value, t=0.
            if(std::fabs(1.0-t) < 1.0e-14) t = 0.0;
            theta2 = std::min(theta2, t);
         }
      }
      
      for(unsigned int i=1; i<dofs_per_cell; ++i)
      {
         density(local_dof_indices[i])  *= theta2;
         momentum1(local_dof_indices[i]) *= theta2;
         energy(local_dof_indices[i])   *= theta2;
      }
   }

   */
}

//------------------------------------------------------------------------------
// Compute time step from cfl condition
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::compute_dt ()
{
   dt = 1.0e20;
   for(unsigned int i=0; i<n_cells; ++i)
   {
      double velocity1 = momentum1_average[i] / density_average[i];
      double pressure11 = 2.0 * ( energy11_average[i] -
                        0.5 * density_average[i] * velocity1 * velocity1 );
      double sonic = std::sqrt ( 3.0 * pressure11 / density_average[i] );
      double speed = std::fabs(velocity1) + sonic;
      dt = std::min (dt, dx/speed);
   }
   
   dt *= cfl;
}

//------------------------------------------------------------------------------
// Update solution by one stage of RK
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::update (const unsigned int rk_stage)
{
   // Update conserved variables
   for(unsigned int i=0; i<dof_handler.n_dofs(); ++i)
   {
      density(i)  = a_rk[rk_stage] * density_old(i) +
                    b_rk[rk_stage] * (density(i) + dt * rhs_density(i));
      momentum1(i) = a_rk[rk_stage] * momentum1_old(i) +
                    b_rk[rk_stage] * (momentum1(i) + dt * rhs_momentum1(i));
      momentum2(i) = a_rk[rk_stage] * momentum2_old(i) +
                    b_rk[rk_stage] * (momentum2(i) + dt * rhs_momentum2(i));
      energy11(i) = a_rk[rk_stage] * energy11_old(i) +
                    b_rk[rk_stage] * (energy11(i) + dt * rhs_energy11(i));
      energy12(i) = a_rk[rk_stage] * energy12_old(i) +
                    b_rk[rk_stage] * (energy12(i) + dt * rhs_energy12(i));
      energy22(i) = a_rk[rk_stage] * energy22_old(i) +
                    b_rk[rk_stage] * (energy22(i) + dt * rhs_energy22(i));
   }

}

//------------------------------------------------------------------------------
// Save solution to file
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::output_results () const
{
   // counter to set file name
   static unsigned int c = 0;
   
   DataOut<dim> data_out;
   
   data_out.attach_dof_handler (dof_handler);
   data_out.add_data_vector (density, "density");
   data_out.add_data_vector (momentum1, "momentum1");
   data_out.add_data_vector (momentum2, "momentum2");
   data_out.add_data_vector (energy11, "energy11");
   data_out.add_data_vector (energy12, "energy12");
   data_out.add_data_vector (energy22, "energy22");
   
   if(fe.degree <= 1)
      data_out.build_patches (1);
   else
      data_out.build_patches (fe.degree+1);
   
   std::string filename = "sol_" + Utilities::int_to_string(c) + ".gpl";
   std::ofstream output (filename.c_str());
   data_out.write_gnuplot (output);

   // save cell average solution
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

   std::ofstream fo;
   filename = "avg.gpl";
   fo.open (filename.c_str());

   for (unsigned int c=0; cell!=endc; ++c, ++cell)
   {
      Point<dim> x = cell->center();
      double velocity1 = momentum1_average[c] / density_average[c];
      double velocity2 = momentum2_average[c] / density_average[c];
      double pressure11 = 2.0 * ( energy11_average[c] -
                        0.5 * density_average[c] * velocity1 * velocity1 );
      double pressure12 = 2.0 * ( energy12_average[c] -
                        0.5 * density_average[c] * velocity1 * velocity2 );
      double pressure22 = 2.0 * ( energy22_average[c] -
                        0.5 * density_average[c] * velocity2 * velocity2 );
      fo << x(0) << " "
         << density_average[c] << "  " 
         << velocity1 << "  " 
         << velocity1 << "  " 
         << pressure11 << "  " 
         << pressure12 << "  " 
         << pressure22
         << std::endl;
   }

   fo.close ();

   // increment filename counter
   ++c;
}
//------------------------------------------------------------------------------
// Compute error in solution
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::compute_errors(double& L2_error,
                                       double& H1_error,
                                       double& Linf_error) const
{
   Vector<double> difference_per_cell (triangulation.n_active_cells());
   VectorTools::integrate_difference (dof_handler,
                                      density,
                                      ExactSolution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree+2),
                                      VectorTools::L2_norm);
   L2_error = difference_per_cell.l2_norm();
   
   VectorTools::integrate_difference (dof_handler,
                                      density,
                                      ExactSolution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree+2),
                                      VectorTools::H1_seminorm);
   H1_error = difference_per_cell.l2_norm();
   
   VectorTools::integrate_difference (dof_handler,
                                      density,
                                      ExactSolution<dim>(),
                                      difference_per_cell,
                                      QIterated<dim>(QTrapez<dim>(),5),
                                      VectorTools::Linfty_norm);
   Linf_error = difference_per_cell.linfty_norm();
}
//------------------------------------------------------------------------------
// Start solving the problem
//------------------------------------------------------------------------------
template <int dim>
void Gauss10<dim>::run (double& h,
                             int& ndof,
                             double& L2_error,
                             double& H1_error,
                             double& Linf_error)
{
    std::cout << "\n Solving 1-D Euler problem ...\n";

    make_grid_and_dofs();
    assemble_mass_matrix ();
    initialize ();
    compute_averages ();
    apply_limiter ();
    if(lim_pos) apply_positivity_limiter ();
    output_results ();

    double time = 0.0;
    unsigned int iter = 0;

    std::cout << "Starting the time stepping ... \n";

    while (time < final_time && iter < max_iter)
    {
       density_old  = density;
       momentum1_old = momentum1;
       momentum2_old = momentum2;
       energy11_old   = energy11;
       energy12_old   = energy12;
       energy22_old   = energy22;
       
       compute_dt ();
       if(time+dt > final_time) dt = final_time - time;

       for(unsigned int rk=0; rk<n_rk_stages; ++rk)
       {
          assemble_rhs ();
          update (rk);
          compute_averages ();
          apply_limiter ();
          if(lim_pos) apply_positivity_limiter ();
       }
       
       if(iter==0)
       {
          std::cout << "Initial residual = " << residual[0] << " "
                    << residual[1] << " "
                    << residual[2] << " "
                    << residual[3] << " "
                    << residual[4] << " "
                    << residual[5] << std::endl;
          for(unsigned int i=0; i<n_var; ++i)
             residual0[i] = residual[i];
       }
       
      time += dt;
      ++iter;
      if(iter % save_freq == 0) output_results ();
       
      std::cout << "Iter = " << iter << " time = " << time 
                << " Res =" << residual[0] << " " << residual[1] << " "
                << residual[2] << " " << residual[3] << " "
                << residual[4] << " " << residual[5] << std::endl;
    }
    output_results ();
   
   if(test_case == "smooth") compute_errors (L2_error, H1_error, Linf_error);
   h = dx;
   ndof = dof_handler.n_dofs();
}

//------------------------------------------------------------------------------
// Declare input parameters
//------------------------------------------------------------------------------
void declare_parameters(ParameterHandler& prm)
{
   prm.declare_entry("degree","0", Patterns::Integer(0,6),
                     "Polynomial degree");
   prm.declare_entry("ncells","100", Patterns::Integer(10,100000),
                     "Number of elements");
   prm.declare_entry("save frequency","100000", Patterns::Integer(0,100000),
                     "How often to save solution");
   prm.declare_entry("test case","sod", 
                     Patterns::Selection("sod|lowd|blast|blastwc|lax|shuosher|sedov|smooth"),
                     "Test case");
   prm.declare_entry("limiter","TVB", 
                     Patterns::Selection("None|TVB|BDF|BSB"),
                     "limiter");
   prm.declare_entry("flux","lxf", 
                     Patterns::Selection("lxf|roe|hllc"),
                     "limiter");
   prm.declare_entry("characteristic limiter", "false",
                     Patterns::Bool(), "Characteristic limiter");
   prm.declare_entry("positivity limiter", "false",
                     Patterns::Bool(), "positivity limiter");
   prm.declare_entry("cfl", "1.0",
                     Patterns::Double(0,1.0), "cfl number");
   prm.declare_entry("M", "0.0",
                     Patterns::Double(0,1.0e20), "TVB constant");
   prm.declare_entry("beta", "1.0",
                     Patterns::Double(0.5,1.0), "Constant in minmod limiter");
   prm.declare_entry("refine","0", Patterns::Integer(0,10),
                     "Number of mesh refinements");
   prm.declare_entry("max iter","1000000000", Patterns::Integer(0,1000000000),
                     "maximum iterations");
}
//------------------------------------------------------------------------------
// Compute convergence rates
//------------------------------------------------------------------------------
void compute_rate(std::vector<double>& h, std::vector<int>& ndof,
                  std::vector<double>& L2_error, std::vector<double>& H1_error,
                  std::vector<double>& Linf_error)
{
   ConvergenceTable   convergence_table;
   unsigned int nrefine = h.size() - 1;
   for(unsigned int i=0; i<=nrefine; ++i)
   {
      convergence_table.add_value("cycle", i);
      convergence_table.add_value("h", h[i]);
      convergence_table.add_value("dofs", ndof[i]);
      convergence_table.add_value("L2", L2_error[i]);
      convergence_table.add_value("H1", H1_error[i]);
      convergence_table.add_value("Linf", Linf_error[i]);
   }

   convergence_table.set_precision("L2", 3);
   convergence_table.set_precision("H1", 3);
   convergence_table.set_precision("Linf", 3);

   convergence_table.set_scientific("h", true);
   convergence_table.set_scientific("L2", true);
   convergence_table.set_scientific("H1", true);
   convergence_table.set_scientific("Linf", true);

   convergence_table.set_tex_caption("h", "$h$");
   convergence_table.set_tex_caption("dofs", "\\# dofs");
   convergence_table.set_tex_caption("L2", "$L^2$-error");
   convergence_table.set_tex_caption("H1", "$H^1$-error");
   convergence_table.set_tex_caption("Linf", "$L_\\infty$-error");

   convergence_table.evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
   convergence_table.evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);
   convergence_table.evaluate_convergence_rates("Linf", ConvergenceTable::reduction_rate_log2);

   std::cout << std::endl;
   convergence_table.write_text (std::cout);

   std::ofstream error_table_file("error.tex");
   convergence_table.write_tex (error_table_file);

}
//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int main (int argc, char** argv)
{
   deallog.depth_console (0);
   {
      ParameterHandler prm;
      declare_parameters (prm);
      if(argc < 2)
      {
         std::cout << "Specify input parameter file\n";
         std::cout << "It should contain following parameters.\n\n";
         prm.print_parameters(std::cout, ParameterHandler::Text);
         return 0;
      }
      bool status = prm.read_input (argv[1], true);
      AssertThrow( status, ExcFileNotOpen(argv[1]) );
      prm.print_parameters(std::cout, ParameterHandler::Text);
      unsigned int degree = prm.get_integer("degree");
      unsigned int nrefine = prm.get_integer("refine");
      std::vector<double> h(nrefine+1), L2_error(nrefine+1), H1_error(nrefine+1),
                          Linf_error(nrefine+1);
      std::vector<int> ndof(nrefine+1);
      for(unsigned int i=0; i<=nrefine; ++i)
      {
         Gauss10<1> euler_problem(degree, prm);
         euler_problem.run (h[i], ndof[i], L2_error[i], H1_error[i], Linf_error[i]);
         const long int ncells = 2 * prm.get_integer("ncells");
         prm.set("ncells", ncells);
      }
      if(nrefine > 0) compute_rate(h, ndof, L2_error, H1_error, Linf_error);
   }

   return 0;
}
