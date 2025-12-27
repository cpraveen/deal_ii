#include <deal.II/base/function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

const int dim = 2;

//-----------------------------------------------------------------------------
// Taken from Arpit thesis
//-----------------------------------------------------------------------------
template <int dim>
class MyMap0 : public Function<dim>
{
   public:
      MyMap0() : Function<dim>(dim) {}

      void vector_value(const Point<dim>& p, 
                        Vector<double>& value) const override
      {
         const double x = p[0];
         const double y = p[1];
         value[0] = x * Lx - Ax * Lx * sin(2 * M_PI * y);
         value[1] = y * Ly + Ay * Lx * sin(2 * M_PI * x);
      }

  private:
     const double Lx = 1.0;
     const double Ly = 1.0;
     const double Ax = 0.1;
     const double Ay = 0.1;
};

//-----------------------------------------------------------------------------
// Taken from Arpit thesis
// Modified to map [0,1]x[0,1] --> [0,1]x[0,1]
//-----------------------------------------------------------------------------
template <int dim>
class MyMap1 : public Function<dim>
{
   public:
      MyMap1() : Function<dim>(dim) {}

      void vector_value(const Point<dim>& p, 
                        Vector<double>& value) const override
      {
         const double xi = 3.0 * p[0];
         const double eta = 3.0 * p[1];
         const double y = eta + 3.0/8.0 * (cos(1.5 * M_PI * (2 * xi - 3)/3) *
                                           cos(0.5 * M_PI * (2 * eta - 3)/3));
         const double x = xi + 3.0/8.0 * (cos(0.5 * M_PI * (2 * xi - 3)/3) *
                                          cos(2 * M_PI * (2 * y - 3)/3));
         value[0] = x/3.0;
         value[1] = y/3.0;
      }
};

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
   const int degree = 2;

   int test = 0;
   if(argc == 2) test = Utilities::string_to_int(argv[1]);
   std::cout << "Test case = " << test << std::endl;

   const FESystem<dim> fe(FE_Q<dim>(degree), dim);

   Triangulation<dim> triangulation;
   GridGenerator::hyper_cube(triangulation);
   triangulation.refine_global(4);

   DoFHandler<dim> dof_handler(triangulation);
   dof_handler.distribute_dofs(fe);

   Vector<double> euler(dof_handler.n_dofs());
   if(test == 0)
      VectorTools::interpolate(dof_handler, MyMap0<dim>(), euler);
   else
      VectorTools::interpolate(dof_handler, MyMap1<dim>(), euler);
   MappingFEField<dim,dim> mapping(dof_handler, euler);

   // Original grid
   {
      GridOut grid_out;
      std::ofstream gfile("grid0.gnu");
      grid_out.write_gnuplot(triangulation, gfile);
   }

   // Mapped grid
   {
      GridOutFlags::Gnuplot flags;
      flags.curved_inner_cells = true;
      GridOut grid_out;
      grid_out.set_flags(flags);
      std::ofstream gfile("grid1.gnu");
      grid_out.write_gnuplot(triangulation, gfile, &mapping);
   }
}
