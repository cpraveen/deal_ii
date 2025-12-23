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
template <int dim>
class MyMap : public Function<dim>
{
   public:
      MyMap() : Function<dim>(dim) {}

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
int main()
{
   const int degree = 2;

   const FESystem<dim> fe(FE_Q<dim>(degree), dim);

   Triangulation<dim> triangulation;
   GridGenerator::hyper_cube(triangulation);
   triangulation.refine_global(4);

   DoFHandler<dim> dof_handler(triangulation);
   dof_handler.distribute_dofs(fe);

   Vector<double> euler(dof_handler.n_dofs());
   VectorTools::interpolate(dof_handler, MyMap<dim>(), euler);
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
