#ifndef __NACA_H__
#define __NACA_H__

#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold.h>

#include <Sacado.hpp>

// Dimension of problem
#define DIM  2

using namespace dealii;

template <typename T>
T naca0012 (const T &x)
{
   T x2 = x * x;
   T x3 = x * x2;
   T x4 = x * x3;
   T y = 0.594689181*(0.298222773*sqrt(x) - 0.127125232*x
                           - 0.357907906*x2 + 0.291984971*x3 - 0.105174606*x4);
   return y;
}


// NACA airfoil
class NACA0012 : public Manifold<2,2>
{
private:
   // used in newton method for projection to manifold
   void func(const double &m, const Point<2> &P, const double &x, double &f, double &d) const
   {
      // set and mark independent variable
      Sacado::Fad::DFad<double> xd = x;
      xd.diff(0,1);
      
      Sacado::Fad::DFad<double> y = naca0012 (xd);
      if(P[1] < 0) y = -y;
      Sacado::Fad::DFad<double> F = m * (y - P[1]) + (xd - P[0]);
      f = F.val();
      d = F.fastAccessDx(0);
   }
   
public:
   virtual std::unique_ptr<Manifold<2,2>> clone() const override
   {
      return std::make_unique<NACA0012>();
   }

   virtual Point<2> project_to_manifold (const ArrayView<const Point<2>> &surrounding_points,
                                         const Point<2>              &candidate) const override
   {
      const double GEOMTOL = 1.0e-13;
      Assert((surrounding_points[0][1] > -GEOMTOL && surrounding_points[1][1] > -GEOMTOL) ||
             (surrounding_points[0][1] <  GEOMTOL && surrounding_points[1][1] <  GEOMTOL),
             ExcMessage("End points not on same side of airfoil"));
      
      const double dx = surrounding_points[1][0] - surrounding_points[0][0];
      const double dy = surrounding_points[1][1] - surrounding_points[0][1];
      Assert (std::fabs(dx) > GEOMTOL, ExcMessage("dx is too small"));
      const double m  = dy/dx;
      
      // initial guess for newton
      double x = candidate[0];
      x = std::min(x, 1.0);
      x = std::max(x, 0.0);
      double f, d;
      func(m, candidate, x, f, d);
      unsigned int i = 0, ITMAX = 10;
      while (std::fabs(f) > GEOMTOL)
      {
         double s = -f/d;
         while(x+s < 0 || x+s > 1) s *= 0.5;
         x = x + s;
         func(m, candidate, x, f, d);
         ++i;
         AssertThrow(i < ITMAX, ExcMessage("Newton did not converge"));
      }
      
      double y = naca0012(x);
      if(candidate[1] < 0) y = -y;
      Point<2> p (x, y);
      return p;
   }
};

struct NACA
{
   static void set_curved_boundaries (Triangulation<DIM> &triangulation)
   {
      
      // we move points on airfoil to actual airfoil curve since we are not
      // sure Gmsh has ensured this.
      Triangulation<DIM>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
      for(; cell != endc; ++cell)
         for (unsigned int f=0; f<GeometryInfo<DIM>::faces_per_cell; ++f)
         {
            if(cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 2)
            {
               cell->face(f)->set_manifold_id (cell->face(f)->boundary_id());
               for(unsigned int v=0; v<GeometryInfo<DIM>::vertices_per_face; ++v)
               {
                  Point<DIM> &space_point = cell->face(f)->vertex(v);
                  double x = space_point[0];
                  x = std::min(x, 1.0);
                  x = std::max(x, 0.0);
                  double y = naca0012 (x);
                  if (space_point[1] > 0)
                     space_point[1] =  y;
                  else
                     space_point[1] = -y;
               }
            }
         }
      
      // attach manifold description
      static NACA0012 airfoil;
      triangulation.set_manifold (2, airfoil);
   }
};

#endif
