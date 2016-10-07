#ifndef __CUBED_SPHERE__
#define __CUBED_SPHERE__

#include <deal.II/grid/manifold.h>

using namespace dealii;

class CubedSphere : public ChartManifold<2,3,2>
{
public:
   CubedSphere (const int face,
                const double radius=1.0,
                const Point<3> center = Point<3>());
   
   virtual Point<2>
   pull_back(const Point<3> &space_point) const;
   
   virtual Point<3>
   push_forward(const Point<2> &chart_point) const;
   
   virtual
   DerivativeForm<1,2,3>
   push_forward_gradient(const Point<2> &chart_point) const;
   
   const int      face;
   const double   radius;
   const double   a;
   const Point<3> center;
};

#endif
