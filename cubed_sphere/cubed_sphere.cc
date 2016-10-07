#include <deal.II/base/point.h>

#include "cubed_sphere.h"

using namespace dealii;

CubedSphere::CubedSphere(const int face, const double radius, const Point<3> center)
:
ChartManifold<2,3,2>(),
face(face),
radius(radius),
a(radius/std::sqrt(3.0)),
center(center)
{
}

Point<2>
CubedSphere::pull_back(const Point<3> &space_point) const
{
   double x, y; 
   switch(face)
   {
      case 1:
      {
         x =  a * space_point[1] / space_point[0];
         y =  a * space_point[2] / space_point[0];
         break;
      }
         
      case 2:
      {
         x = -a * space_point[0] / space_point[1];
         y =  a * space_point[2] / space_point[1];
         break;
      }
         
      case 3:
      {
         x =  a * space_point[1] / space_point[0];
         y = -a * space_point[2] / space_point[0];
         break;
      }
         
      case 4:
      {
         x = -a * space_point[0] / space_point[1];
         y = -a * space_point[2] / space_point[1];
         break;

      }
         
      case 5:
      {
         x =  a * space_point[1] / space_point[2];
         y = -a * space_point[0] / space_point[2];
         break;
      }
         
      case 6:
      {
         x = -a * space_point[1] / space_point[2];
         y = -a * space_point[0] / space_point[2];
         break;
      }
         
      default:
      {
         AssertThrow(false, ExcMessage("Unknown face number"));
         x = 0; y = 0;
      }
   }
   
   Point<2> chart_point;
   chart_point[0] = std::atan(x/a);
   chart_point[1] = std::atan(y/a);
   return chart_point;
}

Point<3>
CubedSphere::push_forward(const Point<2> &chart_point) const
{
   const double x = a * std::tan(chart_point[0]);
   const double y = a * std::tan(chart_point[1]);
   Point<3> space_point;
   const double r = std::sqrt(a*a + x*x + y*y);
   const double rr = radius / r;
   
   switch(face)
   {
      case 1:
      {
         space_point[0] = rr * a;
         space_point[1] = rr * x;
         space_point[2] = rr * y;
         break;
      }
         
      case 2:
      {
         space_point[0] = -rr * x;
         space_point[1] =  rr * a;
         space_point[2] =  rr * y;
         break;
      }
         
      case 3:
      {
         space_point[0] = -rr * a;
         space_point[1] = -rr * x;
         space_point[2] =  rr * y;
         break;
      }
         
      case 4:
      {
         space_point[0] =  rr * x;
         space_point[1] = -rr * a;
         space_point[2] =  rr * y;
         break;
      }
         
      case 5:
      {
         space_point[0] = -rr * y;
         space_point[1] =  rr * x;
         space_point[2] =  rr * a;
         break;
      }
         
      case 6:
      {
         space_point[0] =  rr * y;
         space_point[1] =  rr * x;
         space_point[2] = -rr * a;
         break;
      }
         
      default:
      {
         AssertThrow(false, ExcMessage("Unknown face number"));
      }
   }
   
   return space_point;
}

DerivativeForm<1,2,3>
CubedSphere::push_forward_gradient(const Point<2> &chart_point) const
{
   const double x = a * std::tan(chart_point[0]);
   const double y = a * std::tan(chart_point[1]);
   const double r = std::sqrt(a*a + x*x + y*y);
   const double rr = radius / r;
   const double r3 = std::pow(r, 3);
   const double rr_x = -(radius * x) / r3;
   const double rr_y = -(radius * y) / r3;
   
   DerivativeForm<1,2,3> DX;
   
   switch(face)
   {
      case 1:
      {
         DX[0][0] = rr_x * a;       DX[0][1] = rr_y * a;
         DX[1][0] = rr_x * x + rr;  DX[1][1] = rr_y * x;
         DX[2][0] = rr_x * y;       DX[2][1] = rr_y * y + rr;
         break;
      }
         
      case 2:
      {
         DX[0][0] = -rr_x * x - rr;  DX[0][1] = -rr_y * x;
         DX[1][0] =  rr_x * a;       DX[1][1] =  rr_y * a;
         DX[2][0] =  rr_x * y;       DX[2][1] =  rr_y * y + rr;
         break;
      }
         
      case 3:
      {
         DX[0][0] = -rr_x * a;       DX[0][1] = -rr_y * a;
         DX[1][0] = -rr_x * x - rr;  DX[1][1] = -rr_y * x;
         DX[2][0] =  rr_x * y;       DX[2][1] =  rr_y * y + rr;
         break;
      }
         
      case 4:
      {
         DX[0][0] =  rr_x * x + rr;  DX[0][1] =  rr_y * x;
         DX[1][0] = -rr_x * a;       DX[1][1] = -rr_y * a;
         DX[2][0] =  rr_x * y;       DX[2][1] =  rr_y * y + rr;
         break;
      }
         
      case 5:
      {
         DX[0][0] = -rr_x * y;      DX[0][1] = -rr_y * y - rr;
         DX[1][0] =  rr_x * x + rr; DX[1][1] =  rr_y * x;
         DX[2][0] =  rr_x * a;      DX[2][1] =  rr_y * a;
         break;
      }
         
      case 6:
      {
         DX[0][0] =  rr_x * y;      DX[0][1] =  rr_y * y + rr;
         DX[1][0] =  rr_x * x + rr; DX[1][1] =  rr_y * x;
         DX[2][0] = -rr_x * a;      DX[2][1] = -rr_y * a;
         break;
      }
   }
   const double A = a / std::pow( cos(chart_point[0]), 2);
   const double B = a / std::pow( cos(chart_point[1]), 2);
   DX[0][0] *= A; DX[0][1] *= B;
   DX[1][0] *= A; DX[1][1] *= B;
   DX[2][0] *= A; DX[2][1] *= B;
   return DX;
}
