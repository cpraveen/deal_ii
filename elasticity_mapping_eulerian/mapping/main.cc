#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

#include "elasticity.h"

using namespace dealii;

const int dim = 2;

void test1()
{
   Triangulation<dim> triangulation;
   GridGenerator::subdivided_hyper_rectangle (triangulation,
                                              {50,25},
                                              Point<dim>(0.0,0.0),
                                              Point<dim>(1.0,0.5),
                                              true);

   const int mapping_degree = 2;
   const double lambda = 1.0;
   const double mu = 1.0;
   ElasticityModel<dim> model(triangulation, mapping_degree, lambda, mu);
   Vector<double> euler_vector(model.n_dofs());

   const bool restart = true;
   const int verbosity = 2;
   unsigned int N = 1;
   const double dt = 1.0 / N;
   double t = 0.0;
   for (unsigned int counter = 0; counter < N; ++counter)
   {
      // Apply displacement on top boundary
      const std::string constants = "pi=3.141592653589793";
      FunctionParser<dim> top_displacement("0; 0.1*cos(2*pi*t)*sin(2*pi*x)",
                                          constants);
      top_displacement.set_time(t);
      std::map<types::global_dof_index,double> boundary_values;
      VectorTools::interpolate_boundary_values(model.get_dof_handler(),
                                               types::boundary_id(3),
                                               top_displacement,
                                               boundary_values);
      for(auto [i,v] : boundary_values)
         euler_vector(i) = v;

      model.solve(euler_vector, restart, verbosity);
      MappingQEulerian<dim> mapping(mapping_degree,
                                    model.get_dof_handler(),
                                    euler_vector);

      DataOut<dim> data_out;
      data_out.attach_dof_handler(model.get_dof_handler());
      data_out.add_data_vector(euler_vector, "Euler");
      data_out.build_patches(mapping, mapping_degree);
      std::string filename = "euler-" + Utilities::int_to_string(counter,4) + ".vtu";
      std::ofstream output(filename);
      data_out.write_vtu(output);

      t += dt;
   }

}

void test2(const bool move_normal)
{
   Triangulation<dim> triangulation;
   GridGenerator::subdivided_hyper_rectangle (triangulation,
                                              {50,25},
                                              Point<dim>(0.0,0.0),
                                              Point<dim>(1.0,0.5),
                                              true);

   const int mapping_degree = 2;
   const double lambda = 1.0;
   const double mu = 1.0;
   ElasticityModel<dim> model(triangulation, mapping_degree, lambda, mu);
   Vector<double> euler_vector(model.n_dofs());

   {
      MappingQEulerian<dim> mapping(mapping_degree,
                                    model.get_dof_handler(),
                                    euler_vector);
      DataOut<dim> data_out;
      data_out.attach_dof_handler(model.get_dof_handler());
      data_out.add_data_vector(euler_vector, "Euler");
      data_out.build_patches(mapping, mapping_degree, DataOut<dim>::curved_inner_cells);
      unsigned int counter = 0;
      std::string filename = "euler-" + Utilities::int_to_string(counter, 4) + ".vtu";
      std::ofstream output(filename);
      data_out.write_vtu(output);
   }

   const bool restart = true;
   const int verbosity = 2;
   unsigned int N = 100;
   const double dt = 1.0 / N;
   double t = 0.0;
   for (unsigned int counter = 1; counter <= N; ++counter)
   {
      // Apply displacement on top boundary
      const std::string constants = "pi=3.141592653589793";
      FunctionParser<dim> top_displacement("0; cos(2*pi*t)*sin(2*pi*x)",
                                          constants);
      top_displacement.set_time(t);

      Vector<double> tmp_euler_vector(model.n_dofs());
      Vector<double> counter_vector(model.n_dofs());

      MappingQEulerian<dim> mapping_current(mapping_degree,
                                            model.get_dof_handler(),
                                            euler_vector);

      const auto& dof_handler = model.get_dof_handler();
      const auto& fe = dof_handler.get_fe();
      std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_face());
      for(auto& cell : dof_handler.active_cell_iterators())
         for(auto face_no : GeometryInfo<dim>::face_indices())
            if(cell->face(face_no)->at_boundary() && cell->face(face_no)->boundary_id() == 3)
            {
               const auto& points = fe.get_unit_face_support_points(face_no);
               cell->face(face_no)->get_dof_indices(dof_indices);
               const Quadrature<dim-1> quadrature(points);
               std::vector<Vector<double>> velocity(quadrature.size(), Vector<double>(dim));
               FEFaceValues<dim> fe_face_values(mapping_current, fe, quadrature, 
                                                update_values |
                                                update_quadrature_points |
                                                update_normal_vectors);
               fe_face_values.reinit(cell, face_no);

               for(auto q : fe_face_values.quadrature_point_indices())
               {
                  top_displacement.vector_value(fe_face_values.quadrature_point(q), velocity[q]);
               }

               for(unsigned int i=0; i<fe.n_dofs_per_face(); ++i)
               {
                  Tensor<1,dim> vel;
                  if(move_normal)
                  {
                     const auto normal = fe_face_values.normal_vector(i);
                     const auto vn = velocity[i][0]*normal[0] + velocity[i][1]*normal[1];
                     vel = vn * normal;
                  }
                  else
                  {
                     vel[0] = velocity[i][0];
                     vel[1] = velocity[i][1];
                  }
                  auto comp_i = fe.face_system_to_component_index(i,face_no).first;
                  tmp_euler_vector(dof_indices[i]) += dt * vel[comp_i];
                  counter_vector(dof_indices[i]) += 1.0;
               }
            }

      for(unsigned int i=0; i<dof_handler.n_dofs(); ++i)
         if(counter_vector(i) > 0.0)
            tmp_euler_vector(i) /= counter_vector(i);

      euler_vector += tmp_euler_vector;
      model.solve(euler_vector, restart, verbosity);

      {
         MappingQEulerian<dim> mapping(mapping_degree,
                                       model.get_dof_handler(),
                                       euler_vector);

         DataOut<dim> data_out;
         data_out.attach_dof_handler(model.get_dof_handler());
         data_out.add_data_vector(euler_vector, "Euler");
         data_out.build_patches(mapping, mapping_degree, DataOut<dim>::curved_inner_cells);
         std::string filename = "euler-" + Utilities::int_to_string(counter,4) + ".vtu";
         std::ofstream output(filename);
         data_out.write_vtu(output);
      }

      t += dt;
   }

}

int main()
{
   //test1();
   test2(true);
   return 0;
}
