//------------------------------------------------------------------------------
// We assume that some manifold(s) have been attached to the underlying 
// triangulation object. Then this computes the displacement vector field
// euler_vector = support points with MappingQ(mapping_degree)
//                - support points with MappingQ1
//------------------------------------------------------------------------------
template <int dim, class VecType>
void 
euler_vector_from_manifold(const DoFHandler<dim>& dof_handler,
                           const int              mapping_degree,
                           VecType&               euler_vector)
{
   VecType euler_vector_0(euler_vector);
   const FunctionParser<dim> CoordinatesFunction("x; y");

   VectorTools::interpolate(MappingQ1<dim>(),
                            dof_handler,
                            CoordinatesFunction,
                            euler_vector_0);

   VectorTools::interpolate(MappingQ<dim>(mapping_degree),
                            dof_handler,
                            CoordinatesFunction,
                            euler_vector);

   euler_vector -= euler_vector_0;
}
