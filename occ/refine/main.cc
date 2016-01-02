#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/opencascade/boundary_lib.h>
#include <deal.II/opencascade/utilities.h>

#include <string>
#include <fstream>

using namespace dealii;

int main()
{
   
   const std::string in_mesh_filename = "rae2822/rae2822.msh";
   const std::string cad_file_name = "rae2822/rae2822.igs";
   
   // read iges file
   TopoDS_Shape airfoil = OpenCASCADE::read_IGES(cad_file_name, 1.0);
   const double tolerance = OpenCASCADE::get_shape_tolerance(airfoil) * 5;
   std::cout << "Tolerance = " << tolerance << std::endl;
   std::vector<TopoDS_Compound>  compounds;
   std::vector<TopoDS_CompSolid> compsolids;
   std::vector<TopoDS_Solid>     solids;
   std::vector<TopoDS_Shell>     shells;
   std::vector<TopoDS_Wire>      wires;
   OpenCASCADE::extract_compound_shapes(airfoil,
                                        compounds,
                                        compsolids,
                                        solids,
                                        shells,
                                        wires);
   std::cout << "Number of wires  = " << wires.size() << std::endl;
   std::cout << "Number of shells = " << shells.size() << std::endl;
   
   // read gmsh file
   Triangulation<2,3>   tria;
   
   std::ifstream mesh_file;
   mesh_file.open(in_mesh_filename.c_str());
   GridIn<2,3> grid_in;
   grid_in.attach_triangulation(tria);
   grid_in.read_msh(mesh_file);
   
   std::cout << "Number of cells = " << tria.n_active_cells() << std::endl;
   {
      const std::string filename = "grid0.vtk";
      std::ofstream gridfile(filename.c_str());
      GridOut grid_out;
      grid_out.write_vtk(tria, gridfile);
   }
   
   // set manifold ids
   Triangulation<2,3>::active_cell_iterator cell = tria.begin_active(),
   endc = tria.end();
   for(; cell != endc; ++cell)
   {
      cell->set_manifold_id (100);
      for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
         if(cell->face(f)->at_boundary())
            cell->face(f)->set_manifold_id (cell->face(f)->boundary_id());
   }

   
   Assert(wires.size() > 0,
          ExcMessage("I could not find any wire in the CAD file you gave me. Bailing out."));
   
//   for(unsigned int n=0; n<wires.size(); ++n)
//   {
//      static OpenCASCADE::ArclengthProjectionLineManifold<2,3>
//         line_projector (wires[n], 1.e-3);
//      tria.set_manifold(n+1, line_projector);
//   }
   
   // This is 2-d, in xy plane. Do we really need this projector ?
   static OpenCASCADE::NormalProjectionBoundary<2,3>
   normal_projector(airfoil, tolerance);
   tria.set_manifold(100, normal_projector);
   
   // outer boundary
   static OpenCASCADE::ArclengthProjectionLineManifold<2,3>
      line_projector0 (wires[0], tolerance);
   tria.set_manifold(1, line_projector0);

   // airfoil
   static OpenCASCADE::ArclengthProjectionLineManifold<2,3>
   line_projector1 (wires[1], tolerance);
   tria.set_manifold(2, line_projector1);

   // refine the mesh
   tria.refine_global(1);
   
   std::cout << "Number of cells = " << tria.n_active_cells() << std::endl;
   {
      const std::string filename = "grid1.vtk";
      std::ofstream gridfile(filename.c_str());
      GridOut grid_out;
      grid_out.write_vtk(tria, gridfile);
   }

}
