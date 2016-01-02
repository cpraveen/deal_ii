Merge "rae2822.igs";

//Field[1] = MathEval;
//Field[1].F = "0.05";
//Background Field = 1;

Transfinite Line {1} = 50;
Transfinite Line {2} = 100;

Mesh.Algorithm = 8;
Mesh.RecombineAll = 1;

Physical Surface(100) = {1};
Physical Line(1) = {1};  // outer boundary
Physical Line(2) = {2};  // airfoil
