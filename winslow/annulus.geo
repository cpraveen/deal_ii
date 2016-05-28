Mesh.RecombineAll=1;
Mesh.RecombinationAlgorithm=1; //blossom
//Geometry.Normals = 100;

ri = 1.0;
ro = 2.0;

Point(1) = {0, 0, 0};

Point(2) = {ri, 0, 0};
Point(3) = {0, ri, 0};
Point(4) = {-ri, 0, 0};
Point(5) = {0, -ri, 0};

Point(6) = {ro, 0, 0};
Point(7) = {0, ro, 0};
Point(8) = {-ro, 0, 0};
Point(9) = {0, -ro, 0};

Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};

Circle(5) = {6,1,7};
Circle(6) = {7,1,8};
Circle(7) = {8,1,9};
Circle(8) = {9,1,6};

Line(9) = {2,6};
Line(10) = {3,7};
Line(11) = {4,8};
Line(12) = {5,9};

Line Loop(1) = {9, 5, -10, -1};
Plane Surface(1) = {1};
Transfinite Surface(1) = {2,6,7,3};

Line Loop(2) = {10, 6, -11, -2};
Plane Surface(2) = {2};
Transfinite Surface(2) = {3,7,8,4};

Line Loop(3) = {11, 7, -12, -3};
Plane Surface(3) = {3};
Transfinite Surface(3) = {4,8,9,5};

Line Loop(4) = {12, 8, -9, -4};
Plane Surface(4) = {4};
Transfinite Surface(4) = {5,9,6,2};

Transfinite Line{1,2,3,4,5,6,7,8} = 10;
Transfinite Line{9,10,11,12} = 20 Using Bump 0.1;

Physical Surface(100) = {1,2,3,4};
Physical Line(1) = {1,2,3,4,5,6,7,8};
