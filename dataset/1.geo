// Gmsh project created on Mon Apr  8 18:39:32 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2.2, 0, 0, 1.0};
//+
Point(3) = {2.2, 0.41, 0, 1.0};
//+
Point(4) = {0, 0.41, 0, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {4, 3};
//+
Point(5) = {0.2, 0.2, 0, 1.0};
//+
Point(6) = {0.25, 0.2, 0, 1.0};
//+
Point(7) = {0.15, 0.2, 0, 1.0};
//+
Point(8) = {0.2, 0.25, 0, 1.0};
//+
Point(9) = {0.2, 0.15, 0, 1.0};
//+
Circle(5) = {8, 5, 7};
//+
Circle(6) = {7, 5, 9};
//+
Circle(7) = {9, 5, 6};
//+
Circle(8) = {6, 5, 8};
//+
Recursive Delete {
  Point{5}; 
}
//+
Line Loop(1) = {1, 2, 3, -4};
//+
Line Loop(2) = {5, 6, 7, 8};
//+
Plane Surface(1) = {1, 2};
//+
Transfinite Line {1} = 30 Using Progression 1;
//+
Transfinite Line {3} = 15 Using Progression 1;
//+
Transfinite Line {4, 2} = 90 Using Progression 1.02;
//+
Transfinite Line {5, 8, 7, 6} = 20 Using Progression 1;
//+
Physical Line("94") = {1};
//+
Physical Line("96") = {2, 4};
//+
Physical Line("100") = {8, 5, 6, 7};
//+
Physical Surface("103") = {1};

