#include <deal.II/base/vectorization.h>
#include <iostream>

using namespace dealii;

int main()
{
  std::cout << "SIMD width = " << VectorizedArray<double>::size() << "\n";
  return 0;
}

