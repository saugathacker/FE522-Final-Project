#include <iostream>
#include "matrix.h"

using namespace std;

int main() { 

  
  cout << "Hello Final Project!!" << endl; 

  double arr[4] = {4, 7, 2, 6};

  Matrix<double> A(2, 2, arr);

  A.Inverse();
  std::cout << A << std::endl; 

  return 0;
}