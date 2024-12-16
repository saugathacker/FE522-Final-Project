#include <iostream>
#include "matrix.h"
#include "inputoutput.h"
#include <filesystem>
#include <LinearRegression.h>


using namespace std;

int main() { 
  cout << "Working directory: " << std::filesystem::current_path() << endl;
  // read from file and create X matrix and Y vector
  InputOutputFile fileHandler("./data/model_testdata.csv");
  auto [yVector, xMatrix] = fileHandler.readFile<double>("EXSMSFT");
  Matrix<double> yMatrix = Matrix<double>(yVector.size(), 1, yVector);
  cout << xMatrix.getColumns() << endl;
  cout << xMatrix.getRows() << endl;
  cout << xMatrix.getElement(1,0) << endl;

  cout << yMatrix.getColumns() << endl;
  cout << yMatrix.getRows() << endl;
  cout << yMatrix.getElement(1,0) << endl;

  LinearRegression<double> lr(xMatrix, yMatrix, true);
  xMatrix = lr.getXMatrix();
  auto [trainXMatrix, testXMatrix] = lr.train_test_split(xMatrix, 0.9);
  auto [trainYVector, testYVector] = lr.train_test_split(yMatrix, 0.9);
  lr.setXY(trainXMatrix, trainYVector,false);
  lr.fit();
  lr.predict(testXMatrix, true);
  double arr[] = {1, 0.08,0.67,0.28,-0.15,-1.22};
  Matrix<double> predictOneX(1,testXMatrix.getColumns(),arr );
  double yReal = testYVector.getElement(testYVector.getRows()-1, 0);
  lr.predictOne(predictOneX, true);
  lr.CI_predictOne(predictOneX, true);
  cout << "Real value of Y for the single predict: " << yReal << endl;
  lr.summary_statistics();
  // std::stringstream testStream;
  // testStream << "This is a test2 string\n";
  // testStream << "Line 2 of the file\n";
  //
  // // Specify the filename
  // std::string filename = "test_output.txt";
  // fileHandler.outputResultsInFile(testStream, filename);

  return 0;
}