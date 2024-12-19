#include <iostream>
#include "matrix.h"
#include "inputoutput.h"
#include <filesystem>
#include <LinearRegression.h>


using namespace std;

int main() { 
  // read from file and create X matrix and Y vector
  InputOutputFile fileHandler("./data/model_testdata.csv");

  // readfile takes the name of the first column header as a parameter
  auto [yVector, xMatrix] = fileHandler.readFile<double>("EXSMSFT");
  Matrix<double> yMatrix = Matrix<double>(yVector.size(), 1, yVector);


  LinearRegression<double> lr(xMatrix, yMatrix, true);
  xMatrix = lr.getXMatrix();

  // splitting the data set to training and predict
  auto [trainXMatrix, testXMatrix] = lr.train_test_split(xMatrix, 0.9);
  auto [trainYVector, testYVector] = lr.train_test_split(yMatrix, 0.9);
  lr.setXY(trainXMatrix, trainYVector,false);

  // fits the model
  lr.fit();
  lr.predict(testXMatrix, true);
  double arr[] = {1, 0.08,0.67,0.28,-0.15,-1.22};
  Matrix<double> predictOneX(1,testXMatrix.getColumns(),arr );
  double yReal = testYVector.getElement(testYVector.getRows()-1, 0);
  lr.predictOne(predictOneX, true);
  lr.CI_predictOne(predictOneX, true);
  cout << "Real value of Y for the single predict: " << yReal << endl;
  stringstream summary = lr.summary_statistics_stream();
  // Specify the filename
  string filename = "test_output.txt";
  outputResultsInFile(summary, filename);

  return 0;
}