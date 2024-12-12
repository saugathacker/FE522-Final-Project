#pragma once
#include "matrix.h"
#include "inputoutput.h"
#include <memory>
#include <iostream>


// ****************************LINEAR REGRESSION CLASS ***************************************
template<class T>
class LinearRegression {
private:
  const InputOutputFile<T> &file;
  const Matrix<T> X;  // regression matrix
  const Matrix<T> y;  // dependent variable vector
  Matrix<T> b_hat;    // estimates
  Matrix<T> b_covmat; // beta-estimates covariance matrix
  Matrix<T> yhat;     // ypredicted (in sample)
  bool fitted_model;  // boolean: if the model is fitted to the sample (X,y)
  bool include_bias;
  double mse;
  double Rsquared;
  double sigma_regression;
  double Fstat;
  

  

public:
  LinearRegression();
  LinearRegression(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias);
  LinearRegression(const InputOutputFile<T> &xfile, const InputOutputFile<T> &yfile, bool bias);
  ~LinearRegression();

  // methods
  Matrix<T> fit();
  Matrix<T> predict(const Matrix<T> &matrix);
  Matrix<T> stat_significance();
  Matrix<T> combined_hypothesis();
  Matrix<T> single_hypothesis();
  void summary_statistics();
  void coefficients();

};

// Constructors
// no inputs - create an empty object
template<class T>
LinearRegression<T>::LinearRegression()
: include_bias(false), X(X()), y(y()), b_hat(b_hat()), b_covmat(b_covmat()),yhat(yhat()), fitted_model(false), mse(0), Rsquared(0), sigma_regression(0), Fstat(0){}

// take as input an X matrix
template<class T>
LinearRegression<T>::LinearRegression(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias)
: include_bias(bias),b_hat(b_hat()), b_covmat(b_covmat()),yhat(yhat()), fitted_model(false),mse(0), Rsquared(0), sigma_regression(0), Fstat(0){
  if(X.getRows() != y.getRows()){
    throw std::invalid_argument("Length mismatch!  Number of observations for regressors different from the number of observations for the dependent variable");
  }
  y = ymat;
  X = matrix;
  if (bias == true){
    int numRows = matrix.getRows();
    Matrix<T> const_col(numRows, 1);
    for (int i = 0; i<numRows; i++){
      const_col.setElement(i,0,1);
    }
    X = const_col.Concatenate(matrix, false);
  }
}

template<class T>
// take as input a file
LinearRegression<T>::LinearRegression(const InputOutputFile<T> &xfile, const InputOutputFile<T> &yfile, bool bias)
: include_bias(bias), b_hat(b_hat()), b_covmat(b_covmat()),yhat(yhat()),fitted_model(false), mse(0), Rsquared(0), sigma_regression(0), Fstat(0){
  y = yfile.readFile();          // returns Xmatrix
  X = xfile.readFile(); // returns X matrix
  if(X.getRows() != y.getRows()){
    throw std::invalid_argument("Length mismatch!  Number of observations for regressors different from the number of observations for the dependent variable");
  }
  if (bias == true){
    int numRows = X.getRows();
    Matrix<T> const_col(numRows, 1);
    for (int i = 0; i<numRows; i++){
      const_col.setElement(i,0,1);
    }
    X = const_col.Concatenate(X, false);
  }
}

// Destructor
template <class T> 
LinearRegression<T>::~LinearRegression() {}

// Fitting regression Model
template <class T>
Matrix<T> LinearRegression<T>::fit(){
  // X'X)^-1 * X' * y
  b_hat =( (( (X.Transpose()).MatMul(X) ).Inverse()).MatMul(X.Transpose()) ).MatVecMul(y);
  fitted_model = true;
  yhat = X.MatMul(b_hat);


  b_covmat = ( (X.Transpose()).MatMul(X) ).Inverse()
  


}

// methods
//  Matrix<T> fit();
//  Matrix<T> stat_significance();
//  Matrix<T> combined_hypothesis();
//  Matrix<T> single_hypothesis();



