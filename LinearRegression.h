#pragma once
#include "matrix.h"
#include "inputoutput.h"
#include <memory>
#include <iostream>
#include <system_error>


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
  Matrix<T> residuals; // regression residuals
  Matrix<T> tstats;    // t-statistics of estimates
  bool fitted_model;  // boolean: if the model is fitted to the sample (X,y)
  bool include_bias;
  double mse;
  double TSS;
  double RSS;
  double ESS;
  double Rsquared;
  double AdjRsquared;
  double sigma_regression;
  double Fstat;
  int Nsample;         // sample size

public:

  LinearRegression();
  LinearRegression(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias);
  LinearRegression(const InputOutputFile<T> &xfile, const InputOutputFile<T> &yfile, bool bias);
  ~LinearRegression();

  // for the first constructor ()
  void setXY(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias);
  void setXY(const InputOutputFile<T> &xfile, const InputOutputFile<T> &yfile, bool bias);

  // LR methods
  Matrix<T> fit();
  void coefficients() const;
  void ypredicted() const; // In sample predictions
  void summary_statistics() const;

  Matrix<T> predict(const Matrix<T> &matrix);
  Matrix<T> stat_significance();
  Matrix<T> combined_hypothesis();
  Matrix<T> single_hypothesis();
  
  

};

// Constructors
// no inputs - create an empty object
template<class T>
LinearRegression<T>::LinearRegression()
: include_bias(false), X(X()), y(y()), b_hat(b_hat()), b_covmat(b_covmat()), yhat(yhat()), residuals(residuals()),tstats(tstats()),fitted_model(false), mse(0),TSS(0),RSS(0),ESS(0),Rsquared(0),AdjRsquared(0), sigma_regression(0), Fstat(0), Nsample(0){}

// take as input an X matrix
template<class T>
LinearRegression<T>::LinearRegression(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias)
: include_bias(bias), fitted_model(false),mse(0),TSS(0),RSS(0),ESS(0), Rsquared(0), AdjRsquared(0), sigma_regression(0), Fstat(0){
  X = matrix;
  y = ymat;
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
  Nsample = X.nRows;
  int params = X.nCols;      // might or might not include the bias
  b_hat.Zeros(params,1);           // Kx1 vector
  tstats.Zeros(params, 1);         // Kx1 vector for t-statistics
  yhat.Zeros(Nsample,1);          // Nx1 vector
  b_covmat.Zeros(params,params);  // KxK matrix
  residuals.Zeros(Nsample,1);     // Nx1 vector

}

template<class T>
// take as input a file
LinearRegression<T>::LinearRegression(const InputOutputFile<T> &xfile, const InputOutputFile<T> &yfile, bool bias)
: include_bias(bias), b_hat(b_hat()), b_covmat(b_covmat()),yhat(yhat()),residuals(residuals()),fitted_model(false), mse(0),TSS(0),RSS(0),ESS(0), Rsquared(0), AdjRsquared(0), sigma_regression(0), Fstat(0){
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
  Nsample = X.nRows;
  int params = X.nCols;      // might or might not include the bias
  b_hat.Zeros(params,1);           // Kx1 vector
  tstats.Zeros(params, 1);         // Kx1 vector for t-statistics
  yhat.Zeros(Nsample,1);          // Nx1 vector
  b_covmat.Zeros(params,params);  // KxK matrix
  residuals.Zeros(Nsample,1);     // Nx1 vector
}

// Destructor
template <class T> 
LinearRegression<T>::~LinearRegression() {}

// for the first constructor
template <class T>
void LinearRegression<T>::setXY(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias){
  X = matrix;
  y = ymat;
   if(X.getRows() != y.getRows()){
    throw std::invalid_argument("Length mismatch!  Number of observations for regressors different from the number of observations for the dependent variable");
  }  
  if (bias == true){
    int numRows = matrix.getRows();
    Matrix<T> const_col(numRows, 1);
    for (int i = 0; i<numRows; i++){
      const_col.setElement(i,0,1);
    }
    X = const_col.Concatenate(matrix, false);
  }
  Nsample = X.nRows;
  int params = X.nCols;      // might or might not include the bias
  b_hat.Zeros(params,1);           // Kx1 vector
  tstats.Zeros(params, 1);         // Kx1 vector for t-statistics
  yhat.Zeros(Nsample,1);          // Nx1 vector
  b_covmat.Zeros(params,params);  // KxK matrix
  residuals.Zeros(Nsample,1);     // Nx1 vector
  include_bias = bias;
  fitted_model = false;
  mse = 0;
  TSS = 0;
  RSS = 0;
  ESS = 0;
  Rsquared = 0;
  AdjRsquared = 0;
  sigma_regression = 0;
  Fstat = 0;
}

template<class T>
void LinearRegression<T>::setXY(const InputOutputFile<T> &xfile, const InputOutputFile<T> &yfile, bool bias){
    X = xfile.readFile();
    y = yfile.readFile();
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
    Nsample = X.nRows;
    int params = X.nCols;      // might or might not include the bias
    b_hat.Zeros(params,1);           // Kx1 vector
    tstats.Zeros(params, 1);         // Kx1 vector for t-statistics
    yhat.Zeros(Nsample,1);          // Nx1 vector
    b_covmat.Zeros(params,params);  // KxK matrix
    residuals.Zeros(Nsample,1);     // Nx1 vector
    include_bias = bias;
    fitted_model = false;
    mse = 0;
    TSS = 0;
    RSS = 0;
    ESS = 0;
    Rsquared = 0;
    AdjRsquared = 0;
    sigma_regression = 0;
    Fstat = 0;
}

// Fitting regression Model
template <class T>
Matrix<T> LinearRegression<T>::fit(){
  fitted_model = true;
  // X'X)^-1 * X' * y
  b_hat =( (( (X.Transpose()).MatMul(X) ).Inverse()).MatMul(X.Transpose()) ).MatVecMul(y);
  yhat = X.MatMul(b_hat);
  residuals = y - yhat;
  int params = b_hat.nRows;
  sigma_regression = sqrt( (residuals.Transpose()).InnerProduct(residuals)/(Nsample-params) );
  b_covmat = ( (X.Transpose()).MatMul(X) ).Inverse();
  mse = (residuals.Transpose()).InnerProduct(residuals)/Nsample;
  double y_mean = y.Sum();
  Matrix<T> diff = y-y_mean;
  double TSS = (diff.Transpose()).InnerProduct(diff);             // Total Sum of Squares
  double RSS = (residuals.Transpose()).InnerProduct(residuals);   // Residual Sum of Squares
  double ESS = TSS - RSS;                                         // Explained Sum of Squares (regression)
  Rsquared = ESS/TSS;
  
  int multiplier = Nsample-1;       // temp variable for the calculation of Adjusted R^2
  if (include_bias==false){ multiplier = Nsample; }
  AdjRsquared = 1 - ( (1-Rsquared)*multiplier)/(Nsample-params);

  for (int i = 0; i<b_hat.nRows; i++){
    double b = b_hat.getElement(i,0);
    double SE_b = sqrt(b_covmat.getElement(i,i));
    tstats.setElement(i,0,b/SE_b);
  }

}

template<class T>
void LinearRegression<T>::coefficients() const{
  if(fitted_model==false){
    throw std::invalid_argument("No regression Model Fitted");
  }
  std::cout << "[";
  for(int i = 0; i<b_hat.nRows; i++){
    std::cout << b_hat.getElement(i,0) << " ";
  }
  std::cout << "]" << std::endl;
}

template<class T>
void LinearRegression<T>::ypredicted() const{
  if(fitted_model==false){
    throw std::invalid_argument("No regression Model Fitted");
  }
  std::cout << "[";
  for(int i = 0; i<yhat.nRows; i++){
    std::cout << yhat.getElement(i,0) << " ";
  }
  std::cout << "]" << std::endl;
}


template<class T>
void LinearRegression<T>::summary_statistics() const{
  if(fitted_model==false){
    throw std::invalid_argument("No regression Model Fitted");
  }

  // calculations for degress of freedom - necessary to present results
  int bias_est = 0;
  int predictors = b_hat.nRows;               
  if(include_bias == true){                  // if bias is included
    predictors -= 1;
    bias_est += 1;
    }

  std::cout << "----------------------" << std::endl;
  std::cout << "Regression Statistics " << std::endl;
  std::cout << "----------------------" << std::endl;
  std::cout << "Method: Ordinary Least Squares " << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Rsquared: " << Rsquared << std::endl;
  std::cout << "Adjusted Rsquared: " << AdjRsquared << std::endl;
  std::cout << "Standard Error of Regression: " << sigma_regression << std::endl;
  std::cout << "Number of Observations: " << Nsample << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "ANOVA" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Source " << "degrees of freedom (df) " << "Sum of Squares (SS)" << "Mean Square (MS)" << "Fstat" << "Significance F" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Regression" << Nsample - predictors << ESS << ESS/predictors  << Fstat << "Significance F" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Residuals" << Nsample - predictors-bias_est << RSS << RSS/predictors  << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Total" << Nsample - bias_est << TSS << TSS/(Nsample - bias_est ) << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout  << "Variable "<< "Coefficient" << "Tstat" << "pvalue"<< "95% CI (LB)"<< "95% CI (UB)" << std::endl;
  if(include_bias==true){
    std::cout  << "Intercept "<< b_hat.getElement(0,0) << tstats.getElement(0,0) << "pvalue"<< "95% CI (LB)"<< "95% CI (UB)" << std::endl;
    for(int i = 1; i<b_hat.nRows ; i++){
      double LB =0 ;
      double UB = 0;
      double t_stat = tstats.getElement(i,0);
      std::cout  << "Variable "<< b_hat.getElement(i,0) << t_stat << "pvalue"<< LB << UB << std::endl;
  }
  }else{
    std::cout  << "   -    " << "   -    " << "   -    " << "   -    "<< "   -    "<< "   -    " << std::endl;
    for(int i = 0; i<b_hat.nRows ; i++){
      double LB =0 ;
      double UB = 0;
      double t_stat = tstats.getElement(i,0);
      std::cout  << "Variable "<< b_hat.getElement(i,0) << t_stat << "pvalue"<< LB << UB << std::endl;
  }
    
}

}

