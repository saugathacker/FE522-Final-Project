#pragma once
#include ".\matrix\matrix.h"
#include "inputoutput.h"
#include <memory>
#include <iostream>
#include <cmath>


// ****************************LINEAR REGRESSION CLASS ***************************************
template<class T>
class LinearRegression {
private:
  // regression data
  const Matrix<T> y;          // dependent variable (Nx1) vector
  const Matrix<T> X;          // regression matrix
  const Matrix<T> XT;         // X' - transpose matrix
  const Matrix<T> XTXinv;     // (X'X)^-1
  int Nsample;                // sample size
  int parameters;             // number of regressors
  int nEstimates;             // total_estimates of the model
  int dfs;                    // degrees of freedom

  // coefficients
  bool bias;                  // model with/without constant
  Matrix<T> b_hat;            // estimates
  Matrix<T> b_covmat;         // covariance matrix of the estimated betas

  bool fitted_model;          // boolean: if the model is fitted to the sample (X,y)

  // regression estimates
  Matrix<T> yhat;             // ypredicted (in sample)
  Matrix<T> residuals;        // residuals
  Matrix<T> tstats;           // t-statistics of estimates
  double sigma_regression;    // standard error of regression
  
  // Errors
  double mse;
  double TSS;
  double RSS;
  double ESS;

  // (Adj) Coefficient of Determination
  double Rsquared;
  double AdjRsquared;
 
  // statistical measures
  double F_stat;

  // Information Criteria
  double AIC;
  double SBIC;

  // configuration methods
  void validateData(const Matrix<T>& X, const Matrix<T>& y) const;
  Matrix<T> include_bias(const Matrix<T>& X);
  void InitializeMatrices();                         
  void InitializeValues(bool bias_);

  // private methods - helper functions for calculations
  void model_fit() const;
  void calcMatrices();
  void calcCoeffs();
  void UpdateValues();
  void calcCoeffsStats();
  void calcCoeffsOfDetermination();
  double Fstat();
  double selectCriticalValue(int alpha);
 
public:
  // constructors
  LinearRegression();              
  LinearRegression(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias_);  
  // destructors
  ~LinearRegression();             

  // helper function for the first constructor ()
  void setXY(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias_);

  // Model Fit & estimated values
  Matrix<T> fit();
  Matrix<T> coefficients(bool print_vals) const;
  Matrix<T> predictedInSample(bool print_vals) const; // In sample predictions

  // Methods for out of sample predictions
  Matrix<T> train_test_split(const Matrix<T> &matrix, double train_pct = 0.8);
  Matrix<T> predict(const Matrix<T> &matrix, bool print_vals=false); // vector of values - out of sample prediction
  // Single observation predictions
  double predictOne(const Matrix<T> &vector, bool print_val = false);

  // Confidence Intervals
  Matrix<T> CI_predictOne(const Matrix<T> &vector, bool print_vals = false, int alpha = 5);
  Matrix<T> CI_estimates(int beta_pos, bool print_vals = false, int alpha =5);
  
  
  void summary_statistics() const;

};

// Constructors
// no inputs - create an empty object
template<class T>
LinearRegression<T>::LinearRegression()
: y(y()),X(X()),XT(XT()),XTXinv(XTXinv()),Nsample(0),parameters(0),nEstimates(0),dfs(0),bias(false),b_hat(b_hat()),b_covmat(b_covmat()),
fitted_model(false),yhat(yhat()),residuals(residuals()),tstats(tstats()),sigma_regression(0),mse(0), TSS(0), RSS(0), ESS(0), Rsquared(0),
AdjRsquared(0), F_stat(0), AIC(0), SBIC(0){}

// take as input an X matrix
template<class T>
LinearRegression<T>::LinearRegression(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias_): XT(XT()), XTXinv(XTXinv()), bias(bias_),
fitted_model(false), sigma_regression(0), mse(0), TSS(0), RSS(0), ESS(0), Rsquared(0), AdjRsquared(0), F_stat(0), AIC(0), SBIC(0){
  X = matrix;
  y = ymat;
  validateData(X, y);
  parameters = X.getColumns();
  Nsample = X.getRows();
  nEstimates = parameters;
  dfs = Nsample-parameters;
  if (bias == true){
    X = include_bias(X); 
    nEstimates +=1;
    dfs -=1;
  }
  InitializeMatrices();
}

// Destructor
template <class T> 
LinearRegression<T>::~LinearRegression() {}

// helper function for the first constructor ()
template <class T>
void LinearRegression<T>::setXY(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias_){
  X = matrix;
  y = ymat;
  validateData(X, y);
  parameters = X.getColumns();
  Nsample = X.getRows();
  nEstimates = parameters;
  dfs = Nsample-parameters;
  if (bias == true){
    X = include_bias(X); 
    nEstimates +=1;
    dfs -=1;
  }
  InitializeMatrices();
  InitializeValues(bias_);
}

// configuration methods
template <class T>
void LinearRegression<T>::validateData(const Matrix<T>& X, const Matrix<T>& y) const{
  if (X.getRows() != y.getRows()){
     throw std::invalid_argument("Row mismatch between X and y.");
  }
}

template <class T>
Matrix<T> LinearRegression<T>::include_bias(const Matrix<T>& X){
  int numRows = X.getRows();
    Matrix<T> const_col(numRows, 1);
    for (int i = 0; i<numRows; i++){
      const_col.setElement(i,0,1);
    }
    X = const_col.Concatenate(X, false); 
  return X;
}

template <class T>
void LinearRegression<T>::InitializeMatrices() {
  b_hat.Zeros(nEstimates,1);               // Kx1 vector
  b_covmat.Zeros(nEstimates,nEstimates);   // KxK matrix
  tstats.Zeros(nEstimates, 1);             // Kx1 vector for t-statistics
  yhat.Zeros(Nsample,1);                   // Nx1 vector
  residuals.Zeros(Nsample,1);              // Nx1 vector  
}

template <class T>
void LinearRegression<T>::InitializeValues(bool bias_) {
  bias = bias_;
  fitted_model = false;
  mse = 0;
  TSS = 0;
  RSS = 0;
  ESS = 0;
  Rsquared = 0;
  AdjRsquared = 0;
  sigma_regression = 0;
  F_stat = 0; 
  AIC = 0;
  SBIC = 0;
}

// private methods - helper functions for calculations
template <class T>
void LinearRegression<T>::model_fit() const{
  if(fitted_model==false){
    throw std::invalid_argument("No regression Model Fitted");
  }
}

template <class T>
void LinearRegression<T>::calcMatrices(){
  XT = X.Transpose();
  XTXinv = (XT.MatMul(X)).Inverse();
}

template <class T>
void LinearRegression<T>::calcCoeffs(){
  // (X'X)^-1 * X' * y
  b_hat = XTXinv.MatMul(XT).MatVecMul(y);
}

template <class T>
void LinearRegression<T>::calcCoeffsStats(){
  b_covmat = XTXinv*pow(sigma_regression,2);            // betas covariance matrix
  for (int i = 0; i<b_hat.nRows; i++){
    double b = b_hat.getElement(i,0);
    double SE_b = sqrt(b_covmat.getElement(i,i));
    tstats.setElement(i,0,b/SE_b);
  }
}

template <class T>
void LinearRegression<T>::UpdateValues(){
  yhat = X.MatMul(b_hat);                                 // X*b_hat
  residuals = y - yhat;                                          // regression residuals
  RSS = (residuals.Transpose()).InnerProduct(residuals);         // residual sum of squares (RSS)
  sigma_regression = sqrt(RSS/dfs);
  mse = RSS/Nsample;                                                              // mean squared error (MSE)
  double y_mean = y.Sum()/Nsample;                                                // mean Y
  Matrix<T> total_errors = y-y_mean;                                              // Y_real - Y_mean
  TSS = (total_errors.Transpose()).InnerProduct(total_errors);             // Total Sum of Squares
  ESS = TSS - RSS;                                              // Explained Sum of Squares (regression)
  AIC = log(RSS/Nsample) + 2*static_cast<double>(nEstimates)/Nsample;
  SBIC = log(RSS/Nsample) + nEstimates*log(Nsample)/Nsample;
}

template <class T>
void LinearRegression<T>::calcCoeffsOfDetermination(){
  // R^2 
  Rsquared = ESS/TSS;
  AdjRsquared = 1 - (RSS/(Nsample-parameters)) / (TSS/(Nsample-1));
}

template<class T>
double LinearRegression<T>::Fstat(){
  int J = parameters;               // Estimates excluding bias if existing
  Matrix<T> Rmat;
  Rmat.Zeros(J,nEstimates);
  for(int i = 0; i<J; i++){
    Rmat.setElement(i,i,1); 
  }

  if (bias==true){
    Rmat.setElement(0,0,0);
  }

  // Fstat = ( (R*b -r)' (R CovMat_b R' )^-1  * (R*b -r) )/J - for statistical significance r = zeros vector
  Matrix<T> M1 = Rmat.MatVecMul(b_hat);
  Matrix<T> M2 = (Rmat.MatMul(b_covmat)).MatMul(Rmat.Transpose());
  Matrix<T> Tot_res = ((M1.Transpose()).MatMul(M2.Inverse())).MatMul(M1);
  double nom = Tot_res.getElement(0,0);
  return nom/J;
}

template <class T>
double LinearRegression<T>::selectCriticalValue(int alpha){
  double t_critical = 0;
  switch (alpha) {
    case 1:
        t_critical = 2.576;
        break;
    case 5:
        t_critical = 1.96;
        break;
    case 10:
       t_critical = 1.645;
        break;
    default:
      std::cout << "Please select a valid value for alpha (1, 5, 10) for the 99%, 95% or 90% Confidence Intervals";
        break;
}
return t_critical;  
}

// Fitting regression Model
template <class T>
Matrix<T> LinearRegression<T>::fit(){
  fitted_model = true;
  calcMatrices();     // X transpose - (X'X)^-1
  calcCoeffs();       // (X'X)^-1 * X' * y
  UpdateValues();
  calcCoeffsStats();   
  calcCoeffsOfDetermination(); 
  F_stat = Fstat();
}

template<class T>
Matrix<T> LinearRegression<T>::coefficients(bool print_vals) const{
  model_fit();
  if(print_vals){
    b_hat.printVec(); 
  }
  return b_hat;
}

template<class T>
Matrix<T> LinearRegression<T>::predictedInSample(bool print_vals) const{
  model_fit();
  if (print_vals){
    yhat.printVec();
  }
  return yhat;
}

// Methods for out of sample predictions
template<class T>
Matrix<T> LinearRegression<T>::predict(const Matrix<T> &matrix, bool print_vals){
  Matrix<T> predicted = matrix.MatVecMul(b_hat);
  if (print_vals){
    predicted.printVec();
  }
  return predicted;
}

template<class T>
Matrix<T> LinearRegression<T>::train_test_split(const Matrix<T> &matrix, double train_pct){
  Matrix<T> train_mat, test_mat;
  
  return train_mat, test_mat;
}


// Single observation predictions
template<class T>
double LinearRegression<T>::predictOne(const Matrix<T> &vector,bool print_vals){
  Matrix<T> pred = (vector.Transpose()).MatVecMul(b_hat);
  double prediction = pred.getElement(0,0);
  return prediction;
}


// Confidence Intervals
// CIs for estimates
template <class T>
Matrix<T> LinearRegression<T>::CI_estimates(int beta_pos, bool print_vals, int alpha){
  double SE = sqrt(b_covmat.getElement(beta_pos,beta_pos));
  double tcrit = selectCriticalValue(alpha);
  double LB = b_hat.getElement(beta_pos,0)-tcrit*SE;
  double UB = b_hat.getElement(beta_pos,0)+tcrit*SE;

  Matrix<T> CI(2,1);
  CI.setElement(0,0,LB);
  CI.setElement(1,0, UB);
  if (print_vals){
    CI.printVec();
  }
  return CI;
}

// CIs for a single observation
template<class T>
Matrix<T> LinearRegression<T>::CI_predictOne(const Matrix<T> &vector,bool print_vals, int alpha){
  double prediction = predictOne(vector);
  Matrix<T> mat = vector.Transpose().MatMul(XTXinv).MatVecMul(vector);
  double prediction_SE = sigma_regression*sqrt(mat.getElement(0,0)+1);
  double tcrit = selectCriticalValue(alpha);
  double LB = prediction-tcrit*prediction_SE;
  double UB = prediction+tcrit*prediction_SE;

  Matrix<T> CI(2,1);
  CI.setElement(0,0,LB);
  CI.setElement(1,0, UB);
  if (print_vals){
    CI.printVec();
  }
  return CI;
}


template<class T>
void LinearRegression<T>::summary_statistics() const{
  model_fit();
  std::cout << "----------------------" << std::endl;
  std::cout << "Regression Statistics " << std::endl;
  std::cout << "----------------------" << std::endl;
  std::cout << "Method: Ordinary Least Squares " << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Rsquared: " << Rsquared << std::endl;
  std::cout << "Adjusted Rsquared: " << AdjRsquared << std::endl;
  std::cout << "Information Criteria " << AdjRsquared << std::endl;
  std::cout << "AIC: " << AIC << std::endl;
  std::cout << "SBIC: " << SBIC << std::endl;
  std::cout << "Standard Error of Regression: " << sigma_regression << std::endl;
  std::cout << "Number of Observations: " << Nsample << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "ANOVA" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Source " << "degrees of freedom (df) " << "Sum of Squares (SS)" << "Mean Square (MS)" << "Fstat" << "Significance F" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Regression" << parameters << ESS << ESS/parameters  << F_stat << "Significance F" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Residuals" << dfs << RSS << RSS/dfs  << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "Total" << Nsample - 1 << TSS << TSS/(Nsample - 1) << std::endl;

  int alpha = 5;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout << "------------------------------------------------------------------------------" << std::endl;
  std::cout  << "Variable "<< "Coefficient" << "Tstat" << "pvalue"<< "95% CI (LB)"<< "95% CI (UB)" << std::endl;
  if(bias==true){
    double alpha = b_hat.getElement(0,0);   // coefficient    
    double t_stat = tstats.getElement(0,0); // Tstat
    Matrix<T> CI = CI_estimates(0, false, 5);
    std::cout  << "Intercept "<< b_hat.getElement(0,0) << tstats.getElement(0,0) << "pvalue"<< CI.getElement(0,0)<< CI.getElement(1,0) << std::endl;
    for(int i = 1; i<b_hat.nRows ; i++){
      double estimate = b_hat.getElement(i,0); 
      double t_stat = tstats.getElement(i,0);
      Matrix<T> CI = CI_estimates(i, false, 5);
      double LB = CI.getElement(0,0);
      double UB = CI.getElement(1,0);
      std::cout  << "Variable "<< b_hat.getElement(i,0) << t_stat << "pvalue"<< LB << UB << std::endl;
  }
  }else{
    std::cout  << "   -    " << "   -    " << "   -    " << "   -    "<< "   -    "<< "   -    " << std::endl;
    for(int i = 0; i<b_hat.nRows ; i++){
      double estimate = b_hat.getElement(i,0); 
      double t_stat = tstats.getElement(i,0);
      Matrix<T> CI = CI_estimates(i, false, 5);
      double LB = CI.getElement(0,0);
      double UB = CI.getElement(1,0);
      std::cout  << "Variable "<< b_hat.getElement(i,0) << t_stat << ""<< LB << UB << std::endl;
  }
    
}
}
