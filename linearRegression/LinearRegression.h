#pragma once
#include "../matrix/matrix.h"
#include <memory>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <sstream>


// ****************************LINEAR REGRESSION CLASS ***************************************
template<class T>
class LinearRegression {
private:
  // regression data
  Matrix<T> y;          // dependent variable (Nx1) vector
  Matrix<T> X;          // regression matrix
  Matrix<T> XT;         // X' - transpose matrix
  Matrix<T> XTXinv;     // (X'X)^-1
  int numSamples;                // sample size
  int numParameters;             // number of regressors
  int numEstimates;             // total_estimates of the model
  int degreesOfFreedom;                    // degrees of freedom
  
  // model distributions
  boost::math::students_t_distribution<double> Students_dist;
  boost::math::fisher_f_distribution<double> F_dist;

  // coefficients
  bool bias;                  // model with/without constant
  Matrix<T> b_hat;            // estimates
  Matrix<T> b_covmat;         // covariance matrix of the estimated betas
  Matrix<T> beta_pvals;       // p-values of the estimates

  bool isFitted;          // boolean: if the model is fitted to the sample (X,y)

  // regression estimates
  Matrix<T> yhat;             // ypredicted (in sample)
  Matrix<T> residuals;        // residuals
  Matrix<T> tstats;           // t-statistics of estimates
  double stdErrorRegression;    // standard error of regression
  
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
  double Fpval;
  double Fstat();
  double Fstat_pval(double F_stat);

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
  void calcPvals();
  void UpdateValues();
  void calcCoeffsStats();
  void calcCoeffsOfDetermination();
  double selectCriticalValue(int alpha);

  friend class LinearRegressionTest;
 
public:
  // constructors
  LinearRegression();              
  LinearRegression(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias_);

  // helper function for the first constructor ()
  void setXY(const Matrix<T> &matrix, const Matrix<T> &ymat, bool bias_);
  void initializeRegression(const Matrix<T>& matrix, const Matrix<T>& ymat, bool bias_);

  // Model Fit & estimated values
 void fit();
  Matrix<T> coefficients(bool print_vals) const;
  // train - test split method
  std::pair<Matrix<T>, Matrix<T>> train_test_split(const Matrix<T> &matrix, double train_pct = 0.8);

  // methods for predictions
  Matrix<T> predictedInumSamples(bool print_vals) const; // In sample predictions
  Matrix<T> predict(const Matrix<T> &matrix, bool print_vals=false); // vector of values - out of sample prediction
  // Single observation predictions
  double predictOne(const Matrix<T> &vector, bool print_val = false);

  // Confidence Intervals
  Matrix<T> CI_predictOne(const Matrix<T> &vector, bool print_vals = false, int alpha = 5);
  Matrix<T> CI_estimates(int beta_pos, bool print_vals = false, int alpha =5);
  
  // Summary Statistics
  void summary_statistics();
  std::stringstream summary_statistics_stream();
  std::stringstream generateSummary();
  void appendCoefficients(std::stringstream& summary);
  
  Matrix<T> getXMatrix() const;
};

// Constructors
// no inputs - create an empty object
template<class T>
LinearRegression<T>::LinearRegression()
    : y(), X(), XT(), XTXinv(),
      numSamples(0), numParameters(0), numEstimates(0), degreesOfFreedom(0),
      bias(false), b_hat(), b_covmat(), beta_pvals(),
      isFitted(false), yhat(), residuals(), tstats(),
      stdErrorRegression(0), mse(0), TSS(0), RSS(0), ESS(0),
      Rsquared(0), AdjRsquared(0), F_stat(0), Fpval(0), AIC(0), SBIC(0),Students_dist(1),  // Default to 1 degree of freedom
      F_dist(1, 1) {}


// Parameterized Constructor
template<class T>
LinearRegression<T>::LinearRegression(const Matrix<T>& matrix, const Matrix<T>& ymat, bool bias_)
    : XT(), XTXinv(),
      bias(bias_), isFitted(false),
      stdErrorRegression(0), mse(0), TSS(0), RSS(0), ESS(0),
      Rsquared(0), AdjRsquared(0), F_stat(0), Fpval(0), AIC(0), SBIC(0), Students_dist(1),  // Default to 1 degree of freedom
      F_dist(1, 1) {
    initializeRegression(matrix, ymat, bias_);
}

// Helper Function for Initialization
template<class T>
void LinearRegression<T>::initializeRegression(const Matrix<T>& matrix, const Matrix<T>& ymat, bool bias_) {
    validateData(matrix, ymat);
    X = bias_ ? include_bias(matrix) : matrix;
    y = ymat;

    numParameters = matrix.getColumns();
    numSamples = X.getRows();
    numEstimates = X.getColumns();
    // debug it can be zero if numSamples and numEstimates be same
    degreesOfFreedom = numSamples - numEstimates;

    InitializeMatrices();
    InitializeValues(bias_);

    // Initialize distributions
    Students_dist = boost::math::students_t_distribution<double>(degreesOfFreedom);
    F_dist = boost::math::fisher_f_distribution<double>(numParameters, degreesOfFreedom);
}

// Re-initialize for Default Constructor
template<class T>
void LinearRegression<T>::setXY(const Matrix<T>& matrix, const Matrix<T>& ymat, bool bias_) {
    initializeRegression(matrix, ymat, bias_);
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
    
  return const_col.getConcatenate(X, false);
}

template <class T>
void LinearRegression<T>::InitializeMatrices() {
  b_hat.Zeros(numEstimates,1);               // Kx1 vector
  b_covmat.Zeros(numEstimates,numEstimates);   // KxK matrix
  // debug Zeros is returning the zero matrix instead of making tstats a matrix of zero
  tstats.Zeros(numEstimates, 1);             // Kx1 vector for t-statistics
  beta_pvals.Zeros(numEstimates,1);          // Kx1 vector for p-values of estimates
  yhat.Zeros(numSamples,1);                   // Nx1 vector
  residuals.Zeros(numSamples,1);              // Nx1 vector  
}

template <class T>
void LinearRegression<T>::InitializeValues(bool bias_) {
  bias = bias_;
  isFitted = false;
  mse = 0;
  TSS = 0;
  RSS = 0;
  ESS = 0;
  Rsquared = 0;
  AdjRsquared = 0;
  stdErrorRegression = 0;
  F_stat = 0; 
  Fpval = 0;
  AIC = 0;
  SBIC = 0;
}

// private methods - helper functions for calculations
template <class T>
void LinearRegression<T>::model_fit() const{
  if(isFitted==false){
    throw std::invalid_argument("No regression Model Fitted");
  }
}

template <class T>
void LinearRegression<T>::calcMatrices(){
  XT = X.getTranspose();
  XTXinv = XT*X;
  XTXinv.Inverse();
}

template <class T>
void LinearRegression<T>::calcCoeffs(){
  // (X'X)^-1 * X' * y
  b_hat = XTXinv.MatMul(XT).MatVecMul(y);
}

template <class T>
void LinearRegression<T>::calcCoeffsStats(){
  b_covmat = XTXinv*pow(stdErrorRegression,2);            // betas covariance matrix
  for (int i = 0; i<b_hat.getRows(); i++){
    double b = b_hat.getElement(i,0);
    double SE_b = sqrt(b_covmat.getElement(i,i));
    tstats.setElement(i,0,b/SE_b);
  }
}

template <class T>
void LinearRegression<T>::calcPvals(){
  Matrix<T> beta_pvals(tstats.getRows(),1);
  for (int i = 0; i<tstats.getRows(); i++){
    double t_stat = tstats.getElement(i,0);
    double temp_p = 1 - boost::math::cdf(Students_dist, std::fabs(t_stat));
    beta_pvals.setElement(i,0,2*temp_p);
  }
}

template <class T>
void LinearRegression<T>::UpdateValues(){
  yhat = X.MatMul(b_hat);                                 // X*b_hat
  residuals = y - yhat;                                          // regression residuals
  RSS = (residuals.getTranspose()).InnerProduct(residuals);         // residual sum of squares (RSS)
  stdErrorRegression = sqrt(RSS/degreesOfFreedom);
  mse = RSS/numSamples;                                                              // mean squared error (MSE)
  double y_mean = y.getSum()/numSamples;                                                // mean Y
  Matrix<T> total_errors = y-y_mean;                                              // Y_real - Y_mean
  TSS = (total_errors.getTranspose()).InnerProduct(total_errors);             // Total Sum of Squares
  ESS = TSS - RSS;                                              // Explained Sum of Squares (regression)
  AIC = log(RSS/numSamples) + 2*static_cast<double>(numEstimates)/numSamples;
  SBIC = log(RSS/numSamples) + numEstimates*log(numSamples)/numSamples;
}

template <class T>
void LinearRegression<T>::calcCoeffsOfDetermination(){
  // R^2 
  Rsquared = ESS/TSS;
  AdjRsquared = 1 - (RSS/(numSamples-numParameters)) / (TSS/(numSamples-1));
}

template<class T>
double LinearRegression<T>::Fstat() {
  int J = numParameters;
  F_stat = (ESS/J)
          /((RSS)/degreesOfFreedom);
  return F_stat;
}


template <class T>
double LinearRegression<T>::Fstat_pval(double F_stat){
  double pval = 1-boost::math::cdf(F_dist, F_stat);
  return pval;
}

template <class T>
double LinearRegression<T>::selectCriticalValue(int alpha){
  double t_critical = 0;
  switch (alpha) {
    case 1:
        t_critical = boost::math::quantile(Students_dist, 1-alpha/200.0);
        break;
    case 5:
        t_critical = boost::math::quantile(Students_dist, 1-alpha/200.0);
        break;
    case 10:
       t_critical = boost::math::quantile(Students_dist, 1-alpha/200.0);
        break;
    default:
      std::cout << "Please select a valid value for alpha (1, 5, 10) for the 99%, 95% or 90% Confidence Intervals";
        break;
}
return t_critical;  
}

// Fitting regression Model
template <class T>
void LinearRegression<T>::fit(){
  isFitted = true;
  calcMatrices();     // X transpose - (X'X)^-1
  calcCoeffs();       // (X'X)^-1 * X' * y
  UpdateValues();
  calcCoeffsStats();   
  calcCoeffsOfDetermination(); 
  F_stat = Fstat();
  Fpval = Fstat_pval(F_stat);
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
std::pair<Matrix<T>, Matrix<T>> LinearRegression<T>::train_test_split(const Matrix<T> &matrix, double train_pct){
  int TotalRows = matrix.getRows();
  int train_num_obs = static_cast<int>(train_pct*TotalRows);
  if(train_num_obs == 0 || train_num_obs == TotalRows){
    throw std::invalid_argument("This splitting does not create valid training and test samples");
  }
  std::pair<Matrix<T>, Matrix<T>> train_test = matrix.SplittoMatrices(train_num_obs);
  return train_test;
}

// prediction methods
template<class T>
Matrix<T> LinearRegression<T>::predictedInumSamples(bool print_vals) const{
  model_fit();
  if (print_vals){
    yhat.printVec();
  }
  return yhat;
}

template<class T>
Matrix<T> LinearRegression<T>::predict(const Matrix<T> &matrix, bool print_vals){
  Matrix<T> predicted = matrix.MatVecMul(b_hat);
  if (print_vals){
    predicted.printVec();
  }
  return predicted;
}

// Single observation predictions
template<class T>
double LinearRegression<T>::predictOne(const Matrix<T> &vector,bool print_vals){
  double pred = vector.InnerProduct(b_hat);
  if (print_vals){
    std::cout << "predIctOne: " << pred << std::endl;
  }
  return pred;
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
  Matrix<T> mat = vector.MatMul(XTXinv).MatVecMul(vector.getTranspose());
  double prediction_SE = stdErrorRegression*sqrt(mat.getElement(0,0)+1);
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
void LinearRegression<T>::summary_statistics() {
    model_fit();
    std::stringstream summary = generateSummary(); // Use the helper method
    std::cout << summary.str(); // Output the summary
}

template<class T>
std::stringstream LinearRegression<T>::summary_statistics_stream() {
    model_fit();
    return generateSummary(); // Use the same helper method
}

template<class T>
std::stringstream LinearRegression<T>::generateSummary() {
    std::stringstream summary;

    // Header for regression statistics
    summary << std::string(80, '-') << "\n";
    summary << "Regression Statistics\n";
    summary << std::string(80, '-') << "\n";
    summary << "Method: Ordinary Least Squares\n";

    // General statistics
    summary << std::string(80, '-') << "\n";
    summary << std::left << std::setw(25) << "R-squared:"
            << std::right << std::setw(15) << Rsquared << "\n";
    summary << std::left << std::setw(25) << "Adjusted R-squared:"
            << std::right << std::setw(15) << AdjRsquared << "\n";
    summary << std::left << std::setw(25) << "AIC:"
            << std::right << std::setw(15) << AIC << "\n";
    summary << std::left << std::setw(25) << "SBIC:"
            << std::right << std::setw(15) << SBIC << "\n";
    summary << std::left << std::setw(25) << "Standard Error:"
            << std::right << std::setw(15) << stdErrorRegression << "\n";
    summary << std::left << std::setw(25) << "Number of Observations:"
            << std::right << std::setw(15) << numSamples << "\n";
    summary << std::string(80, '-') << "\n";

    // ANOVA Table
    summary << "ANOVA\n";
    summary << std::string(80, '-') << "\n";
    summary << std::left << std::setw(15) << "Source"
            << std::setw(15) << "DF"
            << std::setw(15) << "SS"
            << std::setw(15) << "MS"
            << std::setw(15) << "F-stat"
            << std::setw(15) << "P-value" << "\n";
    summary << std::string(80, '-') << "\n";
    summary << std::left << std::setw(15) << "Regression"
            << std::setw(15) << numParameters
            << std::setw(15) << ESS
            << std::setw(15) << ESS / numParameters
            << std::setw(15) << F_stat
            << std::setw(15) << Fpval << "\n";
    summary << std::left << std::setw(15) << "Residuals"
            << std::setw(15) << degreesOfFreedom
            << std::setw(15) << RSS
            << std::setw(15) << RSS / degreesOfFreedom
            << std::setw(15) << "-"
            << std::setw(15) << "-" << "\n";
    summary << std::left << std::setw(15) << "Total"
            << std::setw(15) << (numSamples - 1)
            << std::setw(15) << TSS
            << std::setw(15) << TSS / (numSamples - 1)
            << std::setw(15) << "-"
            << std::setw(15) << "-" << "\n";
    summary << std::string(80, '-') << "\n";

    // Coefficients Table
    summary << "Coefficients\n";
    summary << std::string(80, '-') << "\n";
    summary << std::left << std::setw(15) << "Variable"
            << std::setw(15) << "Estimate"
            << std::setw(15) << "T-stat"
            << std::setw(15) << "P-value"
            << std::setw(15) << "95% CI (LB)"
            << std::setw(15) << "95% CI (UB)" << "\n";
    summary << std::string(80, '-') << "\n";
    appendCoefficients(summary); // Add coefficients data
    summary << std::string(80, '-') << "\n";

    return summary;
}

template<class T>
void LinearRegression<T>::appendCoefficients(std::stringstream& summary) {
    int offset = bias ? 1 : 0;

    if (bias) {
        double estimate = b_hat.getElement(0, 0);
        double t_stat = tstats.getElement(0, 0);
        double p_val = beta_pvals.getElement(0, 0);
        Matrix<T> CI = CI_estimates(0, false, 5);
        double LB = CI.getElement(0, 0);
        double UB = CI.getElement(1, 0);

        summary << std::left << std::setw(15) << "Intercept"
                << std::setw(15) << estimate
                << std::setw(15) << t_stat
                << std::setw(15) << p_val
                << std::setw(15) << LB
                << std::setw(15) << UB << "\n";
    }

    for (int i = offset; i < b_hat.getRows(); i++) {
        double estimate = b_hat.getElement(i, 0);
        double t_stat = tstats.getElement(i, 0);
        double p_val = beta_pvals.getElement(i, 0);
        Matrix<T> CI = CI_estimates(i, false, 5);
        double LB = CI.getElement(0, 0);
        double UB = CI.getElement(1, 0);

        summary << std::left << std::setw(15) << "Variable_" + std::to_string(i)
                << std::setw(15) << estimate
                << std::setw(15) << t_stat
                << std::setw(15) << p_val
                << std::setw(15) << LB
                << std::setw(15) << UB << "\n";
    }
}



template <class T>
Matrix<T> LinearRegression<T>::getXMatrix() const {
  return X;
}


