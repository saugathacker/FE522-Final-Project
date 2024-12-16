#include "gtest/gtest.h"
#include "LinearRegression.h"
#include "matrix.h"

class LinearRegressionTest : public ::testing::Test {
protected:
    LinearRegression<double> lr; // LinearRegression instance
    Matrix<double> X;            // Example regression matrix
    Matrix<double> y;            // Example dependent variable
    Matrix<double> b_hat_expected; // Expected coefficients (for validation)

    void SetUp() override {
        // Initialize the regression matrix (X) and dependent variable (y)
        double X_data[] = {7.0, 4.0, 8.0, 
                           5.0, 7.0, 3.0, 
                           7.0, 8.0, 5.0,
                            4.0,8.0,8.0,
                            3.0,6.0,5.0}; // Example 5x3 matrix 5x4 with bias
        double y_data[] = {7.92, -3.53, -1.57, -4.92, -5.61};  // Example 5x1 vector

        X = Matrix<double>(5, 3, X_data);
        y = Matrix<double>(5, 1, y_data);

        // Initialize the LinearRegression instance with bias
        lr.setXY(X, y, true);
        double expectedArr[] = {-2.54170264 ,1.91195303, -1.82248662 ,0.53206182};
        b_hat_expected = Matrix<double>(4, 1, expectedArr);
    }
};

TEST_F(LinearRegressionTest, printingCoefs){
    lr.fit();
    auto coef = lr.coefficients(false);
    ASSERT_EQ(coef, b_hat_expected);
}