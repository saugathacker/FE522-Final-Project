#include <gtest/gtest.h>
#include "matrix.h"

TEST(MatrixConstructorTest, DefaultConstructor) {
    Matrix<int> m;

    // Check dimensions
    EXPECT_EQ(m.getRows(), 1);
    EXPECT_EQ(m.getColumns(), 1);
}

TEST(MatrixConstructorTest, RowsColsConstructor) {
    Matrix<int> m(2,3);

    // Check dimensions
    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getColumns(), 3);

    // Check values to be zero
    for(int i = 0; i < m.getRows(); i++){
      for(int j = 0; j < m.getColumns(); j++){
        EXPECT_EQ(m.getElement(i,j), 0);
      }
    }
}

TEST(MatrixConstructorTest, RowsColsConstructor_Negative) {
    EXPECT_THROW(Matrix<int>(-2,-3), std::invalid_argument);
}

TEST(MatrixConstructorTest, ArrayPointerConstructor){
    int arr[6] = {1,2,3,4,5,6};
    Matrix<int> m(2,3,arr);

    // Check dimensions
    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getColumns(), 3);
    // Check values
  int index = 0;
    for(int i = 0; i < m.getRows(); i++){
      for(int j = 0; j < m.getColumns(); j++){
        EXPECT_EQ(m.getElement(i,j), arr[index]);
        index++;
      }
    }
}

TEST(MatrixConstructorTest, ArrayDoublePointerConstructor){
    double arr[6] = {1.2,2.3,3.4,4.5,5.6,6.7};
    Matrix<double> m(2,3,arr);

    // Check dimensions
    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getColumns(), 3);
    // Check values
  int index = 0;
    for(int i = 0; i < m.getRows(); i++){
      for(int j = 0; j < m.getColumns(); j++){
        EXPECT_EQ(m.getElement(i,j), arr[index]);
        index++;
      }
    }
}


TEST(MatrixConstructorTest, UniquePointerConstructor){
    std::unique_ptr<int[]>arr(new int[6]{1, 2, 3, 4, 5, 6});
    int arr2[6] = {1, 2, 3, 4, 5, 6};
    Matrix<int> m(2,3,std::move(arr));

    // Check dimensions
    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getColumns(), 3);
    // Check values
  int index = 0;
    for(int i = 0; i < m.getRows(); i++){
      for(int j = 0; j < m.getColumns(); j++){
        EXPECT_EQ(m.getElement(i,j), arr2[index]);
        index++;
      }
    }
}

TEST(MatrixConstructorTest, VectorConstructor){
    std::vector<int> vec = {1,2,3,4,5,6};
    Matrix<int> m(2,3,vec);

    // Check dimensions
    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getColumns(), 3);
    // Check values
  int index = 0;
    for(int i = 0; i < m.getRows(); i++){
      for(int j = 0; j < m.getColumns(); j++){
        EXPECT_EQ(m.getElement(i,j), vec[index]);
        index++;
      }
    }
}

TEST(MatrixConstructorTest, VectorConstructor_SizeMisMatch){
    std::vector<int> vec = {1,2,3,4,5};
    EXPECT_THROW(Matrix<int>(2,3,vec), std::invalid_argument);
}

TEST(MatrixConstructorTest, CopyConstructor){
  int arr[6] = {1,2,3,4,5,6};
   Matrix<int> m1(2,3, arr);
   Matrix<int> m2(m1);

    // Check dimensions
    EXPECT_EQ(m2.getRows(), 2);
    EXPECT_EQ(m2.getColumns(), 3);
    // Check values
  int index = 0;
    for(int i = 0; i < m2.getRows(); i++){
      for(int j = 0; j < m2.getColumns(); j++){
        EXPECT_EQ(m2.getElement(i,j), arr[index]);
        index++;
      }
    }
}

TEST(MatrixConstructorTest, CopyAssignment){
  int arr[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr);

  Matrix<int> m2;

  m2 = m1;
  // Check dimensions
  EXPECT_EQ(m2.getRows(), 2);
  EXPECT_EQ(m2.getColumns(), 3);

  int index = 0;
  for(int i = 0; i < m2.getRows(); i++){
    for(int j = 0; j < m2.getColumns(); j++){
      EXPECT_EQ(m2.getElement(i,j), arr[index]);
      index++;
    }
  }
}

TEST(MatrixGetElementTest, GetElement_Invalid_Index){
  int arr[6] = {1,2,3,4,5,6};
  Matrix<int> m(2,3, arr);
  EXPECT_THROW(m.getElement(-1,0), std::out_of_range);
}

TEST(MatrixSetElementTest, SetElement){
  int arr[6] = {1,2,3,4,5,6};
  Matrix<int> m(2,3, arr);
  m.setElement(0,0,10);
  EXPECT_EQ(m.getElement(0,0), 10);
}

TEST(MatrixSetElementTest, SetElement_Negative){
  int arr[6] = {1,2,3,4,5,6};
  Matrix<int> m(2,3, arr);
  EXPECT_EQ(m.setElement(-1,0,10), false);
}

// airthmetic + and - tests
TEST(MatrixAdditionTest, Addition_Matrix_Matrix){
  int arr1[6] = {1,2,3,4,5,6};
  int arr2[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr1);
  Matrix<int> m2(2,3, arr2);
  Matrix<int> m3 = m1 + m2;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for(int i = 0; i < m3.getRows(); i++){
    for(int j = 0; j < m3.getColumns(); j++){
      EXPECT_EQ(m3.getElement(i,j), arr1[index] + arr2[index]);
      index++;
    }
  }
}

TEST(MatrixAdditionTest, Addition_Matrix_Scalar){
  int arr1[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr1);
  Matrix<int> m3 = m1 + 2;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for(int i = 0; i < m3.getRows(); i++){
    for(int j = 0; j < m3.getColumns(); j++){
      EXPECT_EQ(m3.getElement(i,j), arr1[index] + 2);
      index++;
    }
  }
}

TEST(MatrixAdditionTest, Addition_Scalar_Matrix){
  int arr1[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr1);
  Matrix<int> m3 = 2 + m1;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for(int i = 0; i < m3.getRows(); i++){
    for(int j = 0; j < m3.getColumns(); j++){
      EXPECT_EQ(m3.getElement(i,j), 2 + arr1[index]);
      index++;
    }
  }
}

TEST(MatrixSubtractionTest, Subtraction_Matrix_Matrix){
  int arr1[6] = {1,2,3,4,5,6};
  int arr2[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr1);
  Matrix<int> m2(2,3, arr2);
  Matrix<int> m3 = m1 - m2;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for(int i = 0; i < m3.getRows(); i++){
    for(int j = 0; j < m3.getColumns(); j++){
      EXPECT_EQ(m3.getElement(i,j), arr1[index] - arr2[index]);
      index++;
    }
  }
}

TEST(MatrixSubtractionTest, Subtraction_Matrix_Scalar){
  int arr1[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr1);
  Matrix<int> m3 = m1 - 2;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for(int i = 0; i < m3.getRows(); i++){
    for(int j = 0; j < m3.getColumns(); j++){
      EXPECT_EQ(m3.getElement(i,j), arr1[index] - 2);
      index++;
    }
  }
}

TEST(MatrixSubtractionTest, Subtraction_Scalar_Matrix){
  int arr1[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr1);
  Matrix<int> m3 = 2 - m1;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for(int i = 0; i < m3.getRows(); i++){
    for(int j = 0; j < m3.getColumns(); j++){
      EXPECT_EQ(m3.getElement(i,j), 2 - arr1[index]);
      index++;
    }
  }
}

// == operator for int and double
TEST(MatrixEqualityTest, Equal_Int){
  int arr1[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr1);
  Matrix<int> m2(2,3, arr1);
  EXPECT_EQ(m1==m2, true);
}

TEST(MatrixEqualityTest, Equal_Double){
  double arr1[6] = {1.2,2.3,3.4,4.5,5.6,6.7};
  Matrix<double> m1(2,3, arr1);
  Matrix<double> m2(2,3, arr1);
  EXPECT_EQ(m1==m2, true);
}

TEST(MatrixEqualityTest, NotEqual_Int){
  int arr1[6] = {1,2,3,4,5,6};
  int arr2[6] = {1,2,3,4,5,7};
  Matrix<int> m1(2,3, arr1);
  Matrix<int> m2(2,3, arr2);
  EXPECT_EQ(m1==m2, false);
}

TEST(MatrixEqualityTest, NotEqual_Double){
  double arr1[6] = {1.2,2.3,3.4,4.5,5.6,6.7};
  double arr2[6] = {1.2,2.3,3.4,4.5,5.6,7.7};
  Matrix<double> m1(2,3, arr1);
  Matrix<double> m2(2,3, arr2);
  EXPECT_EQ(m1==m2, false);
}

TEST(MatrixEqualityTest, Preceision_Double){
  double arr1[2] = {1.22222222223, 2.33333333333};
  double arr2[2] = {1.22222222225, 2.33333333335};
  Matrix<double> m1(1,2, arr1);
  Matrix<double> m2(1,2, arr2);
  EXPECT_EQ(m1==m2, true);
}

TEST(MatrixConfiguration, Resize){
  int arr1[6] = {1,2,3,4,5,6};
  Matrix<int> m1(2,3, arr1);
  m1.Resize(3,4);
  // Check dimensions
  EXPECT_EQ(m1.getRows(), 3);
  EXPECT_EQ(m1.getColumns(), 4);
  // Check values
  int index = 0;
  for(int i = 0; i < m1.getRows(); i++){
    for(int j = 0; j < m1.getColumns(); j++){
      EXPECT_EQ(m1.getElement(i,j), 0);
    }
  }
}

TEST(MatrixConfiguration, Resize_Negative){
  Matrix<int> m1(2,3);
  EXPECT_EQ(m1.Resize(-2,3), false);
}

TEST(MatrixConfiguration, SetToIdenity){
  Matrix<int> m1(3,3);
  m1.SetToIdentity();
  // Check values
  for(int i = 0; i < m1.getRows(); i++){
    for(int j = 0; j < m1.getColumns(); j++){
      if(i == j){
        EXPECT_EQ(m1.getElement(i,j), 1);
      }
      else{
        EXPECT_EQ(m1.getElement(i,j), 0);
      }
    }
  }
}

TEST(MatrixConfiguration, SetToIdentity_NonSquare){
  Matrix<int> m1(2,3);
  EXPECT_THROW(m1.SetToIdentity(), std::invalid_argument);
}

TEST(MatrixConfiguration, IsSquare){
  Matrix<int> m1(2,2);
  EXPECT_EQ(m1.IsSquare(), true);
}

TEST(MatrixConfiguration, IsSquare_NonSquare){
  Matrix<int> m1(2,3);
  EXPECT_EQ(m1.IsSquare(), false);
}

TEST(MatrixManipulation, seperatingMatrices){
  int arr1[6] = {1,2,3,4,5,6};
  int left[4] = {1,2,4,5};
  int right[2] = {3,6};
  Matrix<int> m1(2,3, arr1);

  Matrix<int> leftM;
  Matrix<int> rightM;

  EXPECT_EQ(m1.Separate(leftM, rightM, 2),true);
  // Check dimensions
  EXPECT_EQ(leftM.getRows(), 2);
  EXPECT_EQ(leftM.getColumns(), 2);
  EXPECT_EQ(rightM.getRows(), 2);
  EXPECT_EQ(rightM.getColumns(), 1);

  //check values
  int index = 0;
  for(int i = 0; i < leftM.getRows(); i++){
    for(int j = 0; j < leftM.getColumns(); j++){
      EXPECT_EQ(leftM.getElement(i,j), left[index]);
      index++;
    }
  }

  index = 0;
  for(int i = 0; i < rightM.getRows(); i++){
    for(int j = 0; j < rightM.getColumns(); j++){
      EXPECT_EQ(rightM.getElement(i,j), right[index]);
      index++;
    }
  }
}

TEST(MatrixManipulation, joiningMatrices){
  int arr1[6] = {1,2,3,4,5,6};
  int arr2[3] = {7,8,9};
  int arr3[9] = {1,2,7,3,4,8,5,6,9};

  Matrix<int> m1(3,2, arr1);
  Matrix<int> m2(3,1, arr2);
  m1.Join(m2);

  // Check dimensions
  EXPECT_EQ(m1.getRows(), 3);
  EXPECT_EQ(m1.getColumns(), 3);
  // Check values
  int index = 0;
  for(int i = 0; i < m1.getRows(); i++){
    for(int j = 0; j < m1.getColumns(); j++){
      EXPECT_EQ(m1.getElement(i,j), arr3[index]);
      index++;
    }
  }
 
}

TEST(MatrixInverseTest, TwoByTwoMatrix) {
    double arr[4] = {4, 7, 2, 6};
    double expectedInverse[4] = {0.6, -0.7, -0.2, 0.4};

    Matrix<double> A(2, 2, arr);
    Matrix<double> expected(2, 2, expectedInverse);

    A.Inverse();

    EXPECT_TRUE(A == expected); // Verify the result matches the expected inverse
}


TEST(MatrixInverseTest, ThreeByThreeMatrix){
  double arr[9] = {3,0,2,
                  2,0,-2,
                  0,1,1};
  double expectedInverse[9] = {0.2, 0.2, 0.0,
                                -0.2, 0.3, 1,
                                0.2, -0.3, 0.0};
  Matrix<double> A(3,3, arr);
  Matrix<double> expected(3,3, expectedInverse);
  A.Inverse();
  EXPECT_TRUE(A == expected);
}