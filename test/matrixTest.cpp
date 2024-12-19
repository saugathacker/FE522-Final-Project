#include "matrix.h"
#include <gtest/gtest.h>

TEST(MatrixConstructorTest, DefaultConstructor) {
  Matrix<int> m;

  // Check dimensions
  EXPECT_EQ(m.getRows(), 1);
  EXPECT_EQ(m.getColumns(), 1);
}

TEST(MatrixConstructorTest, RowsColsConstructor) {
  Matrix<int> m(2, 3);

  // Check dimensions
  EXPECT_EQ(m.getRows(), 2);
  EXPECT_EQ(m.getColumns(), 3);

  // Check values to be zero
  for (int i = 0; i < m.getRows(); i++) {
    for (int j = 0; j < m.getColumns(); j++) {
      EXPECT_EQ(m.getElement(i, j), 0);
    }
  }
}

TEST(MatrixConstructorTest, RowsColsConstructor_Negative) {
  EXPECT_THROW(Matrix<int>(-2, -3), std::invalid_argument);
}

TEST(MatrixConstructorTest, ArrayPointerConstructor) {
  int arr[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m(2, 3, arr);

  // Check dimensions
  EXPECT_EQ(m.getRows(), 2);
  EXPECT_EQ(m.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m.getRows(); i++) {
    for (int j = 0; j < m.getColumns(); j++) {
      EXPECT_EQ(m.getElement(i, j), arr[index]);
      index++;
    }
  }
}

TEST(MatrixConstructorTest, ArrayDoublePointerConstructor) {
  double arr[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};
  Matrix<double> m(2, 3, arr);

  // Check dimensions
  EXPECT_EQ(m.getRows(), 2);
  EXPECT_EQ(m.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m.getRows(); i++) {
    for (int j = 0; j < m.getColumns(); j++) {
      EXPECT_EQ(m.getElement(i, j), arr[index]);
      index++;
    }
  }
}

TEST(MatrixConstructorTest, UniquePointerConstructor) {
  std::unique_ptr<int[]> arr(new int[6]{1, 2, 3, 4, 5, 6});
  int arr2[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m(2, 3, std::move(arr));

  // Check dimensions
  EXPECT_EQ(m.getRows(), 2);
  EXPECT_EQ(m.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m.getRows(); i++) {
    for (int j = 0; j < m.getColumns(); j++) {
      EXPECT_EQ(m.getElement(i, j), arr2[index]);
      index++;
    }
  }
}

TEST(MatrixConstructorTest, VectorConstructor) {
  std::vector<int> vec = {1, 2, 3, 4, 5, 6};
  Matrix<int> m(2, 3, vec);

  // Check dimensions
  EXPECT_EQ(m.getRows(), 2);
  EXPECT_EQ(m.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m.getRows(); i++) {
    for (int j = 0; j < m.getColumns(); j++) {
      EXPECT_EQ(m.getElement(i, j), vec[index]);
      index++;
    }
  }
}

TEST(MatrixConstructorTest, VectorConstructor_SizeMisMatch) {
  std::vector<int> vec = {1, 2, 3, 4, 5};
  EXPECT_THROW(Matrix<int>(2, 3, vec), std::invalid_argument);
}

TEST(MatrixConstructorTest, CopyConstructor) {
  int arr[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr);
  Matrix<int> m2(m1);

  // Check dimensions
  EXPECT_EQ(m2.getRows(), 2);
  EXPECT_EQ(m2.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m2.getRows(); i++) {
    for (int j = 0; j < m2.getColumns(); j++) {
      EXPECT_EQ(m2.getElement(i, j), arr[index]);
      index++;
    }
  }
}

TEST(MatrixConstructorTest, CopyAssignment) {
  int arr[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr);

  Matrix<int> m2;

  m2 = m1;
  // Check dimensions
  EXPECT_EQ(m2.getRows(), 2);
  EXPECT_EQ(m2.getColumns(), 3);

  int index = 0;
  for (int i = 0; i < m2.getRows(); i++) {
    for (int j = 0; j < m2.getColumns(); j++) {
      EXPECT_EQ(m2.getElement(i, j), arr[index]);
      index++;
    }
  }
}

TEST(MatrixGetElementTest, GetElement_Invalid_Index) {
  int arr[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m(2, 3, arr);
  EXPECT_THROW(m.getElement(-1, 0), std::out_of_range);
}

TEST(MatrixSetElementTest, SetElement) {
  int arr[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m(2, 3, arr);
  m.setElement(0, 0, 10);
  EXPECT_EQ(m.getElement(0, 0), 10);
}

TEST(MatrixSetElementTest, SetElement_Negative) {
  int arr[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m(2, 3, arr);
  EXPECT_EQ(m.setElement(-1, 0, 10), false);
}

// airthmetic + and - tests
TEST(MatrixAdditionTest, Addition_Matrix_Matrix) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  int arr2[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m2(2, 3, arr2);
  Matrix<int> m3 = m1 + m2;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m3.getRows(); i++) {
    for (int j = 0; j < m3.getColumns(); j++) {
      EXPECT_EQ(m3.getElement(i, j), arr1[index] + arr2[index]);
      index++;
    }
  }
}

TEST(MatrixAdditionTest, Addition_Matrix_Scalar) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m3 = m1 + 2;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m3.getRows(); i++) {
    for (int j = 0; j < m3.getColumns(); j++) {
      EXPECT_EQ(m3.getElement(i, j), arr1[index] + 2);
      index++;
    }
  }
}

TEST(MatrixAdditionTest, Addition_Scalar_Matrix) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m3 = 2 + m1;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m3.getRows(); i++) {
    for (int j = 0; j < m3.getColumns(); j++) {
      EXPECT_EQ(m3.getElement(i, j), 2 + arr1[index]);
      index++;
    }
  }
}

TEST(MatrixSubtractionTest, Subtraction_Matrix_Matrix) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  int arr2[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m2(2, 3, arr2);
  Matrix<int> m3 = m1 - m2;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m3.getRows(); i++) {
    for (int j = 0; j < m3.getColumns(); j++) {
      EXPECT_EQ(m3.getElement(i, j), arr1[index] - arr2[index]);
      index++;
    }
  }
}

TEST(MatrixSubtractionTest, Subtraction_Matrix_Scalar) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m3 = m1 - 2;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m3.getRows(); i++) {
    for (int j = 0; j < m3.getColumns(); j++) {
      EXPECT_EQ(m3.getElement(i, j), arr1[index] - 2);
      index++;
    }
  }
}

TEST(MatrixSubtractionTest, Subtraction_Scalar_Matrix) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m3 = 2 - m1;
  // Check dimensions
  EXPECT_EQ(m3.getRows(), 2);
  EXPECT_EQ(m3.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m3.getRows(); i++) {
    for (int j = 0; j < m3.getColumns(); j++) {
      EXPECT_EQ(m3.getElement(i, j), 2 - arr1[index]);
      index++;
    }
  }
}

// == operator for int and double
TEST(MatrixEqualityTest, Equal_Int) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m2(2, 3, arr1);
  EXPECT_EQ(m1 == m2, true);
}

TEST(MatrixEqualityTest, Equal_Double) {
  double arr1[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};
  Matrix<double> m1(2, 3, arr1);
  Matrix<double> m2(2, 3, arr1);
  EXPECT_EQ(m1 == m2, true);
}

TEST(MatrixEqualityTest, NotEqual_Int) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  int arr2[6] = {1, 2, 3, 4, 5, 7};
  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m2(2, 3, arr2);
  EXPECT_EQ(m1 == m2, false);
}

TEST(MatrixEqualityTest, NotEqual_Double) {
  double arr1[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 6.7};
  double arr2[6] = {1.2, 2.3, 3.4, 4.5, 5.6, 7.7};
  Matrix<double> m1(2, 3, arr1);
  Matrix<double> m2(2, 3, arr2);
  EXPECT_EQ(m1 == m2, false);
}

TEST(MatrixEqualityTest, Preceision_Double) {
  double arr1[2] = {1.22222222223, 2.33333333333};
  double arr2[2] = {1.22222222225, 2.33333333335};
  Matrix<double> m1(1, 2, arr1);
  Matrix<double> m2(1, 2, arr2);
  EXPECT_EQ(m1 == m2, true);
}

TEST(MatrixConfiguration, Resize) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  Matrix<int> m1(2, 3, arr1);
  m1.Resize(3, 4);
  // Check dimensions
  EXPECT_EQ(m1.getRows(), 3);
  EXPECT_EQ(m1.getColumns(), 4);
  // Check values
  int index = 0;
  for (int i = 0; i < m1.getRows(); i++) {
    for (int j = 0; j < m1.getColumns(); j++) {
      EXPECT_EQ(m1.getElement(i, j), 0);
    }
  }
}

TEST(MatrixConfiguration, Resize_Negative) {
  Matrix<int> m1(2, 3);
  EXPECT_EQ(m1.Resize(-2, 3), false);
}

TEST(MatrixConfiguration, SetToIdenity) {
  Matrix<int> m1(3, 3);
  m1.SetToIdentity();
  // Check values
  for (int i = 0; i < m1.getRows(); i++) {
    for (int j = 0; j < m1.getColumns(); j++) {
      if (i == j) {
        EXPECT_EQ(m1.getElement(i, j), 1);
      } else {
        EXPECT_EQ(m1.getElement(i, j), 0);
      }
    }
  }
}

TEST(MatrixConfiguration, SetToIdentity_NonSquare) {
  Matrix<int> m1(2, 3);
  EXPECT_THROW(m1.SetToIdentity(), std::invalid_argument);
}

TEST(MatrixConfiguration, IsSquare) {
  Matrix<int> m1(2, 2);
  EXPECT_EQ(m1.IsSquare(), true);
}

TEST(MatrixConfiguration, IsSquare_NonSquare) {
  Matrix<int> m1(2, 3);
  EXPECT_EQ(m1.IsSquare(), false);
}

TEST(MatrixManipulation, seperatingMatrices) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  int left[4] = {1, 2, 4, 5};
  int right[2] = {3, 6};
  Matrix<int> m1(2, 3, arr1);

  Matrix<int> leftM;
  Matrix<int> rightM;

  EXPECT_EQ(m1.Separate(leftM, rightM, 2), true);
  // Check dimensions
  EXPECT_EQ(leftM.getRows(), 2);
  EXPECT_EQ(leftM.getColumns(), 2);
  EXPECT_EQ(rightM.getRows(), 2);
  EXPECT_EQ(rightM.getColumns(), 1);

  // check values
  int index = 0;
  for (int i = 0; i < leftM.getRows(); i++) {
    for (int j = 0; j < leftM.getColumns(); j++) {
      EXPECT_EQ(leftM.getElement(i, j), left[index]);
      index++;
    }
  }

  index = 0;
  for (int i = 0; i < rightM.getRows(); i++) {
    for (int j = 0; j < rightM.getColumns(); j++) {
      EXPECT_EQ(rightM.getElement(i, j), right[index]);
      index++;
    }
  }
}

TEST(MatrixManipulation, joiningMatrices) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};
  int arr2[3] = {7, 8, 9};
  int arr3[9] = {1, 2, 7, 3, 4, 8, 5, 6, 9};

  Matrix<int> m1(3, 2, arr1);
  Matrix<int> m2(3, 1, arr2);
  m1.Join(m2);

  // Check dimensions
  EXPECT_EQ(m1.getRows(), 3);
  EXPECT_EQ(m1.getColumns(), 3);
  // Check values
  int index = 0;
  for (int i = 0; i < m1.getRows(); i++) {
    for (int j = 0; j < m1.getColumns(); j++) {
      EXPECT_EQ(m1.getElement(i, j), arr3[index]);
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

TEST(MatrixInverseTest, ThreeByThreeMatrix) {
  double arr[9] = {3, 0, 2, 2, 0, -2, 0, 1, 1};
  double expectedInverse[9] = {0.2, 0.2, 0.0, -0.2, 0.3, 1, 0.2, -0.3, 0.0};
  Matrix<double> A(3, 3, arr);
  Matrix<double> expected(3, 3, expectedInverse);
  A.Inverse();
  EXPECT_TRUE(A == expected);
}

TEST(MatrixMultiplicationTest, Multiplication_Matrix_Matrix) {
  int arr1[6] = {1, 2, 3, 4, 5, 6};     // 2x3 Matrix
  int arr2[6] = {7, 8, 9, 10, 11, 12};  // 3x2 Matrix
  int expected[4] = {58, 64, 139, 154}; // 2x2 Matrix

  Matrix<int> m1(2, 3, arr1);
  Matrix<int> m2(3, 2, arr2);
  Matrix<int> result = m1.MatMul(m2);

  // Check dimensions
  EXPECT_EQ(result.getRows(), 2);
  EXPECT_EQ(result.getColumns(), 2);

  // Check values
  int index = 0;
  for (int i = 0; i < result.getRows(); i++) {
    for (int j = 0; j < result.getColumns(); j++) {
      EXPECT_EQ(result.getElement(i, j), expected[index]);
      index++;
    }
  }
}

TEST(MatrixMultiplicationTest, MatVecMul_Matrix_Vector) {
    int arr[6] = {1, 2, 3, 4, 5, 6}; // 2x3 Matrix
    int vec[3] = {7, 8, 9}; // 3x1 Vector
    int expected[2] = {50, 122}; // 2x1 Result Vector

    Matrix<int> m(2, 3, arr);
    Matrix<int> vector(3, 1, vec);
    Matrix<int> result = m.MatVecMul(vector);

    // Check dimensions
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getColumns(), 1);

    // Check values
    int index = 0;
    for (int i = 0; i < result.getRows(); i++) {
        EXPECT_EQ(result.getElement(i, 0), expected[index]);
        index++;
    }
}


TEST(MatrixMultiplicationTest, InnerProduct_Vector_Vector) {
    int vec1[3] = {1, 2, 3}; // 3x1 Vector
    int vec2[3] = {4, 5, 6}; // 3x1 Vector
    int expected = 32; // Inner product result

    Matrix<int> v1(3, 1, vec1);
    Matrix<int> v2(1, 3, vec2);
    double result = v1.InnerProduct(v2);

    // Check result
    EXPECT_EQ(result, expected);
}

TEST(MatrixMultiplicationTest, MatMul_InvalidDimensions) {
    int arr1[6] = {1, 2, 3, 4, 5, 6}; // 2x3 Matrix
    int arr2[6] = {7, 8, 9, 10, 11, 12}; // 2x3 Matrix (invalid for multiplication)

    Matrix<int> m1(2, 3, arr1);
    Matrix<int> m2(2, 3, arr2);

    EXPECT_THROW(m1.MatMul(m2), std::invalid_argument);
}

TEST(MatrixMultiplicationTest, InnerProduct_InvalidDimensions) {
    int vec1[3] = {1, 2, 3}; // 3x1 Vector
    int vec2[2] = {4, 5};    // 2x1 Vector (invalid for inner product)

    Matrix<int> v1(3, 1, vec1);
    Matrix<int> v2(2, 1, vec2);

    EXPECT_THROW(v1.InnerProduct(v2), std::invalid_argument);
}

TEST(MatrixMultiplicationTest, Operator_Matrix_Matrix) {
    int arr1[6] = {1, 2, 3, 4, 5, 6}; // 2x3 Matrix
    int arr2[6] = {7, 8, 9, 10, 11, 12}; // 3x2 Matrix
    int expected[4] = {58, 64, 139, 154}; // 2x2 Matrix

    Matrix<int> m1(2, 3, arr1);
    Matrix<int> m2(3, 2, arr2);
    Matrix<int> result = m1 * m2;

    // Check dimensions
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getColumns(), 2);

    // Check values
    int index = 0;
    for (int i = 0; i < result.getRows(); i++) {
        for (int j = 0; j < result.getColumns(); j++) {
            EXPECT_EQ(result.getElement(i, j), expected[index]);
            index++;
        }
    }
}

TEST(MatrixMultiplicationTest, Operator_Matrix_Scalar) {
    int arr[6] = {1, 2, 3, 4, 5, 6};
    int expected[6] = {2, 4, 6, 8, 10, 12};

    Matrix<int> m(2, 3, arr);
    Matrix<int> result = m * 2;

    // Check dimensions
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getColumns(), 3);

    // Check values
    int index = 0;
    for (int i = 0; i < result.getRows(); i++) {
        for (int j = 0; j < result.getColumns(); j++) {
            EXPECT_EQ(result.getElement(i, j), expected[index]);
            index++;
        }
    }
}

TEST(MatrixMultiplicationTest, Operator_Scalar_Matrix) {
    int arr[6] = {1, 2, 3, 4, 5, 6};
    int expected[6] = {3, 6, 9, 12, 15, 18};

    Matrix<int> m(2, 3, arr);
    Matrix<int> result = 3 * m;

    // Check dimensions
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getColumns(), 3);

    // Check values
    int index = 0;
    for (int i = 0; i < result.getRows(); i++) {
        for (int j = 0; j < result.getColumns(); j++) {
            EXPECT_EQ(result.getElement(i, j), expected[index]);
            index++;
        }
    }
}

TEST(MatrixMultiplicationTest, Operator_InvalidDimensions) {
    int arr1[6] = {1, 2, 3, 4, 5, 6}; // 2x3 Matrix
    int arr2[6] = {7, 8, 9, 10, 11, 12}; // 2x3 Matrix (invalid for multiplication)

    Matrix<int> m1(2, 3, arr1);
    Matrix<int> m2(2, 3, arr2);

    EXPECT_THROW(m1 * m2, std::invalid_argument);
}

TEST(MatrixUtilityTests, Transpose) {
    int arr[6] = {1, 2, 3, 4, 5, 6}; // 2x3 Matrix
    int expected[6] = {1, 4, 2, 5, 3, 6}; // 3x2 Matrix

    Matrix<int> original(2, 3, arr);
    Matrix<int> transpose = original.getTranspose();

    // Check dimensions
    EXPECT_EQ(transpose.getRows(), 3);
    EXPECT_EQ(transpose.getColumns(), 2);

    // Check values
    int index = 0;
    for (int i = 0; i < transpose.getRows(); i++) {
        for (int j = 0; j < transpose.getColumns(); j++) {
            EXPECT_EQ(transpose.getElement(i, j), expected[index]);
            index++;
        }
    }
}

TEST(MatrixUtilityTests, GetSubmatrices) {
    int arr[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // 3x3 Matrix
    int expected[4] = {1, 3, 7, 9}; // Submatrix excluding row 1, col 1

    Matrix<int> original(3, 3, arr);
    Matrix<int> submatrix = original.getSubmatrces(1, 1);

    // Check dimensions
    EXPECT_EQ(submatrix.getRows(), 2);
    EXPECT_EQ(submatrix.getColumns(), 2);

    // Check values
    int index = 0;
    for (int i = 0; i < submatrix.getRows(); i++) {
        for (int j = 0; j < submatrix.getColumns(); j++) {
            EXPECT_EQ(submatrix.getElement(i, j), expected[index]);
            index++;
        }
    }
}

TEST(MatrixUtilityTests, Determinant_2x2) {
    int arr[4] = {4, 3, 6, 3}; // 2x2 Matrix
    double expected = -6; // Determinant: (4*3 - 6*3)

    Matrix<int> original(2, 2, arr);
    double determinant = original.getDeterminant();

    EXPECT_NEAR(determinant, expected, 1e-9);
}

TEST(MatrixUtilityTests, Determinant_3x3) {
    int arr[9] = {6, 1, 1, 4, -2, 5, 2, 8, 7}; // 3x3 Matrix
    double expected = -306; // Determinant manually calculated

    Matrix<int> original(3, 3, arr);
    double determinant = original.getDeterminant();

    EXPECT_NEAR(determinant, expected, 1e-9);
}

TEST(MatrixUtilityTests, Determinant_NonSquare) {
    int arr[6] = {1, 2, 3, 4, 5, 6}; // 2x3 Matrix
    Matrix<int> nonSquare(2, 3, arr);

    EXPECT_THROW(nonSquare.getDeterminant(), std::invalid_argument);
}
