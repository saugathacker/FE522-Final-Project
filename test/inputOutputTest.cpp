#include "matrix.h"
#include "inputoutput.h"
#include <gtest/gtest.h>

TEST(InputOutputTest, readFile) {
    // Create a sample CSV file
    const std::string filename = "./test/sampleTest.csv";

    // Instantiate CSVHandler
    InputOutputFile fileHandler(filename);

    // Expected data
    std::vector<double> expectedFirstColumn = {196307, 196308, 196309, 196310, 196311, 196312, 196401, 196402, 196403, 196404};
    double expectedMatrixData[] = {
    -0.39, -0.41, -0.97, 0.27,
     5.07, -0.80,  1.80, 0.25,
    -1.57, -0.52,  0.13, 0.27,
     2.53, -1.39, -0.10, 0.29,
    -0.85, -0.88,  1.75, 0.27,
     1.83, -2.10, -0.02, 0.29,
     2.24,  0.13,  1.48, 0.30,
     1.54,  0.28,  2.81, 0.26,
     1.41,  1.23,  3.40, 0.31,
     0.10, -1.52, -0.67, 0.29
    };

    Matrix<double> expectedMatrix(10,4,expectedMatrixData);

    // Process the CSV
    auto [firstColumn, dataMatrix] = fileHandler.readFile<double>("Date");

    // Validate first column
    EXPECT_EQ(firstColumn.size(), expectedFirstColumn.size());
    for (size_t i = 0; i < firstColumn.size(); i++) {
        EXPECT_DOUBLE_EQ(firstColumn[i], expectedFirstColumn[i]);
    }

    // Validate matrix data
    EXPECT_EQ(dataMatrix.getRows(), expectedMatrix.getRows());
    EXPECT_EQ(dataMatrix.getColumns(), expectedMatrix.getColumns());
    for (size_t i = 0; i < dataMatrix.getRows(); i++) {
        for (size_t j = 0; j < dataMatrix.getColumns(); j++) {
            EXPECT_DOUBLE_EQ(dataMatrix.getElement(i, j), expectedMatrix.getElement(i,j));
        }
    }

    std::cout << expectedMatrix << std::endl;
}