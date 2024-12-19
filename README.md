
# Linear Regression Analysis with C++

This project implements a linear regression model using custom libraries for matrix operations, file handling, and regression analysis. The project demonstrates the use of a workflow to preprocess data, train a linear regression model, make predictions, and compute confidence intervals.

## Features

- **Matrix Operations**: A robust library for matrix manipulations and operations.
- **File Handling**: Reads data from CSV files, extracting target (`y`) and feature (`X`) values.
- **Linear Regression**:
  - Fit a model to training data.
  - Generate predictions for test data.
  - Compute confidence intervals for predictions.
  - Export summary statistics to an output file.

## Dependencies

This project uses the following libraries:

1. **Boost Math**:
   - Provides advanced mathematical functions, including tools for statistical analysis and optimization.
   - The Boost Math library is fetched and included automatically during the build process using CMake.
2. **Google Test**:
   - Used for testing the functionalities of the implemented classes and methods.

## Workflow in `main.cpp`

The `main.cpp` file implements the following workflow:

1. **Read Data**:
   - Reads data from a CSV file (`./data/model_testdata.csv`) using the `InputOutputFile` class.
   - Extracts the target column (`y`) and feature matrix (`X`) based on the specified column header (`EXSMSFT`).

2. **Prepare Data**:
   - Initializes a `Matrix` object for `y` values and ensures data consistency for `X` and `y`.

3. **Linear Regression Initialization**:
   - Creates a `LinearRegression` object with the feature matrix (`X`) and target matrix (`y`).
   - Optionally includes an intercept term.

4. **Data Splitting**:
   - Splits the dataset into training and test sets (90% training, 10% test) using the `train_test_split` method.

5. **Model Training**:
   - Fits the model to the training data using the `fit` method.

6. **Predictions**:
   - Generates predictions for the test dataset.
   - Predicts the value for a single observation (`predictOne`).
   - Computes confidence intervals for the single prediction (`CI_predictOne`).

7. **Summary Statistics**:
   - Generates summary statistics for the model (e.g., R-squared, p-values) and exports them to a file (`test_output.txt`).

8. **Output**:
   - Displays the real value of the target (`y`) for a single prediction and other relevant outputs.

## Build Instructions

### Prerequisites

- **C++ Compiler**: Supports C++17 or later.
- **CMake**: Version 3.20 or later.

### Building the Project

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Build the project:

   ```bash
   cmake -S . -B build
   cmake --build build
   ```

3. Run the executable:

   ```bash
   ./build/main
   ```

### Running Tests

To run the unit tests:

```bash
./build/test/test
```
