#pragma once
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <utility>


template <class T> class Matrix {
private:
  int nRows; // Number of rows
  int nCols; // Number of columns
  int nElements; // Total number of elements
  std::unique_ptr<T[]> data; // Smart pointer for matrix data

  int getIndex(int row, int column) const;
  // Check if two values are close enough
  bool CloseEnough(T f1, T f2) const;
  // Swap two rows in the matrix
  void SwapRow(int i, int j);
  // Add multiple of one row to another
  void MultAdd(int i, int j, T multFactor);
  // Multiply a row by a factor
  void MultRow(int i, T multFactor);
  // Find the row with the maximum element in a column starting from a specific row
  int FindRowWithMaxElement(int colNumber, int startingRow);

public:
  // Constructors
  Matrix();
  Matrix(int rows, int columns);
  Matrix(int rows, int columns, T *elements);
  Matrix(int rows, int columns, std::unique_ptr<T[]> elements);
  Matrix(const Matrix<T> &matrix);
  Matrix(int nRows, int nCols, const std::vector<T> &inputData);

  // Destructor
  ~Matrix();

  // Configuration methods.
  bool Resize(int numRows, int numCols); // Resize the matrix
  void SetToIdentity(); // Set matrix to identity
  Matrix<T> getConcatenate(const Matrix<T> &matrix, bool by_row); // Concatenate two matrices
  double getSum() const; // Calculate the sum of all elements
  Matrix<T> Zeros(int nrows, int ncols); // Create a matrix of zeros
  
  // Split the matrix into two matrices
  std::pair<Matrix<T>, Matrix<T>> SplittoMatrices(int num1) const;

  // getters and setters
  int getRows() const;
  int getColumns() const;
  T getElement(int row, int column) const;
  bool setElement(int row, int column, T element);

  // = copy assignment
  Matrix<T> &operator=(const Matrix<T> &matrix);

  // operators ==,+,-,*
  //  == compare
  bool operator==(const Matrix<T> &matrix) const;
  // + addition
  Matrix<T> operator+(const Matrix<T> &matrix) const;
  template <class U> Matrix<T> operator+(const U &scalar) const;
  template <class U, class V>
  friend Matrix<U> operator+(const V &lhs, const Matrix<U> &matrix1);

  // - subtraction
  Matrix<T> operator-(const Matrix<T> &matrix) const;
  template <class U> Matrix<T> operator-(const U &scalar) const;
  template <class U, class V>
  friend Matrix<U> operator-(const V &lhs, const Matrix<U> &matrix);

  // * multiplication
  Matrix <T> MatMul(const Matrix<T> &matrix) const;  // matrix x matrix - matrix multiplication
  Matrix <T> MatVecMul(const Matrix<T> &vector) const; // matrix x vector 
  double InnerProduct(const Matrix<T> &vector) const; // vector x vector 
  

  // operator overloading for matrix multiplications
  Matrix<T> operator*(const Matrix<T> &matrix) const;
  template <class U>
  Matrix<T> operator*(const U &scalar) const;
  template <class U, class V>
  friend Matrix<U> operator*(const V &lhs, const Matrix<U> &matrix);

  // util methods
  bool IsSquare();
  double getDeterminant();
  template <class U>
  friend std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix);

  // matrix manipulation
  bool Separate(Matrix<T> &matrix1, Matrix<T> &matrix2, int colNum);
  bool Join(const Matrix<T> &matrix2);
  Matrix<T> FindSubMatrix(int rowNum, int colNum);
  Matrix<T> getSubmatrces(int row, int col) const;

  // Compute matrix inverse
  bool Inverse();
  // Return the transpose
  Matrix<T> getTranspose() const;

  // print methods
  void printVec() const;
  void printMat() const;
};

template <class T> Matrix<T>::Matrix() {
  nRows = 1;
  nCols = 1;
  nElements = 1;
  data = nullptr;
}

template <class T> Matrix<T>::Matrix(int rows, int columns) {
  if (rows < 0 || columns < 0) {
    throw std::invalid_argument("Invalid number of rows or columns");
  }
  nRows = rows;
  nCols = columns;
  nElements = rows * columns;
  data = std::make_unique<T[]>(nElements);
  for (int i = 0; i < nElements; i++) {
    data[i] = 0;
  }
}

template <class T> Matrix<T>::Matrix(int rows, int columns, T *elements) {
  if (rows < 0 || columns < 0) {
    throw std::invalid_argument("Invalid number of rows or columns");
  }
  nRows = rows;
  nCols = columns;
  nElements = rows * columns;
  data = std::make_unique<T[]>(nElements);
  std::copy(elements, elements + nElements,
            data.get()); // Copy elements into unique_ptr
}

template <class T>
Matrix<T>::Matrix(int rows, int columns, std::unique_ptr<T[]> elements) {
  if (rows < 0 || columns < 0) {
    throw std::invalid_argument("Invalid number of rows or columns");
  }
  nRows = rows;
  nCols = columns;
  nElements = rows * columns;
  data = std::move(elements); // Move elements into unique_ptr
}

template <class T> Matrix<T>::Matrix(const Matrix<T> &matrix) {
  nRows = matrix.nRows;
  nCols = matrix.nCols;
  nElements = matrix.nElements;
  data = std::make_unique<T[]>(nElements);
  std::copy(matrix.data.get(), matrix.data.get() + nElements, data.get());
}

template <class T>
Matrix<T>::Matrix(int rows, int cols, const std::vector<T> &inputData) {
  nRows = rows;
  nCols = cols;
  nElements = nRows * nCols;
  if (inputData.size() != nElements) {
    throw std::invalid_argument(
        "Input vector size does not match matrix dimensions.");
  }
  data = std::make_unique<T[]>(nElements);
  std::copy(inputData.begin(), inputData.end(), data.get());
}

template <class T> Matrix<T>::~Matrix() {
  nRows = 0;
  nCols = 0;
  nElements = 0;
  data.reset();
}

template <class T> Matrix<T> &Matrix<T>::operator=(const Matrix<T> &matrix) {
  if (this != &matrix) {
    nRows = matrix.nRows;
    nCols = matrix.nCols;
    nElements = matrix.nElements;
    if (data != nullptr) {
      data.reset();
    }
    data = std::make_unique<T[]>(nElements);
    std::copy(matrix.data.get(), matrix.data.get() + nElements, data.get());
  }

  return *this;
}

template <class T> int Matrix<T>::getIndex(int row, int column) const {
  if (row < 0 || row >= nRows || column < 0 || column >= nCols) {
    throw std::out_of_range("Invalid matrix index");
  }
  return row * nCols + column;
}

template <class T> bool Matrix<T>::CloseEnough(T f1, T f2) const {
  return std::fabs(f1 - f2) < 1e-6;
}

template <class T> void Matrix<T>::SwapRow(int i, int j) {
  T *temp = new T[nCols];

  for (int k = 0; k < nCols; k++) {
    temp[k] = data[getIndex(i, k)];
  }

  for (int k = 0; k < nCols; k++) {
    data[getIndex(i, k)] = data[getIndex(j, k)];
  }

  for (int k = 0; k < nCols; k++) {
    data[getIndex(j, k)] = temp[k];
  }

  delete[] temp;
}

template <class T> void Matrix<T>::MultAdd(int i, int j, T multFactor) {
  for (int k = 0; k < nCols; k++) {
    data[getIndex(i, k)] += multFactor * data[getIndex(j, k)];
  }
}

template <class T> void Matrix<T>::MultRow(int i, T multFactor) {
  for (int k = 0; k < nCols; k++) {
    data[getIndex(i, k)] *= multFactor;
  }
}

template <class T>
int Matrix<T>::FindRowWithMaxElement(int colNumber, int startingRow) {
  int maxRow = startingRow;
  T maxElement = std::fabs(
      data[startingRow * nCols +
           colNumber]); // Initialize with the first element in the column

  for (int i = startingRow + 1; i < nRows; i++) {
    T currentElement = std::fabs(
        data[i * nCols + colNumber]); // Access element at (i, colNumber)
    if (currentElement > maxElement) {
      maxElement = currentElement;
      maxRow = i; // Update maxRow to the current row
    }
  }

  return maxRow;
}

// getters and setters
template <class T> int Matrix<T>::getRows() const { return nRows; }

template <class T> int Matrix<T>::getColumns() const { return nCols; }

template <class T> T Matrix<T>::getElement(int row, int column) const {
  return data[getIndex(row, column)];
}

template <class T> bool Matrix<T>::setElement(int row, int column, T element) {
  if (row < 0 || row >= nRows || column < 0 || column >= nCols) {
    return false;
  }
  data[getIndex(row, column)] = element;
  return true;
}

// == operator
template <class T> bool Matrix<T>::operator==(const Matrix<T> &matrix) const {
  if (nRows != matrix.nRows || nCols != matrix.nCols) {
    return false;
  }
  for (int i = 0; i < nElements; i++) {
    if (!CloseEnough(data[i], matrix.data[i])) {
      return false;
    }
  }
  return true;
}

// + operator
template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &matrix) const {
  if (nRows != matrix.nRows || nCols != matrix.nCols) {
    throw std::invalid_argument("Matrix dimensions do not match for addition");
  }
  Matrix<T> result(nRows, nCols);
  for (int i = 0; i < nElements; i++) {
    result.data[i] = data[i] + matrix.data[i];
  }

  return result;
}

template <class T>
template <class U>
Matrix<T> Matrix<T>::operator+(const U &scalar) const {
  Matrix<T> result(nRows, nCols);
  for (int i = 0; i < nElements; i++) {
    result.data[i] = data[i] + static_cast<T>(scalar);
  }

  return result;
}

template <class U, class V>
Matrix<U> operator+(const V &lhs, const Matrix<U> &matrix) {

  return matrix + lhs;
}

// - operator
template <class T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &matrix) const {
  if (nRows != matrix.nRows || nCols != matrix.nCols) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for subtraction");
  }
  Matrix<T> result(nRows, nCols);
  for (int i = 0; i < nElements; i++) {
    result.data[i] = data[i] - matrix.data[i];
  }

  return result;
}

template <class T>
Matrix<T> Matrix<T>::getConcatenate(const Matrix<T> &matrix, bool by_row){
  
  if (by_row == true){
    if (nCols != matrix.getColumns()){
      throw std::invalid_argument("Length mismatch! Number of columns should match for concatenation");
      }else{
        Matrix<T> concatenated = Matrix(nRows+matrix.getRows(),nCols);
        // first matrix elements
        for (int i = 0; i < nRows; i++){
          for(int j = 0; j < nCols; j++){
            concatenated.setElement(i,j, getElement(i,j));
          }
        }
        // second matrix elements
        for(int k =0; k < matrix.getRows(); k++){
          for (int l = 0; l < matrix.getColumns(); l++){
            concatenated.setElement(k+nRows, l, matrix.getElement(k,l));
          }
        }
        return concatenated;
      }
    }else{
      // concatenate by column
      if (nRows  != matrix.getRows()){
        throw std::invalid_argument("Length mismatch! Number of row should match for concatenation");
      }else{
        Matrix<T> concatenated = Matrix(nRows, nCols + matrix.getColumns());
        // first matrix elements
        for (int i = 0; i < nRows; i++){
          for(int j = 0; j < nCols; j++){
            concatenated.setElement(i,j, getElement(i,j));
          }
        }
        // second matrix elements
        for(int k =0; k < matrix.getRows(); k++){
          for (int l = 0; l < matrix.getColumns(); l++){
            concatenated.setElement(k, l+nCols, matrix.getElement(k,l));
          }
        }
        return concatenated;
      }
      }
  }

template <class T>
double Matrix<T>::getSum() const{
  double sum = 0.0;
  for (int i = 0; i<nRows; i++){
    for (int j=0; j<nCols; j++){
      sum += getElement(i,j);
    }
  }
  return sum;
}

template<class T>
Matrix<T> Matrix<T>::Zeros(int nrows, int ncols){
  this->Resize(nrows, ncols);
  for(int i =0; i<nrows; i++){
    for(int j =0; j < ncols; j++){
      this->setElement(i,j,0);

    }
  }
  return *this;
}

template <class T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::SplittoMatrices(int num1) const{
  int TotalRows = getRows();
  int TotalCols = getColumns();
  Matrix<T> M1(num1, TotalCols);
  Matrix<T> M2(TotalRows-num1, TotalCols);
  for (int i = 0; i<TotalRows; i++){
    for (int j = 0; j<TotalCols; j++){
      if(i<num1){
        M1.setElement(i,j,getElement(i,j));
      }else{
        M2.setElement(i-num1, j, getElement(i,j));
      }
    } 
  }
  return std::make_pair(M1, M2);
}



template <class T>
template <class U>
Matrix<T> Matrix<T>::operator-(const U &scalar) const {
  Matrix<T> result(nRows, nCols);
  for (int i = 0; i < nElements; i++) {
    result.data[i] = data[i] - static_cast<T>(scalar);
  }

  return result;
}

template <class U, class V>
Matrix<U> operator-(const V &lhs, const Matrix<U> &matrix) {
  Matrix<U> result(matrix.getRows(), matrix.getColumns());
  for (int i = 0; i < matrix.nElements; i++) {
    result.data[i] = static_cast<U>(lhs) - matrix.data[i];
  }
  return result;
}

// * multiplication

// matrix x matrix - matrix multiplication
template <class T> Matrix <T> Matrix<T>::MatMul(const Matrix<T> &matrix) const{
  if (nCols!= matrix.getRows()){
    throw std::invalid_argument("Matrix Multiplication cannot be performed");
    return Matrix();
  }else{
    Matrix<T> Mulmatrix(nRows, matrix.getColumns());
    for (int i = 0; i <nRows; i++){
      for (int j = 0; j <matrix.getColumns(); j++){
        T sum = 0.0;
        for (int k = 0; k <nCols; k++ ){
          sum += getElement(i,k) * matrix.getElement(k, j);
        }
      Mulmatrix.setElement(i,j,sum);
      }
    }
    return Mulmatrix;
  }
}  

template <class T> Matrix <T> Matrix <T>::MatVecMul(const Matrix<T> &vector) const{ // matrix x vector 
  if(nCols != vector.getRows() || vector.getColumns()!=1){
    throw std::invalid_argument("Matrix Multiplication cannot be performed");
    return Matrix();
  }else{
    Matrix<T> matvec(nRows, 1);
    for (int i = 0; i <nRows; i++){
      T sum = 0;
      for (int j=0; j <nCols; j++){
        sum += getElement(i,j)*vector.getElement(j,0);
      }
      matvec.setElement(i,0,sum);
    }
    return matvec;
  }
} 

// inner product
template <class T>
double Matrix<T>::InnerProduct(const Matrix<T> &vector) const{ // vector x vector
if((nCols != vector.nRows || nRows != 1) && (nRows != vector.nCols || nCols != 1)){
  throw std::invalid_argument("Dimension mismatch!");
}
double sum = 0;

    // Case 1: Current matrix is a row vector
    if (nRows == 1 && vector.nCols == 1) {
        for (int j = 0; j < nCols; j++) {
            sum += getElement(0, j) * vector.getElement(j, 0);
        }
    }

    // Case 2: Current matrix is a column vector
    else if (nCols == 1 && vector.nRows == 1) {
        for (int i = 0; i < nRows; i++) {
            sum += getElement(i, 0) * vector.getElement(0, i);
        }
    }

    return sum;
}

// operator overloading for matrix multiplications (mat x mat) or (mat x vec)
template <class T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &matrix) const{
  if(matrix.getColumns()==1){              // matrix x vector case
    return MatVecMul(matrix);
  }else{                                   // matrix multiplication case
    return MatMul(matrix);
  }
  
}

// mutliplication with scalar
template <class T> 
template <class U>
Matrix<T> Matrix<T>::operator*(const U &scalar) const{
  Matrix<T> resultmat(nRows, nCols);
  for(int i = 0; i< nRows; i++){
    for(int j = 0; j <nCols; j++){
      resultmat.setElement(i,j,getElement(i,j)*scalar);
    }
  }
  return resultmat;
}

template <class U, class V>
Matrix<U> operator*(const V &lhs, const Matrix<U> &matrix){
  return matrix * lhs;
}

// confiuration methods
template <class T> bool Matrix<T>::Resize(int numRows, int numCols) {
  if (numRows < 0 || numCols < 0) {
    return false;
  }
  nRows = numRows;
  nCols = numCols;
  nElements = nRows * nCols;
  data.reset();
  data = std::make_unique<T[]>(nElements);
  return true;
}

template <class T> void Matrix<T>::SetToIdentity() {
  if (nRows != nCols) {
    throw std::invalid_argument("Matrix must be square to set to identity");
  }
  for (int i = 0; i < nElements; i++) {
    data[i] = 0;
  }
  for (int i = 0; i < nRows; i++) {
    data[getIndex(i, i)] = 1;
  }
}

// util methods
template <class T> bool Matrix<T>::IsSquare() { return nRows == nCols; }


template<class T> 
Matrix<T> Matrix<T>::getTranspose() const{
  Matrix<T> transposeMatrix(nCols, nRows);
  for (int i = 0; i <nCols; i ++){
    for (int j = 0; j < nRows; j++){
      transposeMatrix.setElement(i, j, getElement(j, i));
    }
  }
  return transposeMatrix;
}

template <class T> Matrix<T> Matrix<T>::getSubmatrces(int row, int col) const {
  Matrix<T> submatrix(nRows - 1, nCols - 1);
  int subrows = 0;
  int subcols = 0;
  for (int i = 0; i < nRows; i++) {
    subcols = 0;
    if (i == row)
      continue; // skip the specific row

    for (int j = 0; j < nCols; j++) {
      if (j == col)
        continue; // skip the specific column
      submatrix.setElement(subrows, subcols, getElement(i, j));
      subcols += 1;
    }
    subrows += 1;
  }
  return submatrix;
}

template <class T> double Matrix<T>::getDeterminant() {
  //check if the matrix is square:
  if (IsSquare()){
    // one element matrix
    if (nElements == 1){ 
      return getElement(0,0);

    }else if (nRows == 2 && nCols == 2) {         // 2 x 2 case
      double det = getElement(0, 0)*getElement(1, 1) - getElement(1, 0)*getElement(0, 1);
      return det;   
    }else{                                        // higher order dimensions
      double det = 0.0;
      for (int col = 0; col <nCols; col++){
        Matrix<T> subMatrix = getSubmatrces(0,col);
        double multiplier = (col % 2 == 0) ? 1 : -1;
        det += multiplier * getElement(0,col) * subMatrix.getDeterminant();
      }
      return det;
    } 
  }else{
      throw std::invalid_argument("Determinant cannot be defined for a non-square matrix.");
  }
 }


// overloading << for printing matrix
template <class U>
std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix) {
  for (int i = 0; i < matrix.nRows; i++) {
    for (int j = 0; j < matrix.nCols; j++) {
      os << matrix.data[matrix.getIndex(i, j)] << " ";
    }
    os << std::endl;
  }

  return os;
}

// matrix manipulation
// separating a matrix into two submatrices around the column number
template <class T>
bool Matrix<T>::Separate(Matrix<T> &matrix1, Matrix<T> &matrix2, int colNum) {
  int numRows = nRows;
  int numCols1 = colNum;
  int numCols2 = nCols - colNum;

  matrix1.Resize(numRows, numCols1);
  matrix2.Resize(numRows, numCols2);

  for (int i = 0; i < numRows; i++) {
    for (int j = 0; j < nCols; j++) {
      if (j < colNum) {
        matrix1.setElement(i, j, data[getIndex(i, j)]);
      } else {
        matrix2.setElement(i, j - colNum, data[getIndex(i, j)]);
      }
    }
  }
  return true;
}

// joining two submatrices into a single matrix
template <class T> bool Matrix<T>::Join(const Matrix<T> &matrix2) {
  if (nRows != matrix2.nRows) {
    throw std::invalid_argument("Matrix dimensions do not match for joining");
  }

  // Calculate the new dimensions
  int newCols = nCols + matrix2.nCols;
  std::unique_ptr<T[]> newData(new T[nRows * newCols]);

  // Copy elements from the original matrix and matrix2 into newData
  for (int i = 0; i < nRows; i++) {
    for (int j = 0; j < nCols; j++) {
      newData[i * newCols + j] =
          data[i * nCols + j]; // Copy from current matrix
    }
    for (int j = 0; j < matrix2.nCols; j++) {
      newData[i * newCols + nCols + j] =
          matrix2.data[i * matrix2.nCols + j]; // Copy from matrix2
    }
  }

  // Update dimensions and data
  nCols = newCols;
  nElements = nRows * nCols;
  data = std::move(newData);

  return true;
}

template <class T>
bool Matrix<T>::Inverse() {
    if (!IsSquare()) {
        throw std::invalid_argument("Matrix must be square to invert.");
    }

    // Augment the matrix with the identity matrix
    Matrix<T> identity(nRows, nCols);
    identity.SetToIdentity();
    int originalCols = nCols;
    Join(identity);

    // Perform Gauss-Jordan elimination
    for (int diagIndex = 0; diagIndex < nRows; diagIndex++) {
        int cRow = diagIndex;
        int cCol = diagIndex;

        // Ensure pivot is non-zero by swapping rows
        int maxRow = FindRowWithMaxElement(cCol, cRow);
        if (maxRow != cRow) {
            SwapRow(cRow, maxRow);
        }

        // Ensure pivot is non-zero after swapping
        if (CloseEnough(data[getIndex(cRow, cCol)], 0.0)) {
            throw std::invalid_argument("Matrix is singular and cannot be inverted.");
        }

        // Normalize pivot row
        T pivotValue = data[getIndex(cRow, cCol)];
        MultRow(cRow, 1.0 / pivotValue);

        // Eliminate all other rows in the column
        for (int row = 0; row < nRows; row++) {
            if (row != cRow) {
                T factor = -data[getIndex(row, cCol)];
                MultAdd(row, cRow, factor);
            }
        }
    }

    // Extract the inverse from the augmented matrix
    std::unique_ptr<T[]> newData(new T[nRows * originalCols]);
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < originalCols; j++) {
            newData[i * originalCols + j] = data[getIndex(i, j + originalCols)];
        }
    }
    data = std::move(newData);
    nCols = originalCols;
    nElements = nRows * nCols;
    return true;
}


// print methods
template <class T>
void Matrix<T>::printVec() const{
  if(getColumns() != 1){
    printMat();
  }else{
    std::cout << "[" ;
    for (int i = 0; i < getRows(); i++){
      std::cout << getElement(i,0) << " ";
    }
    std::cout << "]" << std::endl;
  }
}

template <class T> 
void Matrix<T>::printMat()const{
  if(getColumns()==1){
    printVec();
  }else{
    std::cout << "[";
    for(int i = 0; i<getRows(); i++){
      for(int j =0 ; j<getColumns(); j++){
        std::cout << getElement(i,j);
      }
      std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
  }
}