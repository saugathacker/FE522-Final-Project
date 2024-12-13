#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include "matrix.h"

class InputOutputFile {
private:
	std::string file_name;

public:
	// constructor
	InputOutputFile(const std::string& filename_) : file_name(filename_){}

	// method readFile - to read files - declaration
	template <class T>
	std::pair<std::vector<T>,Matrix<T>> readFile(const std::string& firstColumnHeader);

	//method to run a LR model and print out the results in another file
	void OutputLRResults();


};


// method to read file from csv
template<class T>
std::pair<std::vector<T>,Matrix<T>>  InputOutputFile::readFile(const std::string& firstColumnHeader) {

    // std::cout << file_name << std::endl;
	std::ifstream file(file_name);
	if (!file.is_open()) {
		throw std::runtime_error("Error: could not open file! ");
	}

	std::vector<T> firstColumn;  // Vector to store the first column
    std::vector<T> matrixData; 

	int rowCount = 0, colCount = 0;
	std::string line_;
    bool headerFound = false;

	// Read the file line by line
    while (std::getline(file, line_)) {
        std::stringstream linestream(line_);
        std::string element;
        std::vector<std::string> row;

        // Locate the header row containing the first column header
        if (!headerFound) {
            while (std::getline(linestream, element, ',')) {
                row.push_back(element);
            }
            auto it = std::find(row.begin(), row.end(), firstColumnHeader);
            if (it != row.end()) {
                headerFound = true;
                continue;  // Move to the data rows
            } else {
                continue;  // Skip rows until the header is found
            }
        }

		// Process the data rows after the header row
        row.clear();
        linestream.clear();
        linestream.str(line_);


        // Process the row
        while (std::getline(linestream, element, ',')) {
            row.push_back(element);
        }

        // Handle first column separately
        firstColumn.push_back(static_cast<T>(std::stod(row[0])));

        // Process remaining columns
        for (size_t i = 1; i < row.size(); i++) {
            matrixData.push_back(static_cast<T>(std::stod(row[i])));
        }

        rowCount++;
        colCount = row.size() - 1;  // Update column count dynamically
    }

    file.close();

    // Convert remaining data into a Matrix
    Matrix<T> dataMatrix(rowCount, colCount, matrixData.data());

    return std::make_pair(firstColumn, dataMatrix);

}

void InputOutputFile::OutputLRResults() {
}