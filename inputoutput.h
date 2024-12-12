#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

template <class T>
class InputOutputFile {
private:
	std::string file_name;                    // file name
	std::vector<std::string> column_names;    // column names - if included
	bool includes_column_names;               // flag variable to control for column names

public:
	// constructor
	InputOutputFile(const std::string& filename_, bool includes_column_names_ = true) : file_name(filename_), includes_column_names(includes_column_names_) {}

	// method readFile - to read files - declaration
	std::vector<std::vector<std::string> > readFile();

	//method to run a LR model and print out the results in another file
	void OutputLRResults();


};


// method to read file from csv
template<class T>
std::vector<std::vector<std::string> > InputOutputFile<T>::readFile() {

	std::ifstream file(file_name);
	if (!file.is_open()) {
		throw std::runtime_error("Error: could not open file! ");
	}
	std::vector<std::vector<std::string> > data;
	std::string line_;

	if (includes_column_names && std::getline(file, line_)) {

		std::stringstream columnstream(line_);
		std::string column_name;
		while (std::getline(columnstream, column_name, ',')) {
			column_names.push_back(column_name);
		}
	}
	while (std::getline(file, line_)) {
		std::vector<std::string > row;
		std::stringstream oneline(line_);
		std::string element;

		while (std::getline(oneline, element, ',')) {
			row.push_back(element);
		}
		data.push_back(row);
	}

	file.close();
	return data;

}

template <class T>
void InputOutputFile<T>::OutputLRResults() {
}