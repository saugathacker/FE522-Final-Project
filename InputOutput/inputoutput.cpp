#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "inputoutput.h"



// method to read file from csv
std::vector<std::vector<std::string> > InputOutputFile::readFile() {

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

void InputOutputFile::OutputLRResults() {

}