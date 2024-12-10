#pragma once
#include <vector>
#include <string>

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