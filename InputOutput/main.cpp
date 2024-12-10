#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "inputoutput.h"

int main() {
	std::string file_name = "data.csv";

	try {
		InputOutputFile file(file_name);
		std::vector<std::vector<std::string>> data = file.readFile();

		for (const auto& row : data) {
			for (const auto& element : row) {
				std::cout << element << " ";
			}
			std::cout << std::endl;
		}
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
	}
	
	


	return 0;
}