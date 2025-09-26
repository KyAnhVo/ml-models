#include "nb.h"

#include <fstream>
#include <sstream>
#include <iostream>

int get_nonempty_line(std::ifstream&, std::string&);

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "invalid argument list" << std::endl;
        return 1;
    }
    
    std::ifstream training_file(argv[1]);
    if (!training_file) return 1;
    
    std::string line, param_name;
    std::istringstream iss;
    std::vector<std::string> param_names;
    int param_count = 0;

    // get param names
    get_nonempty_line(training_file, line); 
    iss = std::istringstream(line);
    while (iss >> param_name) {
        param_names.push_back(param_name);
    }
    param_names.pop_back();
    param_count = param_names.size();

    // initialize NB predictor
    std::vector<int> param_domains(param_count, 2);
    NB predictor(0, param_count, param_domains, 2);
    
    // get lines
    std::vector<int> input;
    int tmp;
    int output;
    while (get_nonempty_line(training_file, line) == 0) {
        iss = std::istringstream(line);
        for (int i = 0; i < predictor.param_count; i++){
            iss >> tmp;
            input.push_back(tmp);
        }
        iss >> output;
        predictor.append_input(input, output);
    }
}

int get_nonempty_line(std::ifstream& file, std::string& line) {
    while (std::getline(file, line)) {
        if (line.size() != 0)
            return 0;
    }
    return -1;
}
