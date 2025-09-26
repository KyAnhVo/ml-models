#include "nb.h"

#include <complex>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>

int get_nonempty_line(std::ifstream&, std::string&);
double test_file(NB&, std::string, int&);

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "invalid argument list" << std::endl;
        return 1;
    }
    
    std::ifstream training_file(argv[1]);
    if (!training_file) return 1;
    
    /**
     * CREATE NB PREDICTOR BASED ON TRAINING FILE
     */

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
        input.clear();
        iss = std::istringstream(line);
        for (int i = 0; i < predictor.param_count; i++){
            iss >> tmp;
            input.push_back(tmp);
        }
        iss >> output;
        predictor.append_input(input, output);
    }

    

    // add names
    for (int i = 0; i < param_count; i++) {
        predictor.param_names[i] = param_names[i];
    }

    /**
     * PRINT STRING OF P VALUES
     */

    std::cout << predictor.probability_string();

    /**
     * TEST ON TEST FILE
     */
    int l1, l2;
    double t1 = test_file(predictor, argv[1], l1) * 100;
    double t2 = test_file(predictor, argv[2], l2) * 100;
    std::cout << std::fixed << std::setprecision(2);
    std::cout
        << "\nAccuracy on training set (" << l1 << " instances): " << t1 << "%\n"
        << "\nAccuracy on test set (" << l2 << " instances): " << t2 << "%\n";
}

int get_nonempty_line(std::ifstream& file, std::string& line) {
    while (std::getline(file, line)) {
        if (line.size() != 0)
            return 0;
    }
    return -1;
}

double test_file(NB& predictor, std::string file, int& file_len) {
    std::ifstream fs(file);
    if (!fs) return -1;

    std::vector<int> input;
    int output, output_expected;
    int correct = 0, total = 0;

    int tmp, count = 0;
    std::istringstream iss;
    std::string line;
    get_nonempty_line(fs, line);
    while (get_nonempty_line(fs, line) == 0) {
        input.clear();
        iss = std::istringstream(line);
        for (int i = 0; i < predictor.param_count; i++){
            iss >> tmp;
            input.push_back(tmp);
        }
        iss >> output_expected;
        output = predictor.predict(input);
        if (output == output_expected)
            correct += 1;
        total += 1;
        count++;
    }
    file_len = total;
    return static_cast<double>(correct) / total;
}
