#include "nb.h"

#include <cmath>
#include <math.h>
#include <sstream>
#include <iomanip>

NB::NB(int dataset_size, int param_count, std::vector<int>& param_domains, int class_domain) {
    this->dataset_size = dataset_size;
    this->param_count = param_count;
    this->class_domain = class_domain;
    this->param_domains = std::vector<int>(param_domains);

    this->x = std::vector<std::vector<int>>(
            dataset_size, std::vector<int>(param_count, 0));
    this->y = std::vector<int>(dataset_size, 0);
    this->class_distribution = std::vector<int>(class_domain, 0);

    this->param_names = std::vector<std::string>(param_count);
}

NB::~NB() {}

int NB::append_input(std::vector<int>& training_input, int training_output) {
    if (training_input.size() != this->param_count)
        return 1;
    
    // check input and output wrt domains
    for (int i = 0; i < this->param_count; i++) {
        if (training_input[i] < 0 || training_input[i] >= this->param_domains[i])
            return -1;
    }
    if (training_output < 0 || training_output >= this->class_domain)
        return -1;

    this->x.push_back(training_input);
    this->y.push_back(training_output);
    this->dataset_size++;
    this->class_distribution[training_output]++;
    return 0;
}

int NB::change_input(std::vector<int>& training_input, int training_output, int row) {
    // check input wrt sizes of params
    if (row < 0 || row >= this->dataset_size)
        return 1;
    if (training_input.size() != this->param_count)
        return 1;
    
    // check input and output wrt domains
    for (int i = 0; i < this->param_count; i++) {
        if (training_input[i] < 0 || training_input[i] >= this->param_domains[i])
            return -1;
    }
    if (training_output < 0 || training_output >= this->class_domain)
        return -1;
    
    // change x, y wrt training_input and training_output
    for (int i = 0; i < this->param_count; i++) {
        this->x[row][i] = training_input[i];
    }
    this->y[row] = training_output;
    return 0;
}

double NB::p_class(int y) {
    if (y < 0 || y >= this->class_domain)
        return -1.0f;
    return static_cast<double>(this->class_distribution[y]) / dataset_size;
}

double NB::p_conditional(int param, int val, int y) {
    if (param < 0 || param >= this->param_count)
        return -1.0f;
    if (val < 0 || val >= this->param_domains[param])
        return -1.0f;
    if (y < 0 || y >= this->class_domain)
        return -1.0f;

    int p_cond_true = 0; // P(x_param = val | y) = p_cond_true / class_distribution[y]
    for (int i = 0; i < dataset_size; i++) {
        if (this->y[i] == y && this->x[i][param] == val)
            p_cond_true ++;
    }
    return static_cast<double>(p_cond_true) / this->class_distribution[y];
}

int NB::predict(const std::vector<int>& input) {
    if (input.size() != this->param_count)
        return -1;

    int chosen_class = -1;
    double max_val = -1 * INFINITY;

    for (int curr_class = 0; curr_class < this->class_domain; curr_class++) {
        double val = 0;
        for (int param = 0; param < this->param_count; param++) {
            if (input[param] < 0 || input[param] >= this->param_domains[param])
                return -1;
            val += std::log(this->p_conditional(param, input[param], curr_class));
        }
        val += std::log(this->p_class(curr_class));
        if (val > max_val) {
            chosen_class = curr_class;
            max_val = val;
        }
    }

    return chosen_class;
}

std::string NB::probability_string() {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    for (int curr_class = 0; curr_class < this->class_domain; curr_class++) {
        oss << "P(class=" << curr_class << ")=" << this->p_class(curr_class) << " ";
        for (int param = 0; param < this->param_count; param++) {
            for (int param_val = 0; param_val < this->param_domains[param]; param_val++) {
                oss << "P(" << this->param_names[param]  
                    << "=" << param_val << "|" << curr_class << ")=" 
                    << this->p_conditional(param, param_val, curr_class) 
                    << " ";
            }
        }
        oss << "\n";
    }
    return oss.str();
}
