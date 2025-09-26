#include "nb.h"

NB::NB(int dataset_size, int param_count, int class_count = 2) {
    this->dataset_size = dataset_size;
    this->param_count = param_count;
    this->class_count = class_count;

    this->x = std::vector<std::vector<int>>(
            dataset_size, std::vector<int>(param_count, 0));
    this->y = std::vector<int>(dataset_size, 0);
    this->class_distribution = std::vector<int>(class_count, 0);

    this->class_names = std::vector<std::string>(param_count);
}

/**
 * Calculate P(y)
 *
 * @arg y output class
 * @returns probability that a given data is of class y
 */
float NB::p_class(int y) {
    if (y < 0 || y >= this->class_count)
        return -1.0f;
    return static_cast<float>(this->class_distribution[y]) / dataset_size;
}

/**
 * Calculate P(x_param = val | y)
 *
 * @arg param   parameter
 * @arg val     expected parameter value
 * @arg y       output class expectation
 * returns the probability that x_param = val given the data is of class y
 */
float NB::p_conditional(int param, int val, int y) {
    if (param < 0 || param >= this->param_count)
        return -1.0f;
    if (val != 0 && val != 1)
        return -1.0f;
    if (y < 0 || y >= this->class_count)
        return -1.0f;

    int p_cond_true = 0; // P(x_param = val | y) = p_cond_true / class_distribution[y]
    for (int i = 0; i < dataset_size; i++) {
        if (this->y[i] == y && this->x[i][param] == val)
            p_cond_true ++;
    }
    return static_cast<float>(p_cond_true) / this->class_distribution[y];
}
