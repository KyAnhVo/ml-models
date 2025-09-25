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


float NB::p_class(int y) {
    if (y < 0 || y >= this->class_count)
        return -1.0f;
    return static_cast<float>(this->class_distribution[y]) / dataset_size;
}
