#ifndef NB_H
#define NB_H

#include <string>
#include <vector>

class NB {
    // input matrix: (m, n)
    std::vector<std::vector<int>> x;

    // output matrix: (m,)
    std::vector<int> y;

    // input size
    int dataset_size;

    // param count
    int param_count;

    // class_count
    int class_count;

    // class distribution
    std::vector<int> class_distribution;

    // param names
    std::vector<std::string> class_names;


public:
    NB(int, int, int);
    ~NB();
    float p_class(int);
    float p_conditional(int, int, int);
};

#endif
