#ifndef NB_H
#define NB_H

#include <string>
#include <vector>

class NB {

public:

    // input matrix: (m, n)
    std::vector<std::vector<int>> x;

    // output matrix: (m,)
    std::vector<int> y;

    // input size
    int dataset_size;

    // param count
    int param_count;

    // class_domain
    int class_domain;

    // class distribution
    std::vector<int> class_distribution;

    // param names
    std::vector<std::string> param_names;

    // param domains
    std::vector<int> param_domains;

    NB(int, int, std::vector<int>&, int);
    ~NB();

    int append_input(std::vector<int>&, int);

    /**
     * change the input matrix and output row to the input
     *
     * @param training_input the vector of inputs
     * @param training_output the training class
     * @param row the row waiting to be swapped
     *
     * @returns 0 if success, 1 if failed
     */
    int change_input(std::vector<int>&, int, int);

    /**
     * Calculate P(y)
     *
     * @arg y output class
     *
     * @returns probability that a given data is of class y
     */
    float p_class(int);

    /**
     * Calculate P(x_param = val | y)
     *
     * @arg param   parameter
     * @arg val     expected parameter value
     * @arg y       output class expectation
     *
     * @returns the probability that x_param = val given the data is of class y
     */
    float p_conditional(int, int, int);

    /**
     * Given an input vector, predict the class with the highest probability
     *
     * @arg input the input to be predicted
     *
     * @returns the class most likely represented by the input vector
     */
    int predict(const std::vector<int>&);
};

#endif
