#ifndef ID3_H
#define ID3_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define MAX_PARAM_COUNT         100
#define MAX_TRAINING_DATA_SIZE  1000

struct training_data {
    uint8_t params[MAX_PARAM_COUNT];
    uint8_t classification;
};

struct node {
    // children[0] is child node of class 0, v.v for 1 and 2.
    struct node * children[3];
    // check if node is a leaf
    uint8_t is_leaf;
    // array of untouched parameters
    uint8_t untouched_params[MAX_PARAM_COUNT];
    // indexes of training data s.t. training_set[left...right] are all in node's data set.
    size_t * left, * right;
};

extern uint8_t param_count;
extern char param_names[MAX_PARAM_COUNT][100];
extern struct training_data training_set[MAX_TRAINING_DATA_SIZE];

/**
 * @brief partition subarray based on a certain parameter `param` for 0, 1, 2.
 *
 * Use Dutch National Flag (DNF) algorithm to partition the subarray from
 * index left to index right into 3 partitions: all 0's, all 1's, all 2's.
 * 
 * The values of b1 will be index starting all 1, and b2 will be index
 * starting at all 2
 *
 * @param arr the array pointer
 * @param left left index for partitioning start
 * @param right right index for partitioning end
 * @param param parameter to partition
 * @param b1 points to index later stores starting location of 2nd partition
 * @param b2 points to index later stores starting location of 3rd partition
 *
 * @note if b1 == 0 then there is no first section. if b1 == b2 then there is
 * no 2nd section. if  b2 == right + 1 then there is no 3rd section.
 */
void ternary_partition(uint8_t *, uint8_t, size_t, size_t, size_t *, size_t *);

/**
 * @brief given p, calculate p * lg(p)
 *
 * @param p double value as a probability
 * @return p * log_2(p)
 */
double entropy_term(double);

/**
 * @brief calculate entropy of a node.
 *
 * Given a node's left and right parameters and a parameter to split decisions,
 * calculate the potential entropy of next level given that this parameter is chosen.
 *
 * @param arr the array that is the training set
 * @param param the param that we split
 * @param left first index of subarray
 * @right last last index of subarray + 1
 */
double entropy(uint8_t *, uint8_t, size_t, size_t);

#endif
