#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


struct training_data {
    uint8_t * params;
    uint8_t class;
};

uint8_t param_count;
char param_names[100][100] = {{0}};
uint8_t * training_set;

struct node {
    // children[0] is child node of class 0, v.v for 1 and 2.
    struct node * children[3];
    uint8_t is_leaf;
    uint8_t * untouched_params;
    // indexes of training data s.t. training_set[left...right] are all in node's data set.
    size_t * left, * right;
};

/*****************************************************
 * AUXILIARY FUNCTIONS *******************************
 ****************************************************/

/**
 * @brief setup param_count, param_names, training_set
 *
 * @param fp file pointer to training data file
 */
void setup(FILE *);

/*****************************************************
 * ID3-RELATED FUNCTIONS *****************************
 ****************************************************/

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

/*****************************************************
 * MAIN FUNCTION *************************************
 ****************************************************/

int main(int argc, char** argv) {
    FILE * fp = fopen(argv[1], "r");
    setup(fp);
    printf("Param count: %d\n", param_count);
    for (int i = 0; i < param_count; i++)
        printf("\tPARAM %d: %s\n", i + 1, param_names[i]);
    return 0;
}

/*****************************************************
 * IMPLEMENTATION ************************************
 ****************************************************/

uint8_t is_whitespace(char c) {
    return c == ' ' ||
        c == '\t' ||
        c == '\n' ||
        c == '\r';
}

uint8_t is_empty_line(char * line) {
    uint8_t i = 0;
    while (line[i])
        if (!is_whitespace(line[i]))
            return 0;
    return 1;
}

void setup(FILE * fptr) {
    char line[100]  = {0};
    char tmp[100]   = {0};

    // read names and count inputs
    
    // skip line that does nothing
    do {
        fgets(line, 100, fptr);
    } while (is_empty_line(line));
    
    // read param names
    uint8_t i = 0, tmp_ind = 0;
    while (line[i]) {
        if (is_whitespace(line[i])) {
            if (tmp_ind == 0) ; // do nothing
            if (strcmp(tmp, "class") == 0) {
                // do nothing
            }
            else {
                // reset tmp and tmp_ind, go to next param name
                tmp[tmp_ind] = '\0';
                strcpy(param_names[param_count], tmp);
                param_count++;
                memset(tmp, 0, 100);
                tmp_ind = 0;
            }
        }
        else {
            // add current char to ind
            tmp[tmp_ind] = line[i];
            tmp_ind++;
        }
        i++;
    }
    
    // 
}

double entropy_term(double p) {
    if (p == 0) return 0;
    else return p * log2(p);
}

double entropy(uint8_t * arr, uint8_t param, size_t left, size_t right) {
    double H1 = 0, H2 = 0, H3 = 0, amount1 = 0, amount2 = 0, amount3 = 0, total = 0;
}
