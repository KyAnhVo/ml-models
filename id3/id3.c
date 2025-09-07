#include "id3.h"

uint8_t param_count;
char param_names[MAX_PARAM_COUNT][100];
struct training_data training_set[MAX_TRAINING_DATA_SIZE];

void swap_training_data(struct training_data * a, struct training_data * b) {
    if (a == b) return;
    for (int i = 0; i < MAX_PARAM_COUNT; i++) {
        uint8_t temp = a->params[i];
        a->params[i] = b->params[i];
        b->params[i] = temp;
    }
    uint8_t temp = a->classification;
    a->classification = b->classification;
    b->classification = temp;
}

void ternary_partition(uint8_t param, size_t left, size_t right, size_t * b1, size_t * b2) {
    // apply Dutch National Flag here

    size_t end_1 = left, end_2 = left, end_3 = right - 1;
    for (size_t i = left; i < right; i++) {
        if (i >= end_3) break; // done
        if (training_set[i].params[param] == 0) {
            swap_training_data(&(training_set[end_1]), &(training_set[end_2]));
            swap_training_data(&(training_set[end_1]), &(training_set[i]));
            end_1++;
            end_2++;
        }
        else if (training_set[i].params[param] == 1) {
            swap_training_data(&(training_set[end_2]), &(training_set[i]));
            end_2++;
        }
        else {
            swap_training_data(&(training_set[end_3]), &(training_set[i]));
            end_3--;
        }
    }

    *b1 = end_1;
    *b2 = end_2;
}

double entropy_term(double p) {
    if (p == 0) return 0;
    else return p * log2(p);
}

double entropy(uint8_t * arr, uint8_t param, size_t left, size_t right) {
    double H1 = 0, H2 = 0, H3 = 0, amount1 = 0, amount2 = 0, amount3 = 0, total = 0;
    return (H1 * amount1 + H2 * amount2 + H3 * amount3) / total;
}
