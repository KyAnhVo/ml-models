#include "id3.h"

uint8_t param_count;
char param_names[MAX_PARAM_COUNT][100];
struct training_data training_set[MAX_TRAINING_DATA_SIZE];

double entropy_term(double p) {
    if (p == 0) return 0;
    else return p * log2(p);
}

double entropy(uint8_t * arr, uint8_t param, size_t left, size_t right) {
    double H1 = 0, H2 = 0, H3 = 0, amount1 = 0, amount2 = 0, amount3 = 0, total = 0;
    return (H1 * amount1 + H2 * amount2 + H3 * amount3) / total;
}
