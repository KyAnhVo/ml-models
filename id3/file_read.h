#ifndef FILE_READ_H
#define FILE_READ_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "id3.h"

/**
 * @brief setup param_count, param_names, training_set
 *
 * @param fp file pointer to training data file
 */
void setup(FILE *);

#endif
