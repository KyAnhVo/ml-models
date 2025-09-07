#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "id3.h"
#include "file_read.h"

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






