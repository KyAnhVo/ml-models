#include "file_read.h"

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
