#ifndef FILEHELPERS_H
#define FILEHELPERS_H

#include <stdio.h>
#include "neural_net.h"
#include "mpi_helper.h"

typedef struct jcky_file {
    FILE *stream;
    unsigned char offset;
    unsigned char datum_size;
    unsigned int bytes_per_record, bytes_per_data;
    unsigned int data_len, targets_len;
} jcky_file;

char jcky_write_file(nn_type **data, nn_type **targets, const unsigned int len, const unsigned int data_len, const unsigned int targets_len, char *filename);
void jcky_read_record(jcky_file *file, const unsigned int record, nn_type *batch, nn_type *targets);
char jcky_test_file(char *filename);
unsigned int jcky_get_num_inputs(jcky_file file);
unsigned int jcky_get_num_outputs(jcky_file file);
jcky_file jcky_open_file(char *filename);
char jcky_close_file(jcky_file file);
unsigned char jcky_file_byte_offset();

#endif
