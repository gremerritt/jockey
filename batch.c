#include "file_helpers.h"
#include <stdio.h>

void create_batch_with_sequence_file(
    nn_type *batch,
    nn_type *targets,
    jcky_file *file,
    const unsigned int batch_size,
    const unsigned int iteration,
    unsigned int *sequence)
{
    nn_type *batch_tmp = malloc( file->data_len * sizeof(nn_type) );
    const unsigned int offset = iteration * batch_size;
    unsigned short int i;
    unsigned int j;
	for (i=0; i<batch_size; i++) {
        jcky_read_record(file, sequence[offset + i], batch_tmp, targets + (i * file->targets_len));

        for (j=0; j<file->data_len; j++) {
            batch[(j*batch_size) + i] = batch_tmp[j];
        }
    }

    free(batch_tmp);
}


void create_batch_no_sequence_file(
    nn_type *batch,
    nn_type *targets,
    jcky_file *file,
    const unsigned int batch_size,
    const unsigned int iteration,
    unsigned short int rank,
    unsigned int process_offset)
{
    nn_type *batch_tmp = malloc( file->data_len * sizeof(nn_type) );
    const unsigned int offset = (iteration * batch_size) + (rank * process_offset);
    unsigned short int i;
    unsigned int j;
	for (i=0; i<batch_size; i++) {
        jcky_read_record(file, offset + i, batch_tmp, targets + (i * file->targets_len));

        for (j=0; j<file->data_len; j++) {
            batch[(j*batch_size) + i] = batch_tmp[j];
        }
    }

    free(batch_tmp);
}
