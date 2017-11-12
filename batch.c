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
    const unsigned int offset = iteration * batch_size;
    unsigned short int i;
	for (i=0; i<batch_size; i++) {
        jcky_read_record(file, sequence[offset + i], batch + (i * file->data_len), targets + (i * file->targets_len));
	}
}


void create_batch_no_sequence_file(
    nn_type *batch,
    nn_type *targets,
    jcky_file *file,
    const unsigned int batch_size,
    const unsigned int iteration,
    unsigned short int rank,
    sample_manager *sample_manager)
{
    const unsigned int offset = (iteration * batch_size) + (rank * (*sample_manager).base);
    unsigned short int i;
	for (i=0; i<batch_size; i++) {
        jcky_read_record(file, offset + i, batch + (i * file->data_len), targets + (i * file->targets_len));
	}
}
