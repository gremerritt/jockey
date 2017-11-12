#ifndef BATCH_H
#define BATCH_H

#include "file_helpers.h"

void create_batch_with_sequence_file(
    nn_type *batch,
    nn_type *targets,
    jcky_file *file,
    const unsigned int batch_size,
    const unsigned int iteration,
    unsigned int *sequence
);

void create_batch_no_sequence_file(
    nn_type *batch,
    nn_type *targets,
    jcky_file *file,
    const unsigned int batch_size,
    const unsigned int iteration,
    unsigned short int rank,
    sample_manager *sample_manager
);

#endif
