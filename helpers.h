#ifndef HELPERS_H
#define HELPERS_H


#include "neural_net.h"


void welcome(unsigned char master);
unsigned char process_command_line(
    int argc,
    char **argv,
    int *number_of_hidden_layers,
    int *number_of_nodes_in_hidden_layers,
    int *batch_size,
    nn_type *learning_rate,
    unsigned short int *epochs,
    int *seed,
    unsigned char *memory_layout,
    unsigned char *num_blocks,
    unsigned int *block_size,
    unsigned char *action,
    char *training_filename,
    char *testing_filename);
unsigned int round_up_multiple(unsigned int number, unsigned int multiple);


#endif
