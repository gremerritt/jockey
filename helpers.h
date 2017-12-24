#ifndef HELPERS_H
#define HELPERS_H

#include "constants.h"
#include "file_helpers.h"


typedef struct jcky_cli {
    int number_of_hidden_layers, number_of_nodes_in_hidden_layers, batch_size, seed;
    nn_type learning_rate;
    unsigned char memory_layout, num_blocks, action;
    unsigned int block_size;
    unsigned short int epochs;
    char training_filename[128], testing_filename[128];
    jcky_file training_file, testing_file;
} jcky_cli;

void welcome(unsigned char master);
unsigned char process_command_line(
    int argc,
    char **argv,
    jcky_cli *cli);
unsigned int round_up_multiple(unsigned int number, unsigned int multiple);


#endif
