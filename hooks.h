#ifndef HOOKS_H
#define HOOKS_H


#include "neural_net.h"


char write_file();
double get_score(unsigned short int batch_size, int number_of_outputs, nn_type *outputs, nn_type *targets);


#endif
