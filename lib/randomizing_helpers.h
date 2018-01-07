#ifndef RANDOMIZING_HELPERS_H
#define RANDOMIZING_HELPERS_H


#include "constants.h"


int set_seed(int seed);
void generate_guassian_distribution(nn_type *numbers, int size);
void shuffle(unsigned int *array, int size);


#endif
