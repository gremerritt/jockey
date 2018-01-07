#ifndef MODELHELPERS_H
#define MODELHELPERS_H

#include "neural_net.h"


void write_model(struct meta_neural_net *meta, char *filename);
char read_model_bulk(struct meta_neural_net *meta, char *filename);
FILE * open_model_file(char *filename);
void read_model_file(nn_type *dest, unsigned long int len, FILE *model_file);
char validate_model_file(struct meta_neural_net *meta, char *filename);


#endif
