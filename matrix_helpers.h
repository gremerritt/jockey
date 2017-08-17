#ifndef MATRIXHELPERS_H
#define MATRIXHELPERS_H

#include "neural_net.h"

// This method multiplies the weight matrix by the
// preivous layers activation matrix, and adds the
// bias matrix to the result. The result is the
// layers so-called 'z_vector'
void calculate_z_matrix(nn_type *z_matrix,
                        nn_type *weight,
                        nn_type *activation,
                        nn_type *bias,
                        int weight_rows,
                        int weight_cols,
                        int activation_cols);

inline void calculate_z_matrix2(nn_type *z_matrix,
                        nn_type *weight,
                        nn_type *activation,
                        nn_type *bias,
                        int weight_rows,
                        int weight_cols,
                        int activation_cols);

inline void mmm(nn_type *z_matrix,
                        nn_type *weight,
                        nn_type *activation,
                        int weight_rows,
                        int weight_cols,
                        int activation_cols);
inline void vec_add(nn_type *z_matrix, nn_type *bias, int cols, int len);

inline void sigmoidify(nn_type *activation,
                nn_type *z_vector,
                int rows,
                int cols);

inline void delta_output_layer(nn_type *delta,
                        nn_type *activation,
                        nn_type *z_matrix,
                        int *target_value,
                        int outputs,
                        int batch_size);

void delta_hidden_layers(nn_type *delta,
                         nn_type *weight_downstream,
                         nn_type *delta_downstream,
                         nn_type *z_matrix,
                         int weight_rows,
                         int weight_cols,
                         int batch_size);

inline void mmm_T(nn_type *delta,
                        nn_type *weight_downstream,
                        nn_type *delta_downstream,
                        nn_type *z_matrix,
                        int weight_rows,
                        int weight_cols,
                        int batch_size);

inline void adjust_weight(nn_type *activation_initial,
                   nn_type *weight,
                   nn_type *delta,
                   int weight_rows,
                   int weight_cols,
                   int batch_size,
                   nn_type eta);

inline void adjust_bias(nn_type *bias,
                 nn_type *delta,
                 int dim,
                 int batch_size,
                 nn_type eta);

inline nn_type sigmoid(nn_type z);
inline nn_type sigmoidPrime(nn_type z);

inline void add_vectors(nn_type *trgt, nn_type *src, const unsigned int len);
inline void subtract_vectors(nn_type *trgt, nn_type *src, const unsigned long int len);
inline void copy_vectors(nn_type *trgt, nn_type *src, const unsigned long int len);

#endif
