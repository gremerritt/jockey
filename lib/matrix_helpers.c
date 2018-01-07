#include <stdio.h>

#include "matrix_helpers.h"
#include "neural_net.h"


inline void calculate_z_matrix(
    nn_type *z_matrix,
    nn_type *weight,
    nn_type *activation,
    nn_type *bias,
    int weight_rows,
    int weight_cols,
    int activation_cols)
{
    int Act_col, W_row, W_col;

    for (Act_col=0; Act_col<activation_cols; Act_col++) {
        for (W_row=0; W_row<weight_rows; W_row++) {
            nn_type accum = 0.0;
            int weight_offset = W_row*weight_cols;
            for (W_col=0; W_col<weight_cols; W_col++) {
                accum += weight[weight_offset + W_col] * activation[(W_col * activation_cols) + Act_col];
            }
            accum += bias[W_row];
            z_matrix[(W_row*activation_cols) + Act_col] = accum;
        }
    }
}


inline void sigmoidify(
    nn_type *activation,
    nn_type *z_matrix,
    int rows,
    int cols)
{
    int i, size = rows*cols;
    for (i=0; i<size; i++) {
        activation[i] = sigmoid(z_matrix[i]);
    }
}


inline void delta_output_layer(
    nn_type *delta,
    nn_type *activation,
    nn_type *z_matrix,
    nn_type *target_values,
    int outputs,
    int batch_size)
{
    int i, j, offset, target, index;

    for (i=0; i<batch_size; i++) {
        offset = i * outputs;
        for (j=0; j<outputs; j++) {
            index = (j*batch_size) + i;
            delta[index] = (activation[index] - target_values[offset + j]) * sigmoidPrime(z_matrix[index]);
        }
    }
}


inline void delta_hidden_layers(
    nn_type *delta,
    nn_type *weight_downstream,
    nn_type *delta_downstream,
    nn_type *z_matrix,
    int weight_rows,
    int weight_cols,
    int batch_size)
{
    int weight_row, weight_col, delta_col;

    for (delta_col=0; delta_col<batch_size; delta_col++) {
        for (weight_col=0; weight_col<weight_cols; weight_col++) {
            nn_type accum = 0.0;
            for (weight_row=0; weight_row<weight_rows; weight_row++) {
                accum += weight_downstream[(weight_cols*weight_row) + weight_col] *
                         delta_downstream[(weight_row*batch_size) + delta_col];
            }
            accum *= sigmoidPrime(z_matrix[(weight_col*batch_size)+delta_col]);
            delta[(weight_col*batch_size)+delta_col] = accum;
        }
    }
}


inline void adjust_weight(
    nn_type *activation,
    nn_type *weight,
    nn_type *delta,
    int weight_rows,
    int weight_cols,
    int batch_size,
    nn_type eta)
{
    int i, j, k;
    nn_type accum;
    double scale = eta / batch_size;

    for (i=0; i<weight_rows; i++) {
        int offset = i*weight_cols;
        for (j=0; j<weight_cols; j++) {
            accum = 0.0;
            int activation_offset = j*batch_size;
            int delta_offset = i*batch_size;
            for(k=0; k<batch_size; k++) {
                accum += activation[activation_offset + k] * delta[delta_offset + k];
            }
            weight[offset + j] -= scale * accum;
        }
    }
}


inline void adjust_bias(
    nn_type *bias,
    nn_type *delta,
    int dim,
    int batch_size,
    nn_type eta)
{
    int i, j, offset;
    double scale = eta / batch_size;
    nn_type accum;

    for (i=0; i<dim; i++) {
        accum = 0.0;
        offset = i*batch_size;
        for (j=0; j<batch_size; j++) {
            accum += delta[offset + j];
        }
        bias[i] -= scale * accum;
    }
}


inline nn_type sigmoid(nn_type z) {
    return 1.0 / (1.0 + exp(-z));
}


inline nn_type sigmoidPrime(nn_type z) {
    nn_type exponential = exp(z);
    return exponential / pow(exponential + 1, 2);
}


inline void add_vectors(nn_type *trgt, nn_type *src, const unsigned int len) {
    unsigned int i;
    for (i=0; i<len; i++) {
        trgt[i] += src[i];
    }
}


inline void subtract_vectors(nn_type *trgt, nn_type *src, const unsigned long int len) {
    unsigned long int i;
    for (i=0; i<len; i++) {
        trgt[i] -= src[i];
    }
}


inline void copy_vectors(nn_type *trgt, nn_type *src, const unsigned long int len) {
    unsigned long int i;
    for (i=0; i<len; i++) {
        trgt[i] = src[i];
    }
}
