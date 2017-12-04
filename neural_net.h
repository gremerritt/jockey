#ifndef NEURALNET_H
#define NEURALNET_H

#include "math.h"

typedef double nn_type;


// This base object defines the neural_net and contains
// all of the bias and weight information. It is managed
// by the neural_net object.
typedef struct neural_net {
    // The container holds all of the neural net data.
    // Each entry in the bias and weight arrays simply
    // point to an address in the container. Thus to send
    // a neural net over the wire, we only need to ship
    // the container.
    nn_type *container;
    unsigned long int container_len;

    // Each entry in 'biases' is a pointer to an array -
    // one for each hidden layer and one for the output layer.
    // Note that the input layer doesn't have any biases.
    nn_type **bias;

    // Each entry in 'weights' is a pointer to a 'matrix' (really
    // an array that we'll treat as a matrix). Each matrix
    // represents the weights connecting two layers of neurons.
    nn_type **weight;

    // The manger manages MPI calls for the neural net.
    struct request_manager *manager;
} neural_net;

struct meta_neural_net {
    int number_of_hidden_layers;
    int number_of_nodes_in_hidden_layers;
    int number_of_inputs;
    int number_of_outputs;
    int batch_size;
    nn_type eta;   // eta is the learning rate
    int seed;
    neural_net nns[2];
    unsigned short int cms_len;
    neural_net *cms;

    unsigned char memory_layout;
    unsigned char num_blocks;
    unsigned int block_size;
    struct functions *functions;

    // Each entry in 'z_vector' is a pointer to an array of the
    // z-values for the corresponding layer in the neural net.
    // The z-value is:
    //    weights * activation(previous level) + biases
    // This value is passed through the sigmoid function to get
    // the layer's activation.
    nn_type **z_matrix;

    // Each entry in 'activation' is a pointer to an array of
    // the activations for the layer. The activation is the
    // z-vector passed through the sigmoid function.
    nn_type **activation;

    // Each entry in 'delta' is a pointer to an array of the
    // deltas for the layer. In the output layer delta has the
    //    Grad(Cost) . sigmoidPrime(z)
    //      = (Activation - y) . sigmoidPrime(z)
    // where the "." is the Hadamard product (element-wise
    // multiplication) and y is the expected output of the
    // neural net for a given input.
    nn_type **delta;
};

typedef struct functions {
    void (*alloc)(struct meta_neural_net *, neural_net *);
    void (*init)(struct meta_neural_net *, int);
    void (*copy)(struct meta_neural_net *, const unsigned char, const unsigned char);
    void (*get_change)(struct meta_neural_net *);
    void (*apply_changes)(struct meta_neural_net *);
} functions;

void meta_nn_alloc(struct meta_neural_net *meta);
void nn_alloc_cms(struct meta_neural_net *meta, const unsigned short int num);
void nn_init_contiguous(struct meta_neural_net *meta, int seed);
void nn_init_logical(struct meta_neural_net *meta, int seed);
void nn_copy_contiguous(struct meta_neural_net *meta, const unsigned char trgt, const unsigned char src);
void nn_copy_logical(struct meta_neural_net *meta, const unsigned char trgt, const unsigned char src);
void destroy_meta_nn(struct meta_neural_net *meta);
void destroy_nn(struct meta_neural_net *meta, neural_net *nn);
void nn_get_change_contiguous(struct meta_neural_net *meta);
void nn_get_change_logical(struct meta_neural_net *meta);
void nn_apply_changes_contiguous(struct meta_neural_net *meta);
void nn_apply_changes_logical(struct meta_neural_net *meta);

//void nn_alloc_optimized(struct meta_neural_net *meta, neural_net *nn);
void nn_alloc_contiguous(struct meta_neural_net *meta, neural_net *nn);
void nn_alloc_logical(struct meta_neural_net *meta, neural_net *nn);

struct meta_neural_net create_neural_net(
    int number_of_hidden_layers,
    int number_of_nodes_in_hidden_layers,
    int number_of_inputs,
    int number_of_outputs,
    int batch_size,
    nn_type eta,
    unsigned char memory_layout,
    unsigned char num_blocks,
    unsigned int block_size
);

void feed_forward(
    struct meta_neural_net *meta,
    nn_type *result,
    nn_type *activation_initial,
    nn_type *target_values,
    char training,
    double *score
);

void backpropagate(
    struct meta_neural_net *meta,
    nn_type *activation_initial,
    nn_type *target_values
);

#endif
