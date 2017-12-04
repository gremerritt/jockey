#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "hooks.h"
#include "matrix_helpers.h"
#include "mpi_helper.h"
#include "neural_net.h"
#include "randomizing_helpers.h"


functions contiguous_functions = {
    .alloc = nn_alloc_contiguous,
    .init = nn_init_contiguous,
    .copy = nn_copy_contiguous,
    .get_change = nn_get_change_contiguous,
    .apply_changes = nn_apply_changes_contiguous
};

functions logical_functions = {
    .alloc = nn_alloc_logical,
    .init = nn_init_logical,
    .copy = nn_copy_logical,
    .get_change = nn_get_change_logical,
    .apply_changes = nn_apply_changes_logical
};


// This function sets the various 'hyperparameters' (i.e. learning rate,
// number of layers, nodes per layer, etc.) It also initializes the
// arrays and matrices used in the neural net. All arrays and matrices
// are initialized to a random Gaussian value between -1 and 1 in order
// to give the neural net a place to start.
struct meta_neural_net create_neural_net(
    int number_of_hidden_layers,
    int number_of_nodes_in_hidden_layers,
    int number_of_inputs,
    int number_of_outputs,
    int batch_size,
    nn_type eta,
    unsigned char memory_layout,
    unsigned char num_blocks,
    unsigned int block_size)
{
    struct meta_neural_net nn;
    struct functions *NN_FUNCTIONS[2] = {
        &contiguous_functions,
        &logical_functions
    };

    nn.number_of_hidden_layers = number_of_hidden_layers;
    nn.number_of_nodes_in_hidden_layers = number_of_nodes_in_hidden_layers;
    nn.number_of_inputs = number_of_inputs;
    nn.number_of_outputs = number_of_outputs;
    nn.batch_size = batch_size;
    nn.eta = eta;
    nn.cms_len = 0;
    nn.memory_layout = memory_layout;
    nn.num_blocks = num_blocks;
    nn.block_size = block_size;
    nn.functions = NN_FUNCTIONS[memory_layout];

    meta_nn_alloc(&nn);

    return nn;
}


// This allocates space in memory for the neural net
void meta_nn_alloc(struct meta_neural_net *meta) {
    int i;
    const unsigned int number_of_hidden_layers = meta->number_of_hidden_layers;
    const unsigned int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const unsigned int number_of_inputs = meta->number_of_inputs;
    const unsigned int number_of_outputs = meta->number_of_outputs;
    const unsigned int batch_size = meta->batch_size;
    unsigned int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;

    //---------------------------------------------------------------------------
    // allocate space for bias, z_vector, activation and delta arrays
    meta->z_matrix = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
    meta->activation = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
    meta->delta = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );

    // hidden layers
    for (i=0; i<number_of_hidden_layers; i++) {
        meta->delta[i] = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
        meta->z_matrix[i] = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
        meta->activation[i] = (nn_type *)malloc( number_of_nodes_in_hidden_layers * batch_size * sizeof( nn_type ) );
    }

    // output layer
    meta->delta[number_of_hidden_layers] = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
    meta->z_matrix[number_of_hidden_layers] = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
    meta->activation[number_of_hidden_layers] = (nn_type *)malloc( number_of_outputs * batch_size * sizeof( nn_type ) );
    //---------------------------------------------------------------------------

    meta->functions->alloc(meta, &(meta->nns[JCKY_NN_BASE]));
    meta->functions->alloc(meta, &(meta->nns[JCKY_NN_SCRATCH]));
}


void nn_alloc_logical(struct meta_neural_net *meta, neural_net *nn) {
    int i;
    const unsigned int number_of_hidden_layers = meta->number_of_hidden_layers;
    const unsigned int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const unsigned int number_of_inputs = meta->number_of_inputs;
    const unsigned int number_of_outputs = meta->number_of_outputs;
    unsigned int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
    nn->container_len =
        (number_of_hidden_layers * number_of_nodes_in_hidden_layers) + // bias matrices in hidden layers
        number_of_outputs + // bias vector for output layer
        number_of_nodes_in_hidden_layers * ( // weights
            number_of_inputs + // input layer
            (number_of_hidden_layers * number_of_nodes_in_hidden_layers ) + // hidden layers
            number_of_outputs // output layer
        );

    //---------------------------------------------------------------------------
    // allocate space for bias, z_vector, activation and delta arrays
    nn->bias = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );

    // hidden layers
    for (i=0; i<number_of_hidden_layers; i++) {
        nn->bias[i] = (nn_type *)malloc( number_of_nodes_in_hidden_layers * sizeof( nn_type ) );
    }

    // output layer
    nn->bias[number_of_hidden_layers] = (nn_type *)malloc( number_of_outputs * sizeof( nn_type ) );
    //---------------------------------------------------------------------------

    //---------------------------------------------------------------------------
    // allocate space for weight array
    nn->weight = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );

    // input layer to first hidden layer
    nn->weight[0] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );

    // between hidden layers
    number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
    for (i=1; i<number_of_hidden_layers; i++) {
        nn->weight[i] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
    }

    // last hidden layer to output layer
    number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
    nn->weight[number_of_hidden_layers] = (nn_type *)malloc( number_of_matrix_elements * sizeof( nn_type ) );
    //---------------------------------------------------------------------------
}


void nn_alloc_contiguous(struct meta_neural_net *meta, neural_net *nn) {
    const unsigned long int number_of_hidden_layers = meta->number_of_hidden_layers;
    const unsigned long int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const unsigned long int number_of_inputs = meta->number_of_inputs;
    const unsigned long int number_of_outputs = meta->number_of_outputs;
    unsigned long int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
    unsigned long int offset = 0;
    unsigned short int i;
    const unsigned long int container_len =
        (number_of_hidden_layers * number_of_nodes_in_hidden_layers) + // bias matrices in hidden layers
        number_of_outputs + // bias vector for output layer
        number_of_nodes_in_hidden_layers * ( // weights
            number_of_inputs + // input layer
            (number_of_hidden_layers * number_of_nodes_in_hidden_layers ) + // hidden layers
            number_of_outputs // output layer
        );

    nn->container_len = container_len;
    nn->container = (nn_type*)malloc( container_len * sizeof( nn_type ) );

    nn->bias = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );
    nn->weight = malloc( (number_of_hidden_layers+1) * sizeof( nn_type* ) );

    //--BIAS---------------------------------------------------------------------
    // Hidden Layers
    for (i=0; i<number_of_hidden_layers; i++) {
        nn->bias[i] = nn->container + offset;
        offset += number_of_nodes_in_hidden_layers;
    }

    // Output Layer
    nn->bias[number_of_hidden_layers] = nn->container + offset;
    offset += number_of_outputs;
    //---------------------------------------------------------------------------

    //--WEIGHT-------------------------------------------------------------------
    // Input layer to first hidden layer
    nn->weight[0] = nn->container + offset;
    offset += number_of_matrix_elements;

    // Between hidden layers
    number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
    for (i=1; i<number_of_hidden_layers; i++) {
        nn->weight[i] = nn->container + offset;
        offset += number_of_matrix_elements;
    }

    // Last hidden layer to output layer
    nn->weight[number_of_hidden_layers] = nn->container + offset;
}


void nn_alloc_cms(struct meta_neural_net *meta, const unsigned short int len) {
    unsigned int i;

    meta->cms_len = len;
    meta->cms = malloc( len * sizeof( neural_net ) );
    for (i=0; i<len; i++) {
        meta->functions->alloc(meta, &(meta->cms[i]));
    }
}


// Initialize the bias and weight vectors
void nn_init_logical(struct meta_neural_net *meta, int seed) {
    const unsigned int number_of_hidden_layers = meta->number_of_hidden_layers;
    const unsigned int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const unsigned int number_of_inputs = meta->number_of_inputs;
    const unsigned int number_of_outputs = meta->number_of_outputs;
    unsigned int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;

    int i, j, random_count = 0;
    double *random = malloc( meta->nns[JCKY_NN_BASE].container_len * sizeof(double) );

    meta->seed = generate_guassian_distribution(random, meta->nns[JCKY_NN_BASE].container_len, seed);

    //---------------------------------------------------------------------------
    // Initialize bias vectors

    // Hidden layers
    for (i=0; i<number_of_hidden_layers; i++) {
        for (j=0; j<number_of_nodes_in_hidden_layers; j++) {
            meta->nns[JCKY_NN_BASE].bias[i][j] = random[random_count++];
        }
    }

    // Output layer
    for (i=0; i<number_of_outputs; i++) {
        meta->nns[JCKY_NN_BASE].bias[number_of_hidden_layers][i] = random[random_count++];
    }
    //---------------------------------------------------------------------------

    //---------------------------------------------------------------------------
    // Initialize weight vectors

    // Input layer to first hidden layer
    for (i=0; i<number_of_matrix_elements; i++) {
        meta->nns[JCKY_NN_BASE].weight[0][i] = random[random_count++];
    }

    // Between hidden layers
    number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
    for (i=1; i<number_of_hidden_layers; i++) {
        for (j=0; j<number_of_matrix_elements; j++) {
            meta->nns[JCKY_NN_BASE].weight[i][j] = random[random_count++];
        }
    }

    // Last hidden layer to output layer
    number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
    for (i=0; i<number_of_matrix_elements; i++) {
        meta->nns[JCKY_NN_BASE].weight[number_of_hidden_layers][i] = random[random_count++];
    }
    //---------------------------------------------------------------------------

    free(random);
}


void nn_init_contiguous(struct meta_neural_net *meta, int seed) {
    meta->seed = generate_guassian_distribution(
        meta->nns[JCKY_NN_BASE].container, meta->nns[JCKY_NN_BASE].container_len, seed);
}


void destroy_meta_nn(struct meta_neural_net *meta) {
    unsigned int i;
    const int number_of_hidden_layers = meta->number_of_hidden_layers;
    const unsigned short int cms_len = meta->cms_len;

    for (i=0; i<=number_of_hidden_layers; i++) {
        free(meta->z_matrix[i]);
        free(meta->activation[i]);
        free(meta->delta[i]);
    }
    free (meta->z_matrix);
    free (meta->activation);
    free (meta->delta);

    destroy_nn(meta, &(meta->nns[JCKY_NN_BASE]));
    destroy_nn(meta, &(meta->nns[JCKY_NN_SCRATCH]));
    for (i=0; i<cms_len; i++) {
        destroy_nn(meta, &(meta->cms[i]));
        if (i == cms_len - 1) {
            free(meta->cms);
        }
    }
}


void destroy_nn(struct meta_neural_net *meta, neural_net *nn) {
    unsigned int i;
    const int number_of_hidden_layers = meta->number_of_hidden_layers;

    if (meta->memory_layout == JCKY_CONTIGUOUS_LAYOUT_ID) {
        free(nn->container);
    }
    else if (meta->memory_layout == JCKY_LOGICAL_LAYOUT_ID) {
        for (i=0; i<=number_of_hidden_layers; i++) {
            free(nn->bias[i]);
            free(nn->weight[i]);
        }
    }

    free (nn->bias);
    free (nn->weight);
}


void nn_copy_contiguous(struct meta_neural_net *meta, const unsigned char trgt, const unsigned char src) {
    copy_vectors(meta->nns[trgt].container, meta->nns[src].container, meta->nns[trgt].container_len);
}


void nn_copy_logical(struct meta_neural_net *meta, const unsigned char trgt, const unsigned char src) {
    unsigned int i;
    const int number_of_hidden_layers          = meta->number_of_hidden_layers;
    const int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const int number_of_inputs                 = meta->number_of_inputs;
    const int number_of_outputs                = meta->number_of_outputs;
    int number_of_matrix_elements              = number_of_inputs * number_of_nodes_in_hidden_layers;

    for (i=0; i<number_of_hidden_layers; i++) {
        copy_vectors(meta->nns[trgt].bias[i], meta->nns[src].bias[i], number_of_nodes_in_hidden_layers);
    }
    copy_vectors(meta->nns[trgt].bias[number_of_hidden_layers], meta->nns[src].bias[number_of_hidden_layers], number_of_outputs);
    copy_vectors(meta->nns[trgt].weight[0], meta->nns[src].weight[0], number_of_matrix_elements);

    number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
    for (i=1; i<number_of_hidden_layers; i++) {
        copy_vectors(meta->nns[trgt].weight[i], meta->nns[src].weight[i], number_of_matrix_elements);
    }

    number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
    copy_vectors(meta->nns[trgt].weight[number_of_hidden_layers], meta->nns[src].weight[number_of_hidden_layers], number_of_matrix_elements);
}


void nn_get_change_contiguous(struct meta_neural_net *meta) {
    subtract_vectors(meta->nns[JCKY_NN_SCRATCH].container, meta->nns[JCKY_NN_BASE].container, meta->nns[JCKY_NN_BASE].container_len);
}


void nn_get_change_logical(struct meta_neural_net *meta) {
    unsigned int i;
    const int number_of_hidden_layers          = meta->number_of_hidden_layers;
    const int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const int number_of_inputs                 = meta->number_of_inputs;
    const int number_of_outputs                = meta->number_of_outputs;
    int number_of_matrix_elements              = number_of_inputs * number_of_nodes_in_hidden_layers;

    for (i=0; i<number_of_hidden_layers; i++) {
        subtract_vectors(meta->nns[JCKY_NN_SCRATCH].bias[i], meta->nns[JCKY_NN_BASE].bias[i], number_of_nodes_in_hidden_layers);
    }
    subtract_vectors(meta->nns[JCKY_NN_SCRATCH].bias[number_of_hidden_layers], meta->nns[JCKY_NN_BASE].bias[number_of_hidden_layers], number_of_outputs);
    subtract_vectors(meta->nns[JCKY_NN_SCRATCH].weight[0], meta->nns[JCKY_NN_BASE].weight[0], number_of_matrix_elements);

    number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
    for (i=1; i<number_of_hidden_layers; i++) {
        subtract_vectors(meta->nns[JCKY_NN_SCRATCH].weight[i], meta->nns[JCKY_NN_BASE].weight[i], number_of_matrix_elements);
    }

    number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
    subtract_vectors(meta->nns[JCKY_NN_SCRATCH].weight[number_of_hidden_layers], meta->nns[JCKY_NN_BASE].weight[number_of_hidden_layers], number_of_matrix_elements);
}


void nn_apply_changes_contiguous(struct meta_neural_net *meta) {
    unsigned long int i;
    unsigned short int j;
    nn_type accum;
    const unsigned long int container_len = meta->nns[JCKY_NN_SCRATCH].container_len;
    const unsigned short int change_matrices = meta->cms_len;
    const nn_type divisor = (nn_type)(change_matrices + 1);

    for (i=0; i<container_len; i++) {
        accum = meta->nns[JCKY_NN_SCRATCH].container[i];
        for (j=0; j<change_matrices; j++) {
            accum += meta->cms[j].container[i];
        }
        meta->nns[JCKY_NN_BASE].container[i] += accum / divisor;
    }
}


void nn_apply_changes_logical(struct meta_neural_net *meta) {
    unsigned int i, j, k;
    const int number_of_hidden_layers          = meta->number_of_hidden_layers;
    const int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const int number_of_inputs                 = meta->number_of_inputs;
    const int number_of_outputs                = meta->number_of_outputs;
    const int change_matrices                  = meta->cms_len;
    const nn_type divisor                      = (nn_type)(change_matrices + 1);
    int number_of_matrix_elements              = number_of_inputs * number_of_nodes_in_hidden_layers;
    nn_type accum;

    for (i=0; i<number_of_hidden_layers; i++) {
        for (j=0; j<number_of_nodes_in_hidden_layers; j++) {
            accum = meta->nns[JCKY_NN_SCRATCH].bias[i][j];
            for (k=0; k<change_matrices; k++) {
                accum += meta->cms[k].bias[i][j];
            }
            meta->nns[JCKY_NN_BASE].bias[i][j] += accum / divisor;
        }
    }

    for (i=0; i<number_of_outputs; i++) {
        accum = meta->nns[JCKY_NN_SCRATCH].bias[number_of_hidden_layers][i];
        for (k=0; k<change_matrices; k++) {
            accum += meta->cms[k].bias[number_of_hidden_layers][i];
        }
        meta->nns[JCKY_NN_BASE].bias[number_of_hidden_layers][i] += accum / divisor;
    }

    for (i=0; i<number_of_matrix_elements; i++) {
        accum = meta->nns[JCKY_NN_SCRATCH].weight[0][i];
        for (k=0; k<change_matrices; k++) {
            accum += meta->cms[k].weight[0][i];
        }
        meta->nns[JCKY_NN_BASE].weight[0][i] += accum / divisor;
    }

    number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
    for (i=0; i<number_of_hidden_layers; i++) {
        for (j=0; j<number_of_matrix_elements; j++) {
            accum = meta->nns[JCKY_NN_SCRATCH].weight[i][j];
            for (k=0; k<change_matrices; k++) {
                accum += meta->cms[k].weight[i][j];
            }
            meta->nns[JCKY_NN_BASE].weight[i][j] += accum / divisor;
        }
    }

    number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
    for (i=0; i<number_of_matrix_elements; i++) {
        accum = meta->nns[JCKY_NN_SCRATCH].weight[number_of_hidden_layers][i];
        for (k=0; k<change_matrices; k++) {
            accum += meta->cms[k].weight[number_of_hidden_layers][i];
        }
        meta->nns[JCKY_NN_BASE].weight[number_of_hidden_layers][i] += accum / divisor;
    }
}


void feed_forward(struct meta_neural_net *meta,
                  nn_type *result,
                  nn_type *activation_initial,
                  nn_type *target_values,
                  char training,
                  double *score)
{
  int i, j;
  int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = meta->number_of_inputs;
  int number_of_hidden_layers          = meta->number_of_hidden_layers;
  int number_of_outputs                = meta->number_of_outputs;
  int batch_size                       = meta->batch_size;

  //---------------------------------------------------------------------------
  // feed from input layer -> first hidden layer
  //  do matrix multiply
  //    Dimensions:
  //      Weight Matrix:      'Nodes in target layer' rows X 'Nodes in source layer' columns
  //      Activation Matric:  'Nodes in source layer' rows X 'Batch size' columns
  //  Get the z-matrix
  calculate_z_matrix(meta->z_matrix[0],
                     meta->nns[training].weight[0],
                     activation_initial,
                     meta->nns[training].bias[0],
                     number_of_nodes_in_hidden_layers,
                     number_of_inputs,
                     batch_size);

  //  Compute activation
  sigmoidify(meta->activation[0],
             meta->z_matrix[0],
             number_of_nodes_in_hidden_layers,
             batch_size);
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // feed through the hidden layers
  for(i=1; i<number_of_hidden_layers; i++) {
    //  Get the z-matrix
    calculate_z_matrix(meta->z_matrix[i],
                       meta->nns[training].weight[i],
                       meta->activation[i-1],
                       meta->nns[training].bias[i],
                       number_of_nodes_in_hidden_layers,
                       number_of_nodes_in_hidden_layers,
                       batch_size);
    //  Compute activation
    sigmoidify(meta->activation[i],
               meta->z_matrix[i],
               number_of_nodes_in_hidden_layers,
               batch_size);
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // feed from the last hidden layer -> output layer
  //  Get the z-matrix
  calculate_z_matrix(meta->z_matrix[number_of_hidden_layers],
                     meta->nns[training].weight[number_of_hidden_layers],
                     meta->activation[number_of_hidden_layers-1],
                     meta->nns[training].bias[number_of_hidden_layers],
                     number_of_outputs,
                     number_of_nodes_in_hidden_layers,
                     batch_size);

  //  compute activation
  sigmoidify(meta->activation[number_of_hidden_layers],
             meta->z_matrix[number_of_hidden_layers],
             number_of_outputs,
             batch_size);
  //---------------------------------------------------------------------------

  int num_outputs = number_of_outputs * batch_size;
  for (i=0; i<num_outputs; i++) {
    result[i] = meta->activation[number_of_hidden_layers][i];
    }

  if (training == JCKY_TRAIN) {
        backpropagate(meta, activation_initial, target_values);
    }
  else {
        *score += get_score(batch_size, number_of_outputs, meta->activation[number_of_hidden_layers], target_values);
  }
}

void backpropagate(struct meta_neural_net *meta,
                   nn_type *activation_initial,
                   nn_type *target_values)
{
  int i;
  int number_of_hidden_layers          = meta->number_of_hidden_layers;
  int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
  int number_of_inputs                 = meta->number_of_inputs;
  int number_of_outputs                = meta->number_of_outputs;
  int batch_size                       = meta->batch_size;
  nn_type eta                          = meta->eta;

  // find the delta value in the output layer
  delta_output_layer(meta->delta[number_of_hidden_layers],
                     meta->activation[number_of_hidden_layers],
                     meta->z_matrix[number_of_hidden_layers],
                     target_values,
                     number_of_outputs,
                     batch_size);

  // backpropagate delta -> last hidden layer
  //  Note that row, col dimensions here are for the matrix W
  //  NOT the transpose of W. The transpose will be taken care
  //  of in the function.
  delta_hidden_layers(meta->delta[number_of_hidden_layers-1],
                      meta->nns[JCKY_NN_SCRATCH].weight[number_of_hidden_layers],
                      meta->delta[number_of_hidden_layers],
                      meta->z_matrix[number_of_hidden_layers-1],
                      number_of_outputs,
                      number_of_nodes_in_hidden_layers,
                      batch_size);

  // backpropagate delta -> hidden layers
  for (i=number_of_hidden_layers-2; i>=0; i--) {
    delta_hidden_layers(meta->delta[i],
                        meta->nns[JCKY_NN_SCRATCH].weight[i+1],
                        meta->delta[i+1],
                        meta->z_matrix[i],
                        number_of_nodes_in_hidden_layers,
                        number_of_nodes_in_hidden_layers,
                        batch_size);
  }

  // -----------------------------------------------------------------
  // now that we have all of our deltas, adjust the weights and biases
  //  adjust the first hidden layer
  adjust_weight(activation_initial,
                meta->nns[JCKY_NN_SCRATCH].weight[0],
                meta->delta[0],
                number_of_nodes_in_hidden_layers,
                number_of_inputs,
                batch_size,
                eta);

  adjust_bias(meta->nns[JCKY_NN_SCRATCH].bias[0],
              meta->delta[0],
              number_of_nodes_in_hidden_layers,
              batch_size,
              eta);
  //
  //  adjust the hidden layers
  for (i=1; i<number_of_hidden_layers; i++) {
    adjust_weight(meta->activation[i-1],
                  meta->nns[JCKY_NN_SCRATCH].weight[i],
                  meta->delta[i],
                  number_of_nodes_in_hidden_layers,
                  number_of_nodes_in_hidden_layers,
                  batch_size,
                  eta);
    adjust_bias(meta->nns[JCKY_NN_SCRATCH].bias[i],
                meta->delta[i],
                number_of_nodes_in_hidden_layers,
                batch_size,
                eta);
  }
  //
  //  adjust the output hidden layer
  adjust_weight(meta->activation[number_of_hidden_layers-1],
                meta->nns[JCKY_NN_SCRATCH].weight[number_of_hidden_layers],
                meta->delta[number_of_hidden_layers],
                number_of_outputs,
                number_of_nodes_in_hidden_layers,
                batch_size,
                eta);
  adjust_bias(meta->nns[JCKY_NN_SCRATCH].bias[number_of_hidden_layers],
              meta->delta[number_of_hidden_layers],
              number_of_outputs,
              batch_size,
              eta);
  // -----------------------------------------------------------------
}
