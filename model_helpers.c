#include <stdio.h>

#include "constants.h"
#include "neural_net.h"


void write_model(struct meta_neural_net *meta, char *filename) {
    unsigned int i;
    unsigned int number_of_matrix_elements;

    FILE *stream = fopen(filename, "w+");
    if (stream != NULL) {
        if (meta->memory_layout == JCKY_CONTIGUOUS_LAYOUT_ID) {
            fwrite(
                meta->nns[JCKY_NN_BASE].container,
                sizeof(nn_type),
                meta->nns[JCKY_NN_BASE].container_len,
                stream
            );
        }
        else if (meta->memory_layout == JCKY_LOGICAL_LAYOUT_ID) {
            // -- Biases ----------------------------------------------------
            // Hidden layers
            int tmp_pos = 0;
            for (i=0; i<meta->number_of_hidden_layers; i++) {
                fwrite(
                    meta->nns[JCKY_NN_BASE].bias[i],
                    sizeof(nn_type),
                    meta->number_of_nodes_in_hidden_layers,
                    stream
                );
            }

            // Output layer
            fwrite(
                meta->nns[JCKY_NN_BASE].bias[meta->number_of_hidden_layers],
                sizeof(nn_type),
                meta->number_of_outputs,
                stream
            );
            //--------------------------------------------------------------

            // -- Weights --------------------------------------------------
            // Input layer to first hidden layer
            fwrite(
                meta->nns[JCKY_NN_BASE].weight[0],
                sizeof(nn_type),
                meta->number_of_inputs * meta->number_of_nodes_in_hidden_layers,
                stream
            );

            // Between hidden layers
            number_of_matrix_elements = meta->number_of_nodes_in_hidden_layers * meta->number_of_nodes_in_hidden_layers;
            for (i=1; i<meta->number_of_hidden_layers; i++) {
                fwrite(
                    meta->nns[JCKY_NN_BASE].weight[i],
                    sizeof(nn_type),
                    number_of_matrix_elements,
                    stream
                );
            }

            // Last hidden layer to output layer
            fwrite(
                meta->nns[JCKY_NN_BASE].weight[meta->number_of_hidden_layers],
                sizeof(nn_type),
                meta->number_of_outputs * meta->number_of_nodes_in_hidden_layers,
                stream
            );
            //--------------------------------------------------------------
        }
        fclose(stream);
    }
    else {
        printf(KYEL "WARNING: Unabled to write model file %s\n." KNRM, filename);
    }
}

char read_model_bulk(struct meta_neural_net *meta, char *filename) {
    char err = 1;

    FILE *stream = fopen(filename, "rb");
    if (stream != NULL) {
        fread(
            meta->nns[JCKY_NN_BASE].container,
            sizeof(nn_type),
            meta->nns[JCKY_NN_BASE].container_len,
            stream
        );
        err = 0;
        fclose(stream);
    }
    else {
        printf(KYEL "\nWARNING: Unabled to read model file %s\n. " KNRM, filename);
    }

    return err;
}

FILE * open_model_file(char *filename) {
    FILE *stream = fopen(filename, "rb");
    if (stream == NULL) {
        printf(KYEL "\nWARNING: Unabled to read model file %s\n. " KNRM, filename);
    }

    return stream;
}

void read_model_file(nn_type *dest, unsigned long int len, FILE *model_file) {
    fread(dest, sizeof(nn_type), len, model_file);
}

char validate_model_file(struct meta_neural_net *meta, char *filename) {
    char err = 1;
    unsigned long int file_length, expected_length;

    FILE *stream = fopen(filename, "rb");
    if (stream != NULL) {
        fseek(stream, 0, SEEK_END);
        file_length = (unsigned long int)ftell(stream);
        expected_length = container_length(meta) * sizeof(nn_type);

        if (file_length == expected_length) err = 0;
        else printf(KYEL "\nWARNING: The architecture of the neural network in the %s model "\
                    "file does not match the current neural network architecture. " KNRM, filename);
    }
    else {
        printf(KYEL "\nWARNGING: Unabled to read model file %s. " KNRM, filename);
    }

    return err;
}
