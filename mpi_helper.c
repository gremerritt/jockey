#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "constants.h"
#include "helpers.h"
#include "mpi_helper.h"


void (*JCKY_SEND_NN_ASYNC_FUNCS[2])(struct meta_neural_net *, neural_net *, int, mpi_manager *) = {
    jcky_send_nn_async_contiguous,
    jcky_send_nn_async_logical
};


void (*JCKY_RECV_NN_ASYNC_FUNCS[2])(struct meta_neural_net *, neural_net *, int, mpi_manager *) = {
    jcky_recv_nn_async_contiguous,
    jcky_recv_nn_async_logical
};


// TODO: this should handle errors
mpi_manager mpi_init(int argc, char **argv) {
    mpi_manager manager;
    int name_len, world_size, rank, thread_support;

    // Initialize MPI
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &thread_support);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    manager.world_size = (unsigned short int)world_size;

    // Get the rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    manager.rank = (unsigned short int)rank;
    manager.master = !rank;
    manager.child_procs = manager.world_size - 1;

    // Get the name of the processor
    MPI_Get_processor_name(manager.processor_name, &name_len);

    return manager;
}


void mpi_announce(jcky_cli *cli, mpi_manager *manager) {
    if (cli->verbose) {
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Reporting from processor %s, rank %u of %u\n",
            manager->processor_name, manager->rank, manager->world_size);
    }
}


// TODO:
//   - Make sure types are correct for world size and rank
//   - Use destructor for this
//   - Use a data manager
void update_mpi_manager(struct meta_neural_net *meta, mpi_manager *manager,
                        unsigned int training_samples, unsigned int testing_samples,
                        jcky_cli *cli, unsigned char *err) {
    *err = 0;
    const unsigned char memory_layout = meta->memory_layout;
    const unsigned short int child_procs_or_one = (manager->rank == 0) ? manager->child_procs : 1;
    unsigned short int nn_number_of_requests =
        ((meta->memory_layout == (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID ) ? 1 : ((meta->number_of_hidden_layers + 1) * 2))
        * child_procs_or_one;

    if (memory_layout == (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID) {
        const unsigned long int container_len = meta->nns[JCKY_NN_BASE].container_len;
        unsigned long int elements_per_request = container_len;
        if (meta->num_blocks) {
            if (container_len % meta->num_blocks == 0) {
                elements_per_request = container_len / meta->num_blocks;
            }
            else {
                elements_per_request = (container_len / meta->num_blocks) + 1;
            }
        }
        else if (meta->block_size) {
            elements_per_request = meta->block_size / sizeof(nn_type);
        }

        manager->elements_per_request = (int)((elements_per_request > INT_MAX) ? INT_MAX : elements_per_request);
        manager->requests_per_transaction = (container_len % manager->elements_per_request == 0) ?
                                            (container_len / manager->elements_per_request) :
                                            (container_len / manager->elements_per_request) + 1;
        manager->elements_in_last_request = (int)(container_len - (manager->elements_per_request * (manager->requests_per_transaction - 1)));
        nn_number_of_requests *= manager->requests_per_transaction;
        // printf("Elements per request: %u\n", manager->elements_per_request);
        // printf("Requests per transaction: %u\n", manager->requests_per_transaction);
        // printf("Elements in last request: %u\n", manager->elements_in_last_request);
        // printf("Number of requests: %u\n", nn_number_of_requests);
    }

    manager->neural_net = create_request_manager(nn_number_of_requests);
    manager->sequence = create_request_manager(child_procs_or_one);

    if (cli->verbose && manager->master) printf("\nCreating sample managers:\n");
    manager->training_samples = create_sample_manager(
        training_samples, meta->batch_size, manager, TRAINING_DATA, cli, err);
    manager->testing_samples = create_sample_manager(
        testing_samples, meta->batch_size, manager, TESTING_DATA, cli, err);

    manager->send_nn_async_func = JCKY_SEND_NN_ASYNC_FUNCS[memory_layout];
    manager->recv_nn_async_func = JCKY_RECV_NN_ASYNC_FUNCS[memory_layout];
}


request_manager create_request_manager(unsigned short int number_of_requests) {
    request_manager request_manager;

    request_manager.request_num = 0;
    request_manager.number_of_requests = number_of_requests;
    request_manager.request = (MPI_Request*)malloc(sizeof(MPI_Request) * number_of_requests);
    request_manager.status = (MPI_Status*)malloc(sizeof(MPI_Status) * number_of_requests);

    return request_manager;
}


// Each process will have a multiple of the batch size.
// If we have more processes than are useful just error out
// because it's a bad configuration.
sample_manager create_sample_manager(unsigned int samples, unsigned short int batch_size,
                                     mpi_manager *manager, char type_code,
                                     jcky_cli *cli, unsigned char *err) {
    sample_manager sample_manager;
    char type[9];
    unsigned short int world_size = manager->world_size;

    if (type_code == TRAINING_DATA) strcpy(type, "training");
    else if (type_code == TESTING_DATA) strcpy(type, "testing");
    else {
        *err = 1;
        if (manager->master) printf("    Error: Invalid sample type.\n");
    }

    if (!(*err)) {
        if (world_size > round_up_multiple(samples, batch_size) / batch_size) {
            *err = 1;
            if (manager->master) printf("    Error: Too many processes, too few %s samples, or too large of a batch size.\n", type);
        }
        else {
            sample_manager.base = round_up_multiple(samples / world_size, batch_size);
            sample_manager.procn = samples - ((world_size - 1) * sample_manager.base);
            sample_manager.local = (manager->rank < world_size - 1) ? sample_manager.base : sample_manager.procn;

            // Process 0 -> N-2 will have the same number of batches.
            // Process N-1 may have a slightly different number of batchs.
            sample_manager.batches = sample_manager.local / batch_size;
            sample_manager.extra = sample_manager.local % batch_size;

            // Currently, the data set needs to be square with the batch size.
            // We'll chop off extra samples to enforce this. We may revisit this,
            // but it screws up the neural net to have a differently sized batch.
            // So just throw out a couple samples to make it work nicely.
            sample_manager.procn = (sample_manager.procn / batch_size) * batch_size;
            sample_manager.local = (manager->rank < world_size - 1) ? sample_manager.base : sample_manager.procn;
            sample_manager.total_len = manager->master ? ((world_size - 1) * sample_manager.base) + sample_manager.procn : sample_manager.local;

            if (cli->verbose) {
                if (manager->master) printf("    Handling %u total %s samples.\n", sample_manager.total_len, type);

                printf("    Process %u will handle %u %s samples (%u batches)\n",
                    manager->rank, sample_manager.local, type, sample_manager.batches);
                if (sample_manager.extra > 0) {
                    printf("    WARNING: Ignoring last %u %s samples. To avoid this, make sure your %s "
                           "sample size is evenly divisible by your batch size.\n",
                            sample_manager.extra, type, type);
                }
            }
        }
    }

    return sample_manager;
}


void destroy_mpi_manager(mpi_manager *manager) {
    destroy_request_manager(&(manager->neural_net));
    destroy_request_manager(&(manager->sequence));
}


void destroy_request_manager(request_manager *request_manager) {
    free(request_manager->request);
    free(request_manager->status);
}


void jcky_waitall(request_manager *request_manager) {
    MPI_Waitall(request_manager->number_of_requests, request_manager->request, request_manager->status);
    request_manager->request_num = 0;
}


// TODO:
//  - handle errors
void jcky_sync_neural_net(struct meta_neural_net *meta, mpi_manager *manager, const char waitall) {
    int i;
    const unsigned short int child_processes = manager->child_procs;

    if (manager->master) {
        for (i=1; i<=child_processes; i++) {
            manager->send_nn_async_func(meta, &(meta->nns[JCKY_NN_BASE]), i, manager);
        }
    }
    else {
        manager->recv_nn_async_func(meta, &(meta->nns[JCKY_NN_BASE]), 0, manager);
    }

    if (waitall) {
        jcky_waitall(&(manager->neural_net));
    }
}


// TODO:
//  - handle errors
void jcky_sync_changes(struct meta_neural_net *meta, mpi_manager *manager) {
    unsigned short int i;
    const unsigned short int child_procs = manager->child_procs;

    if (manager->master) {
        for (i=0; i<child_procs; i++) {
            manager->recv_nn_async_func(meta, &(meta->cms[i]), i+1, manager);
        }
    }
    else {
        manager->send_nn_async_func(meta, &(meta->nns[JCKY_NN_SCRATCH]), JCKY_MASTER, manager);
    }
    jcky_waitall(&(manager->neural_net));
}


void jcky_send_nn_async_contiguous(struct meta_neural_net *meta, neural_net *nn, int dest, mpi_manager *manager) {
    unsigned short int *request_num = &(manager->neural_net.request_num);
    MPI_Request *request = manager->neural_net.request;
    const unsigned short int requests = manager->requests_per_transaction - 1;
    const int elements_per_request = manager->elements_per_request;
    unsigned short int i;
    nn_type *container = nn->container;

    for(i=0; i<requests; i++) {
        MPI_Isend(container, elements_per_request, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, request + (*request_num)++);
        container += elements_per_request;
    }
    MPI_Isend(container, manager->elements_in_last_request, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, request + (*request_num)++);
}


void jcky_recv_nn_async_contiguous(struct meta_neural_net *meta, neural_net *nn, int source, mpi_manager *manager) {
    unsigned short int *request_num = &(manager->neural_net.request_num);
    MPI_Request *request = manager->neural_net.request;
    const unsigned short int requests = manager->requests_per_transaction - 1;
    const int elements_per_request = manager->elements_per_request;
    unsigned short int i;
    nn_type *container = nn->container;

    for(i=0; i<requests; i++) {
        MPI_Irecv(container, elements_per_request, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, request + (*request_num)++);
        container += elements_per_request;
    }
    MPI_Irecv(container, manager->elements_in_last_request, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, request + (*request_num)++);
}


// TODO:
//   - handle errors
//   - variable types
void jcky_send_nn_async_logical(struct meta_neural_net *meta, neural_net *nn, int dest, mpi_manager *manager) {
    unsigned short int *request_num = &((*manager).neural_net.request_num);
    MPI_Request *request = (*manager).neural_net.request;
    const unsigned int number_of_hidden_layers = meta->number_of_hidden_layers;
    const unsigned int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const unsigned int number_of_inputs = meta->number_of_inputs;
    const unsigned int number_of_outputs = meta->number_of_outputs;
    unsigned int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
    unsigned int i;

    //---------------------------------------------------------------------------
    // Send each bias vector
    //
    // Each hidden layer
    for (i=0; i<number_of_hidden_layers; i++) {
        MPI_Isend(nn->bias[i], number_of_nodes_in_hidden_layers, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, request + (*request_num)++);
    }

    // Output layer
    MPI_Isend(nn->bias[number_of_hidden_layers], number_of_outputs, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, request + (*request_num)++);
    //---------------------------------------------------------------------------

    //---------------------------------------------------------------------------
    // Send each weight vectors
    //
    // Input layer to first hidden layer
    MPI_Isend(nn->weight[0], number_of_matrix_elements, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, request + (*request_num)++);

    // Between hidden layers
    number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
    for (i=1; i<number_of_hidden_layers; i++) {
        MPI_Isend(nn->weight[i], number_of_matrix_elements, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, request + (*request_num)++);
    }

    // Last hidden layer to output layer
    number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
    MPI_Isend(nn->weight[number_of_hidden_layers], number_of_matrix_elements, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, request + (*request_num)++);
    //---------------------------------------------------------------------------
}


// TODO:
//   - handle errors
//   - variable types
void jcky_recv_nn_async_logical(struct meta_neural_net *meta, neural_net *nn, int source, mpi_manager *manager) {
    unsigned short int *request_num = &((*manager).neural_net.request_num);
    MPI_Request *request = (*manager).neural_net.request;
    const unsigned int number_of_hidden_layers = meta->number_of_hidden_layers;
    const unsigned int number_of_nodes_in_hidden_layers = meta->number_of_nodes_in_hidden_layers;
    const unsigned int number_of_inputs = meta->number_of_inputs;
    const unsigned int number_of_outputs = meta->number_of_outputs;
    unsigned int number_of_matrix_elements = number_of_inputs * number_of_nodes_in_hidden_layers;
    unsigned int i;

    //---------------------------------------------------------------------------
    // Send each bias vector
    //
    // Each hidden layer
    for (i=0; i<number_of_hidden_layers; i++) {
        MPI_Irecv(nn->bias[i], number_of_nodes_in_hidden_layers, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, request + (*request_num)++);
    }

    // Output layer
    MPI_Irecv(nn->bias[number_of_hidden_layers], number_of_outputs, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, request + (*request_num)++);
    //---------------------------------------------------------------------------

    //---------------------------------------------------------------------------
    // Send each weight vectors
    //
    // Input layer to first hidden layer
    MPI_Irecv(nn->weight[0], number_of_matrix_elements, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, request + (*request_num)++);

    // Between hidden layers
    number_of_matrix_elements = number_of_nodes_in_hidden_layers * number_of_nodes_in_hidden_layers;
    for (i=1; i<number_of_hidden_layers; i++) {
        MPI_Irecv(nn->weight[i], number_of_matrix_elements, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, request + (*request_num)++);
    }

    // Last hidden layer to output layer
    number_of_matrix_elements = number_of_outputs * number_of_nodes_in_hidden_layers;
    MPI_Irecv(nn->weight[number_of_hidden_layers], number_of_matrix_elements, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, request + (*request_num)++);
    //---------------------------------------------------------------------------
}


// TODO:
//  - handle errors
void jcky_sync_sequence(unsigned int *sequence, mpi_manager *manager) {
    MPI_Barrier(MPI_COMM_WORLD);

    if ((*manager).master) {
        jcky_send_sequence(sequence, manager);
    }
    else {
        jcky_recv_sequence(sequence, manager);
    }

    jcky_waitall(&((*manager).sequence));
}


// TODO:
//  - handle errors
void jcky_send_sequence(unsigned int *sequence, mpi_manager *manager) {
    unsigned short int *request_num = &((*manager).sequence.request_num);
    MPI_Request *request = (*manager).sequence.request;
    unsigned short int sends = (*manager).sequence.number_of_requests;
    unsigned int base = (*manager).training_samples.base;

    unsigned short int i;
    for (i=1; i<sends; i++) {
        MPI_Isend(sequence + (base * i), base, MPI_UNSIGNED, i, 1, MPI_COMM_WORLD, request + (*request_num)++);
    }
    MPI_Isend(sequence + (base * sends), (*manager).training_samples.procn, MPI_UNSIGNED, sends, 1, MPI_COMM_WORLD, request + (*request_num)++);
}


// TODO:
//  - handle errors
void jcky_recv_sequence(unsigned int *sequence, mpi_manager *manager) {
    unsigned short int *request_num = &((*manager).sequence.request_num);
    MPI_Request *request = (*manager).sequence.request;

    MPI_Irecv(sequence, (*manager).training_samples.local, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, request + (*request_num)++);
}
