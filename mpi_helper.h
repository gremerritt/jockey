#ifndef MPIHELPER_H
#define MPIHELPER_H

#include <stdlib.h>
#include <mpi.h>
#include "neural_net.h"

#define TRAINING_DATA 0
#define TESTING_DATA 1

typedef struct request_manager {
    unsigned short int number_of_requests;
    unsigned short int request_num;
    MPI_Request *request;
    MPI_Status *status;
} request_manager;

typedef struct sample_manager {
    unsigned int base;
    unsigned int procn;
    unsigned int local;
    unsigned int batches;
    unsigned int extra;
    unsigned int total_len;
} sample_manager;

typedef struct mpi_manager {
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    unsigned short int world_size;
    unsigned short int rank;
    unsigned char master;
    unsigned short int child_procs;
    int elements_per_request;
    int elements_in_last_request;
    unsigned short int requests_per_transaction;

    void (*send_nn_async_func)(struct meta_neural_net *, neural_net *, int, struct mpi_manager *);
    void (*recv_nn_async_func)(struct meta_neural_net *, neural_net *, int, struct mpi_manager *);

    request_manager neural_net;
    request_manager sequence;
    sample_manager training_samples;
    sample_manager testing_samples;
} mpi_manager;

mpi_manager mpi_init(int argc, char **argv);
void mpi_announce(mpi_manager *manager);
void update_mpi_manager(struct meta_neural_net *nn, mpi_manager *manager,
                        unsigned int training_samples, unsigned int testing_samples, char *err);
request_manager create_request_manager(unsigned short int number_of_requests);
sample_manager create_sample_manager(unsigned int samples, unsigned short int batch_size,
                                     mpi_manager *manager, char type_code, char *err);

void destroy_mpi_manager(mpi_manager *manager);
void destroy_request_manager(request_manager *request_manager);

void jcky_waitall(request_manager *request_manager);

void jcky_sync_neural_net(struct meta_neural_net *meta, mpi_manager *manager, const char waitall);
void jcky_sync_changes(struct meta_neural_net *meta, mpi_manager *manager);
void jcky_send_nn_async_contiguous(struct meta_neural_net *meta, neural_net *nn, int dest, mpi_manager *manager);
void jcky_recv_nn_async_contiguous(struct meta_neural_net *meta, neural_net *nn, int source, mpi_manager *manager);
void jcky_send_nn_async_logical(struct meta_neural_net *meta, neural_net *nn, int dest, mpi_manager *manager);
void jcky_recv_nn_async_logical(struct meta_neural_net *meta, neural_net *nn, int source, mpi_manager *manager);

void jcky_sync_sequence(unsigned int *sequence, mpi_manager *manager);
void jcky_send_sequence(unsigned int *sequence, mpi_manager *manager);
void jcky_recv_sequence(unsigned int *sequence, mpi_manager *manager);

#endif
