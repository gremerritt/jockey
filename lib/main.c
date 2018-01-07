#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "batch.h"
#include "constants.h"
#include "file_helpers.h"
#include "helpers.h"
#include "hooks.h"
#include "matrix_helpers.h"
#include "model_helpers.h"
#include "mpi_helper.h"
#include "neural_net.h"
#include "randomizing_helpers.h"
#include "timing_helpers.h"


int main(int argc, char **argv) {
    // Set up local variables
    int i;
    unsigned char err;

    jcky_cli cli;
    jcky_file training_file, testing_file;
    struct meta_neural_net neural_net;
    mpi_manager mpi_manager;

    unsigned short int epoch;
    double total_score, local_score = 0;
    unsigned short int percent_done, last_percent_done = 0;

    unsigned int *sequence;
    nn_type *batch, *result, *targets;
    //-----------------------------------------------------

    remove(JCKY_TIMING_FILENAME);
    mpi_manager = mpi_init(argc, argv);

    err = process_command_line(argc, argv, &cli, mpi_manager.master);
    if (err != 0) goto finalize;
    else if (cli.action == JCKY_ACTION_WRITE) {
        if (mpi_manager.master) write_file();
        goto finalize;
    }
    else if (cli.action == JCKY_ACTION_RUN) {
        training_file = jcky_open_file(cli.training_filename);
        if (training_file.stream == NULL) goto finalize;

        testing_file = jcky_open_file(cli.testing_filename);
        if (testing_file.stream == NULL) {
            jcky_close_file(&training_file);
            goto finalize;
        }
    }

    welcome(&cli, mpi_manager.master);
    mpi_announce(&cli, &mpi_manager);
    MPI_Barrier(MPI_COMM_WORLD);

    neural_net = create_neural_net(
        &cli,
        jcky_get_num_inputs(training_file),
        jcky_get_num_outputs(training_file)
    );

    if (mpi_manager.master) {
        neural_net.functions->init(&neural_net, &cli);
        nn_alloc_cms(&neural_net, mpi_manager.child_procs);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    update_mpi_manager(&neural_net, &mpi_manager, training_file.records, testing_file.records, &cli, &err);
    if (err) goto finalize;
    const unsigned int training_batches = mpi_manager.training_samples.batches;
    const unsigned int testing_batches = mpi_manager.testing_samples.batches;

    // Wait until master has initialized
    MPI_Barrier(MPI_COMM_WORLD);

    jcky_sync_neural_net(&neural_net, &mpi_manager, 0);

    if (mpi_manager.master) {
        printf("\n--------------------------------------\n");
        printf("Configuration:\n");
    	printf("    Total Layers:           %i\n", neural_net.number_of_hidden_layers+2);
    	printf("    Hidden Layers:          %i\n", neural_net.number_of_hidden_layers);
    	printf("    Inputs:                 %i\n", neural_net.number_of_inputs);
    	printf("    Outputs:                %i\n", neural_net.number_of_outputs);
    	printf("    Nodes in Hidden Layers: %i\n", neural_net.number_of_nodes_in_hidden_layers);
    	printf("    Batch Size:             %i\n", neural_net.batch_size);
    	printf("    Learning Rate:          %f\n", neural_net.eta);
        printf("    Initialization Seed:    ");
        if (neural_net.seed != -1) printf("%i\n", neural_net.seed);
        else printf("N/A\n");
        printf("    Initialization File:    %s\n", (neural_net.seed == -1) ? cli.init_model_filename : "N/A");
        printf("    Epochs:                 %i\n", cli.epochs);
    	printf("--------------------------------------\n\n");
    }
    jcky_waitall(&(mpi_manager.neural_net));

	sequence = malloc( mpi_manager.training_samples.total_len * sizeof(unsigned int) );
    batch = malloc(neural_net.number_of_inputs * neural_net.batch_size * sizeof(nn_type));
    targets = malloc(neural_net.number_of_outputs * neural_net.batch_size * sizeof(nn_type));
    result = malloc(neural_net.number_of_outputs * neural_net.batch_size * sizeof(nn_type));

    setbuf(stdout, NULL);
    INIT_TIMERS

	for (epoch=0; epoch<cli.epochs; epoch++) {
        GET_TIMER
        START_TIME_EPOCH

		if (mpi_manager.master) printf("Epoch %i\n", epoch);

        START_TIME_COPY
        neural_net.functions->copy(&neural_net, JCKY_NN_SCRATCH, JCKY_NN_BASE);
        END_TIME_COPY

        START_TIME_SHUFFLE
        if (mpi_manager.master) {
		    for (i=0; i<training_file.records; i++) sequence[i] = i;
		    shuffle(sequence, training_file.records);
        }
        jcky_sync_sequence(sequence, &mpi_manager);
        END_TIME_SHUFFLE

        if (mpi_manager.master) printf("    Training");
		START_TIME_TRAINING
		for (i=0; i<training_batches; i++) {
            START_TIME_TRAINING_BATCH
			create_batch_with_sequence_file(batch, targets, &training_file, neural_net.batch_size, i, sequence);
            END_TIME_TRAINING_BATCH

            START_TIME_TRAINING_RUN
            feed_forward(&neural_net, result, batch, targets, JCKY_TRAIN, &local_score);
            END_TIME_TRAINING_RUN

            if (cli.verbose && mpi_manager.master) {
                percent_done = (unsigned short int)((((i+1)*1.0) / training_batches) * 100);
                if (percent_done > last_percent_done) {
                    printf("\r    Training - ");
                    print_number(percent_done, 3);
                    printf("%%");
                }
            }
		}
        END_TIME_TRAINING
        if (mpi_manager.master) {
            printf("\n");
            last_percent_done = 0;
        }

        START_TIME_SYNC
        neural_net.functions->get_change(&neural_net);
        jcky_sync_changes(&neural_net, &mpi_manager);
        if (mpi_manager.master) {
            neural_net.functions->apply_changes(&neural_net);
        }
        jcky_sync_neural_net(&neural_net, &mpi_manager, 1);
        END_TIME_SYNC

        if (mpi_manager.master && (!cli.no_save || epoch == cli.epochs-1)) {
            write_model(&neural_net, cli.model_filename);
        }

		START_TIME_TESTING
        if (mpi_manager.master) printf("    Testing");
		for (i=0; i<testing_batches; i++) {
            START_TIME_TESTING_BATCH
			create_batch_no_sequence_file(batch, targets, &testing_file, neural_net.batch_size, i, mpi_manager.rank, mpi_manager.testing_samples.base);
            END_TIME_TESTING_BATCH

            START_TIME_TESTING_RUN
            feed_forward(&neural_net, result, batch, targets, JCKY_TEST, &local_score);
            END_TIME_TESTING_RUN

            if (cli.verbose && mpi_manager.master) {
                percent_done = (unsigned short int)((((i+1)*1.0) / testing_batches) * 100);
                if (percent_done > last_percent_done) {
                    printf("\r    Testing  - ");
                    print_number(percent_done, 3);
                    printf("%%");
                }
            }
		}
        END_TIME_TESTING
        if (mpi_manager.master) {
            printf("\n");
            last_percent_done = 0;
        }

        MPI_Reduce(&local_score, &total_score, 1, MPI_DOUBLE, MPI_SUM, JCKY_MASTER, MPI_COMM_WORLD);
        if (mpi_manager.master) printf("    Total Score: %f\n", total_score);
		local_score = 0.0;

        END_TIME_EPOCH
        WRITE_TIME
	}

    WRITE_TIMES

    free(sequence);
    free(batch);
    free(targets);
    free(result);
    FREE_TIMERS
    destroy_mpi_manager(&mpi_manager);
    destroy_meta_nn(&neural_net);
    jcky_close_file(&training_file);
    jcky_close_file(&testing_file);
finalize:
    MPI_Finalize();
	return 0;
}
