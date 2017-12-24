#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "batch.h"
#include "constants.h"
#include "file_helpers.h"
#include "helpers.h"
#include "hooks.h"
#include "matrix_helpers.h"
#include "mpi_helper.h"
#include "neural_net.h"
#include "randomizing_helpers.h"

#define TRAINING_SAMPLES 60000
#define TEST_SAMPLES 10000
#define TEST_PRINT_RESULTS_EVERY 100000

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"


int main(int argc, char **argv) {
    // int number_of_hidden_layers = NUM_HIDDEN_LAYERS;
    // int number_of_nodes_in_hidden_layers = NUM_NODES_IN_HIDDEN_LAYERS;
    // int batch_size = BATCH_SIZE;
    // nn_type learning_rate = LEARNING_RATE;
    // int seed = -1;
    // unsigned char memory_layout = (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID;
    // unsigned char num_blocks = 0;
    // unsigned int block_size = 0;
    // unsigned char action = JCKY_ACTION_RUN;
    // char training_filename[128] = "\0";
    // char testing_filename[128] = "\0";
    // jcky_file training_file;
    // jcky_file testing_file;

    // int number_of_hidden_layers, number_of_nodes_in_hidden_layers, batch_size, seed;
    // nn_type learning_rate;
    // unsigned char memory_layout, num_blocks, action;
    // unsigned int block_size;
    // unsigned short int epochs;
    // char training_filename[128], testing_filename[128];
    jcky_file training_file, testing_file;

    jcky_cli cli;
    unsigned char pcl = process_command_line(argc, argv, &cli);
    if (pcl != 0) {
        return -1;
    }
    else if (cli.action == JCKY_ACTION_WRITE) {
        return write_file();
    }
    else if (cli.action == JCKY_ACTION_RUN) {
        training_file = jcky_open_file(cli.training_filename);
        if (training_file.stream == NULL) return -1;

        testing_file = jcky_open_file(cli.testing_filename);
        if (testing_file.stream == NULL) {
            jcky_close_file(training_file);
            return -1;
        }
    }

    mpi_manager mpi_manager = mpi_init(argc, argv);
    welcome(mpi_manager.master);
    mpi_announce(&mpi_manager);
    const unsigned short int world_size = mpi_manager.world_size;

	double epoch_t1, epoch_t2, epoch_duration;
	double training_t1, training_t2, training_duration;
	double syncing_t1, syncing_t2, syncing_duration;
	double testing_t1, testing_t2, testing_duration;
	// int counts[EPOCHS];
	// double epoch_times[EPOCHS];
	// double training_times[EPOCHS];
	// double syncing_times[EPOCHS];
	// double testing_times[EPOCHS];
    unsigned short int k;

    MPI_Barrier(MPI_COMM_WORLD);

    struct meta_neural_net neural_net = create_neural_net(
        &cli,
        jcky_get_num_inputs(training_file),
        jcky_get_num_outputs(training_file)
    );

    if (mpi_manager.master) {
        neural_net.functions->init(&neural_net, cli.seed);
        nn_alloc_cms(&neural_net, mpi_manager.child_procs);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    char err;
    update_mpi_manager(&neural_net, &mpi_manager, TRAINING_SAMPLES, TEST_SAMPLES, &err);
    if (err) {
        MPI_Finalize();
        return 0;
    }

    // Wait until master has initialized
    MPI_Barrier(MPI_COMM_WORLD);

    jcky_sync_neural_net(&neural_net, &mpi_manager, 0);

    if (mpi_manager.master) {
    	printf("\n  Total Layers:           %i", neural_net.number_of_hidden_layers+2);
    	printf("\n  Hidden Layers:          %i", neural_net.number_of_hidden_layers);
    	printf("\n  Inputs:                 %i", neural_net.number_of_inputs);
    	printf("\n  Outputs:                %i", neural_net.number_of_outputs);
    	printf("\n  Nodes in Hidden Layers: %i", neural_net.number_of_nodes_in_hidden_layers);
    	printf("\n  Batch Size:             %i", neural_net.batch_size);
    	printf("\n  Learning Rate:          %f", neural_net.eta);
        printf("\n  Initialization Seed:    %i", neural_net.seed);
        printf("\n  Epochs:                 %i", cli.epochs);
    	printf("\n------------------\n");
    }

    jcky_waitall(&(mpi_manager.neural_net));

	double local_score = 0;
    double total_score;
	int epoch;
	int i;
	unsigned int *sequence = malloc( mpi_manager.training_samples.total_len * sizeof(unsigned int) );

	if (mpi_manager.master) printf("Training...\n");
    const unsigned int training_batches = mpi_manager.training_samples.batches;
    const unsigned int testing_batches = mpi_manager.testing_samples.batches;
    const unsigned short int child_procs = mpi_manager.child_procs;
	for (epoch=0; epoch<cli.epochs; epoch++) {
		if (mpi_manager.master) printf("  Epoch %i\n", epoch);
		//epoch_t1 = omp_get_wtime();
        neural_net.functions->copy(&neural_net, JCKY_NN_SCRATCH, JCKY_NN_BASE);

        if (mpi_manager.master) {
		    for (i=0; i<TRAINING_SAMPLES; i++) sequence[i] = i;
		    shuffle(sequence, TRAINING_SAMPLES);
        }

        jcky_sync_sequence(sequence, &mpi_manager);

		//training_t1 = omp_get_wtime();
		for (i=0; i<training_batches; i++) {
			nn_type result[neural_net.number_of_outputs * neural_net.batch_size];
			nn_type batch[neural_net.number_of_inputs * neural_net.batch_size];
			nn_type targets[neural_net.number_of_outputs * neural_net.batch_size];

			create_batch_with_sequence_file(batch, targets, &training_file, neural_net.batch_size, i, sequence);
            feed_forward(&neural_net, result, batch, targets, JCKY_TRAIN, &local_score);
		}
		//training_t2 = omp_get_wtime();
		//training_duration = (training_t2 - training_t1);

		// determine weight change matrix
		//syncing_t1 = omp_get_wtime();
        neural_net.functions->get_change(&neural_net);
        jcky_sync_changes(&neural_net, &mpi_manager);
        if (mpi_manager.master) {
            neural_net.functions->apply_changes(&neural_net);
        }
        jcky_sync_neural_net(&neural_net, &mpi_manager, 1);

		// //syncing_t2 = omp_get_wtime();
		// //syncing_duration = (syncing_t2 - syncing_t1);
		// //testing_t1 = omp_get_wtime();
		for (i=0; i<testing_batches; i++) {
			nn_type result[neural_net.number_of_outputs * neural_net.batch_size];
			nn_type batch[neural_net.number_of_inputs * neural_net.batch_size];
			nn_type targets[neural_net.number_of_outputs * neural_net.batch_size];
			char correct[neural_net.batch_size];

			create_batch_no_sequence_file(batch, targets, &testing_file, neural_net.batch_size, i, mpi_manager.rank, mpi_manager.testing_samples.base);
            feed_forward(&neural_net, result, batch, targets, JCKY_TEST, &local_score);
		}
		//testing_t2 = omp_get_wtime();
		//testing_duration = (testing_t2 - testing_t1);

		//epoch_t2 = omp_get_wtime();
		//epoch_duration = (epoch_t2 - epoch_t1);
	//   printf("\n      Epoch Duration:    %f\n", epoch_duration);
	// 	printf("      Training Duration: %f\n", training_duration);
	// 	printf("      Syncing Duration:  %f\n", syncing_duration);
	// 	printf("      Testing Duration:  %f\n", testing_duration);
        MPI_Reduce(&local_score, &total_score, 1, MPI_DOUBLE, MPI_SUM, JCKY_MASTER, MPI_COMM_WORLD);
        if (mpi_manager.master) printf("          Total Score: %f\n", total_score);
	// 	epoch_times[epoch]    = epoch_duration;
	// 	training_times[epoch] = training_duration;
	// 	syncing_times[epoch]  = syncing_duration;
	// 	testing_times[epoch]  = testing_duration;
	// 	counts[epoch]         = count;
		local_score = 0.0;
	}

	// printf("Count, Epoch Time (s), Training Time (s), Syncing Time (s), Testing Time (s)\n");
	// for (epoch=0; epoch<EPOCHS; epoch++)
	// {
	// 	printf("%i, %f, %f, %f, %f\n", counts[epoch],
	// 	                               epoch_times[epoch],
	// 														     training_times[epoch],
	// 																 syncing_times[epoch],
	// 														     testing_times[epoch]);
	// }

    destroy_mpi_manager(&mpi_manager);
    destroy_meta_nn(&neural_net);

mpi_finalize:
    MPI_Finalize();
    jcky_close_file(training_file);
    jcky_close_file(testing_file);
	return 0;
}
