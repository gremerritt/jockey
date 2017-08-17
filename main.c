#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include "mnist.h"
#include "constants.h"
#include "helpers.h"
#include "neural_net.h"
#include "randomizing_helpers.h"
#include "matrix_helpers.h"
#include "mpi_helper.h"
//#include <omp.h>
#include <mpi.h>

#define DIM 28
#define DATA_SIZE DIM*DIM
#define NUM_OUTPUTS 10
#define NUM_NODES_IN_HIDDEN_LAYERS 60
#define NUM_HIDDEN_LAYERS 2
#define LEARNING_RATE 1.5
#define BATCH_SIZE 5
#define EPOCHS 3
#define TRAINING_SAMPLES 60000
#define TEST_SAMPLES 10000
#define TEST_PRINT_RESULTS_EVERY 100000

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"

void create_batch_with_sequence(nn_type *batch,
                                int *label,
                                mnist_data *data,
                                int batch_size,
                                const unsigned int iteration,
                                unsigned int *sequence);
void create_batch_no_sequence(nn_type *batch,
	                          int *label,
							  mnist_data *data,
							  int batch_size,
                              const unsigned int iteration,
                              unsigned short int rank,
                              sample_manager *sample_manager);
void print_result(int iter,
	              int *label,
                  nn_type *result,
				  char *correct);
unsigned char process_command_line(
    int argc,
    char **argv,
    int *number_of_hidden_layers,
    int *number_of_nodes_in_hidden_layers,
    int *batch_size,
    nn_type *learning_rate,
    int *seed,
    unsigned char *nn_alloc_method,
    unsigned char *nn_num_blocks,
    unsigned int *nn_block_size
);
int main(int argc, char **argv) {
    mpi_manager mpi_manager = mpi_init(argc, argv);

    welcome(mpi_manager.master);
    mpi_announce(&mpi_manager);
    const unsigned short int world_size = mpi_manager.world_size;

	mnist_data *training_data;
	mnist_data *test_data;
	unsigned int cnt;
	int ret;
	double epoch_t1, epoch_t2, epoch_duration;
	double training_t1, training_t2, training_duration;
	double syncing_t1, syncing_t2, syncing_duration;
	double testing_t1, testing_t2, testing_duration;
	int counts[EPOCHS];
	double epoch_times[EPOCHS];
	double training_times[EPOCHS];
	double syncing_times[EPOCHS];
	double testing_times[EPOCHS];
    unsigned short int k;

    // If the future, only master will load data. It will then distribute it
    // to the child procs.
    //
    // Also in the future, you won't need to load the whole data set into memory.
    // You should be able to load the data in chunks of at least the batch size
    // as you need it.
	printf("Loading training image set... ");
	ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &training_data, &cnt);
	if (ret) {
		printf("An error occured: %d\n", ret);
		printf("Make sure image files (*-ubyte) are in the current directory.\n");
		return 0;
	}
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", cnt);
	}

	printf("\nLoading test image set... ");
	ret = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &test_data, &cnt);
	if (ret) {
		printf("An error occured: %d\n", ret);
		printf("Make sure image files (*-ubyte) are in the current directory.\n");
		return 0;
	}
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", cnt);
	}

    MPI_Barrier(MPI_COMM_WORLD);

	int number_of_hidden_layers = NUM_HIDDEN_LAYERS;
	int number_of_nodes_in_hidden_layers = NUM_NODES_IN_HIDDEN_LAYERS;
	int number_of_inputs = DIM*DIM;
	int number_of_outputs = NUM_OUTPUTS;
	int batch_size = BATCH_SIZE;
	nn_type learning_rate = LEARNING_RATE;
    int seed = -1;
    unsigned char memory_layout = (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID;
    unsigned char num_blocks = 0;
    unsigned int block_size = 0;

	unsigned char pcl = process_command_line(
        argc,
        argv,
        &number_of_hidden_layers,
        &number_of_nodes_in_hidden_layers,
        &batch_size,
        &learning_rate,
        &seed,
        &memory_layout,
        &num_blocks,
        &block_size
    );
    if (pcl != 0) return 0;

    struct meta_neural_net neural_net = create_neural_net(
        number_of_hidden_layers,
        number_of_nodes_in_hidden_layers,
        number_of_inputs,
        number_of_outputs,
        batch_size,
        learning_rate,
        memory_layout,
        num_blocks,
        block_size
    );

    if (mpi_manager.master) {
        neural_net.functions->init(&neural_net, seed);
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
    	printf("\n------------------\n");
    }

    jcky_waitall(&(mpi_manager.neural_net));

	unsigned int local_count = 0;
    unsigned int total_count;
	int epoch;
	int i;
	unsigned int *sequence = malloc( mpi_manager.training_samples.total_len * sizeof(int) );

	if (mpi_manager.master) printf("Training...\n");
    const unsigned int training_batches = mpi_manager.training_samples.batches;
    const unsigned int testing_batches = mpi_manager.testing_samples.batches;
    const unsigned short int child_procs = mpi_manager.child_procs;
	for (epoch=0; epoch<EPOCHS; epoch++) {
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
			nn_type result [NUM_OUTPUTS * BATCH_SIZE];
			nn_type batch  [DIM * DIM * BATCH_SIZE];
			int     label  [BATCH_SIZE];
			char    correct[BATCH_SIZE];

			create_batch_with_sequence(batch, label, training_data, BATCH_SIZE, i, sequence);
			feed_forward(&neural_net, result, batch, label, JCKY_TRAIN, &local_count, correct);
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
			nn_type result [NUM_OUTPUTS * BATCH_SIZE];
			nn_type batch  [DIM * DIM * BATCH_SIZE];
			int     label  [BATCH_SIZE];
			char    correct[BATCH_SIZE];

			create_batch_no_sequence(batch, label, test_data, BATCH_SIZE, i, mpi_manager.rank, &(mpi_manager.testing_samples));
			feed_forward(&neural_net, result, batch, label, JCKY_TEST, &local_count, correct);
            //if (mpi_manager.master) printf(" - %i correct\n", count);
			// if (mpi_manager.master && i != 0 && i%TEST_PRINT_RESULTS_EVERY == 0) {
			// 	print_result(i, label, result, correct);
			// }
		}
		//testing_t2 = omp_get_wtime();
		//testing_duration = (testing_t2 - testing_t1);

		//epoch_t2 = omp_get_wtime();
		//epoch_duration = (epoch_t2 - epoch_t1);
	//   printf("\n      Epoch Duration:    %f\n", epoch_duration);
	// 	printf("      Training Duration: %f\n", training_duration);
	// 	printf("      Syncing Duration:  %f\n", syncing_duration);
	// 	printf("      Testing Duration:  %f\n", testing_duration);
        MPI_Reduce(&local_count, &total_count, 1, MPI_UNSIGNED, MPI_SUM, JCKY_MASTER, MPI_COMM_WORLD);
        if (mpi_manager.master) printf("          Total Count: %u\n", total_count);
	// 	epoch_times[epoch]    = epoch_duration;
	// 	training_times[epoch] = training_duration;
	// 	syncing_times[epoch]  = syncing_duration;
	// 	testing_times[epoch]  = testing_duration;
	// 	counts[epoch]         = count;
		local_count = 0;
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
	free(training_data);
	free(test_data);

    MPI_Finalize();
	return 0;
}

void create_batch_with_sequence(
    nn_type *batch,
    int *label,
    mnist_data *data,
    int batch_size,
    const unsigned int iteration,
    unsigned int *sequence)
{
    const unsigned int offset = iteration * batch_size;
    unsigned short int i;
	unsigned int j, seq_index;
	for (i=0; i<batch_size; i++) {
        seq_index = i + offset;
		label[i] = data[sequence[seq_index]].label;
		for (j=0; j<DATA_SIZE; j++) {
			batch[(j*batch_size) + i] = data[sequence[seq_index]].data[j];
		}
	}
}

void create_batch_no_sequence(nn_type *batch,
	                          int *label,
							  mnist_data *data,
							  int batch_size,
                              const unsigned int iteration,
                              unsigned short int rank,
                              sample_manager *sample_manager)
{
    const unsigned int offset = (iteration * batch_size) + (rank * (*sample_manager).base);
    unsigned short int i;
	unsigned int j, data_index;
	for (i=0; i<batch_size; i++) {
        data_index = i + offset;
		label[i] = data[data_index].label;
		for (j=0; j<DATA_SIZE; j++) {
			batch[(j*batch_size) + i] = data[data_index].data[j];
		}
	}
}

void print_result(int iter, int *label, nn_type *result, char *correct)
{
	int row, col;
	printf("\n    ITERATION %i\n    ", iter);
	for (row=0; row<BATCH_SIZE; row++) printf("       %i  ", label[row]);
	printf("\n    ");
	for (row=0; row<NUM_OUTPUTS; row++) {
		for (col=0; col<BATCH_SIZE; col++) {
			if (correct[col] == row+10)   printf(KRED "%f  ", result[(row*BATCH_SIZE)+col]);
			else if (correct[col] == row) printf(KGRN "%f  ", result[(row*BATCH_SIZE)+col]);
			else                          printf(KNRM "%f  ", result[(row*BATCH_SIZE)+col]);
		}
		printf(KNRM "\n    ");
	}
}

unsigned char process_command_line(
    int argc,
    char **argv,
    int *number_of_hidden_layers,
    int *number_of_nodes_in_hidden_layers,
    int *batch_size,
    nn_type *learning_rate,
    int *seed,
    unsigned char *nn_alloc_method,
    unsigned char *nn_num_blocks,
    unsigned int *nn_block_size)
{
	int i;
    unsigned char err = 0;

	for (i=1; i<argc; i++) {
		char *str   = argv[i];
		char *param = strtok(str, "=");
		char *val   = strtok(NULL, "=");

		if ((strcmp(param, "--hidden-layers") == 0) ||
			(strcmp(param, "--hl") == 0)) {
			*number_of_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strcmp(param, "--hidden-nodes") == 0) ||
				 (strcmp(param, "--hn") == 0)) {
			*number_of_nodes_in_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strcmp(param, "--batch-size") == 0) ||
				 (strcmp(param, "--bs") == 0)) {
			*batch_size = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strcmp(param, "--learning-rate") == 0) ||
				 (strcmp(param, "--lr") == 0)) {
			*learning_rate = (nn_type)strtod( strtok(val, " "), NULL);
		}
        else if (strcmp(param, "--seed") == 0) {
			*seed = (int)strtol( strtok(val, " "), NULL, 10);
		}
        else if (strcmp(param, "--memory-layout") == 0) {
            if (strcmp(val, JCKY_CONTIGUOUS_LAYOUT) == 0) {
                *nn_alloc_method = (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID;
            }
            else if (strcmp(val, JCKY_LOGICAL_LAYOUT) == 0) {
                *nn_alloc_method = (unsigned char)JCKY_LOGICAL_LAYOUT_ID;
            }
            else {
                printf("Error: Unknown option '%s' for memory-layout.\n", val);
                err = 1;
                break;
            }
        }
        else if (strcmp(param, "--blocks") == 0) {
            unsigned long tmp_nn_num_blocks = strtoul( strtok(val, " "), NULL, 10);
            if ((tmp_nn_num_blocks < 1) || (tmp_nn_num_blocks > UCHAR_MAX)) {
                printf("Error: Invalid value %lu for 'blocks'. Must be between 1 and %u.\n", tmp_nn_num_blocks, UCHAR_MAX);
                err = 1;
                break;
            }
            else {
                *nn_num_blocks = (unsigned char)tmp_nn_num_blocks;
            }
        }
        else if (strcmp(param, "--block-size") == 0) {
            unsigned long tmp_nn_block_size = strtoul( strtok(val, " "), NULL, 10);
            if (tmp_nn_block_size % sizeof(nn_type) != 0) {
                printf("Error: Invalid value %lu for 'blocks'. Must be divisible by %lu.\n",
                    tmp_nn_block_size, sizeof(nn_type));
                err = 1;
                break;
            }
            else if ((tmp_nn_block_size < 1) || ((tmp_nn_block_size / sizeof(nn_type) > INT_MAX))) {
                printf("Error: Invalid value %lu for 'blocks'. Must be between 1 and %lu.\n",
                    tmp_nn_block_size, INT_MAX * sizeof(nn_type));
                err = 1;
                break;
            }
            else {
                *nn_block_size = (unsigned int)tmp_nn_block_size;
            }
        }
	}

    if (!err) {
        if ((*nn_alloc_method == JCKY_LOGICAL_LAYOUT_ID) && (*nn_num_blocks || *nn_block_size)) {
            printf("Warning: 'blocks' and 'block-size' parameters have no effect when using logical memory layout.\n");
        }
        if ((*nn_alloc_method == JCKY_CONTIGUOUS_LAYOUT_ID) && *nn_num_blocks && *nn_block_size) {
            printf("Error: 'blocks' and 'block-size' parameters are mutually exclusive.\n");
            err = 1;
        }
    }

    return err;
}
