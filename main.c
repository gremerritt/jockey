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
#include "file_helpers.h"
#include "batch.h"
//#include <omp.h>
#include <mpi.h>

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

// void print_result(int iter,
// 	              int *label,
//                   nn_type *result,
// 				  char *correct);
int write_file();
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
    unsigned int *nn_block_size,
    unsigned char *action,
    char *training_filename,
    char *testing_filename
);
int main(int argc, char **argv) {
    int number_of_hidden_layers = NUM_HIDDEN_LAYERS;
    int number_of_nodes_in_hidden_layers = NUM_NODES_IN_HIDDEN_LAYERS;
    int batch_size = BATCH_SIZE;
    nn_type learning_rate = LEARNING_RATE;
    int seed = -1;
    unsigned char memory_layout = (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID;
    unsigned char num_blocks = 0;
    unsigned int block_size = 0;
    unsigned char action = JCKY_ACTION_RUN;
    char training_filename[128] = "\0";
    char testing_filename[128] = "\0";
    jcky_file training_file;
    jcky_file testing_file;

    // DELETE ME
    mnist_data *training_data;
    mnist_data *testing_data;

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
        &block_size,
        &action,
        training_filename,
        testing_filename
    );
    if (pcl != 0) return -1;
    else if (action == JCKY_ACTION_WRITE) {
        return write_file();
    }
    else if (action == JCKY_ACTION_RUN) {
        training_file = jcky_open_file(training_filename);
        if (training_file.stream == NULL) return -1;

        testing_file = jcky_open_file(testing_filename);
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
	int counts[EPOCHS];
	double epoch_times[EPOCHS];
	double training_times[EPOCHS];
	double syncing_times[EPOCHS];
	double testing_times[EPOCHS];
    unsigned short int k;

    MPI_Barrier(MPI_COMM_WORLD);

    struct meta_neural_net neural_net = create_neural_net(
        number_of_hidden_layers,
        number_of_nodes_in_hidden_layers,
        jcky_get_num_inputs(training_file),
        jcky_get_num_outputs(training_file),
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
			nn_type result[neural_net.number_of_outputs * neural_net.batch_size];
			nn_type batch[neural_net.number_of_inputs * neural_net.batch_size];
			nn_type targets[neural_net.number_of_outputs * neural_net.batch_size];
			char correct[neural_net.batch_size];

			create_batch_with_sequence_file(batch, targets, &training_file, neural_net.batch_size, i, sequence);
			feed_forward(&neural_net, result, batch, targets, JCKY_TRAIN, &local_count, correct);
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

			create_batch_no_sequence_file(batch, targets, &testing_file, neural_net.batch_size, i, mpi_manager.rank, &(mpi_manager.testing_samples));
			feed_forward(&neural_net, result, batch, targets, JCKY_TEST, &local_count, correct);
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
	free(testing_data);

mpi_finalize:
    MPI_Finalize();
    jcky_close_file(training_file);
    jcky_close_file(testing_file);
	return 0;
}

// void print_result(int iter, int *label, nn_type *result, char *correct)
// {
// 	int row, col;
// 	printf("\n    ITERATION %i\n    ", iter);
// 	for (row=0; row<BATCH_SIZE; row++) printf("       %i  ", label[row]);
// 	printf("\n    ");
// 	for (row=0; row<NUM_OUTPUTS; row++) {
// 		for (col=0; col<BATCH_SIZE; col++) {
// 			if (correct[col] == row+10)   printf(KRED "%f  ", result[(row*BATCH_SIZE)+col]);
// 			else if (correct[col] == row) printf(KGRN "%f  ", result[(row*BATCH_SIZE)+col]);
// 			else                          printf(KNRM "%f  ", result[(row*BATCH_SIZE)+col]);
// 		}
// 		printf(KNRM "\n    ");
// 	}
// }

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
    unsigned int *nn_block_size,
    unsigned char *action,
    char *training_filename,
    char *testing_filename)
{
	int i;
    unsigned char err = 0;

	for (i=1; i<argc; i++) {
		char *str   = argv[i];
		char *param = strtok(str, "=");
		char *val   = strtok(NULL, "=");

		if ((strncmp(param, "--hidden-layers", 15) == 0) ||
			(strncmp(param, "--hl", 4) == 0)) {
			*number_of_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strncmp(param, "--hidden-nodes", 14) == 0) ||
				 (strncmp(param, "--hn", 4) == 0)) {
			*number_of_nodes_in_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strncmp(param, "--batch-size", 12) == 0) ||
				 (strncmp(param, "--bs", 4) == 0)) {
			*batch_size = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strncmp(param, "--learning-rate", 15) == 0) ||
				 (strncmp(param, "--lr", 4) == 0)) {
			*learning_rate = (nn_type)strtod( strtok(val, " "), NULL);
		}
        else if (strncmp(param, "--seed", 6) == 0) {
			*seed = (int)strtol( strtok(val, " "), NULL, 10);
		}
        else if (strncmp(param, "--memory-layout", 15) == 0) {
            if (strncmp(val, JCKY_CONTIGUOUS_LAYOUT, strlen(JCKY_CONTIGUOUS_LAYOUT)) == 0) {
                *nn_alloc_method = (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID;
            }
            else if (strncmp(val, JCKY_LOGICAL_LAYOUT, strlen(JCKY_LOGICAL_LAYOUT)) == 0) {
                *nn_alloc_method = (unsigned char)JCKY_LOGICAL_LAYOUT_ID;
            }
            else {
                printf("Error: Unknown option '%s' for memory-layout.\n", val);
                err = 1;
                break;
            }
        }
        else if (strncmp(param, "--blocks", 8) == 0) {
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
        else if (strncmp(param, "--block-size", 12) == 0) {
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
        else if (strncmp(param, "--write", 7) == 0) {
            *action = JCKY_ACTION_WRITE;
        }
        else if (strncmp(param, "--training-filename", 19) == 0 ||
                 strncmp(param, "--training-file", 15) == 0||
                 strncmp(param, "--train", 7) == 0) {
            if (val == NULL) continue;
            size_t training_filename_len = strlen(val);
            strncpy(training_filename, val, 127);
            training_filename[(training_filename_len > 126) ? 127 : training_filename_len] = '\0';
        }
        else if (strncmp(param, "--testing-filename", 18) == 0 ||
                 strncmp(param, "--testing-file", 14) == 0 ||
                 strncmp(param, "--test", 6) == 0) {
            if (val == NULL) continue;
            size_t testing_filename_len = strlen(val);
            strncpy(testing_filename, val, 127);
            testing_filename[(testing_filename_len > 126) ? 127 : testing_filename_len] = '\0';
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
        if (*action == JCKY_ACTION_RUN &&
            (strlen(training_filename) == 0 || strlen(testing_filename) == 0)) {
            printf("Error: Must provide a training file and a testing file.\n");
            err = 1;
        }
    }

    return err;
}

int write_file() {
    int i, j, ret = 0;
    mnist_data *training_data;
    mnist_data *testing_data;
    unsigned int training_cnt, testing_cnt;
    nn_type **training_data_writable, **training_labels_writable;
    nn_type **testing_data_writable, **testing_labels_writable;

    printf("Loading training image set... ");
	ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &training_data, &training_cnt);
	if (ret) {
		printf("An error occured: %d\n", ret);
		printf("Make sure image files (*-ubyte) are in the current directory.\n");
		return ret;
	}
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", training_cnt);
	}

	printf("\nLoading test image set... ");
	ret = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &testing_data, &testing_cnt);
	if (ret) {
		printf("An error occured: %d\n", ret);
		printf("Make sure image files (*-ubyte) are in the current directory.\n");
		return ret;
	}
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", testing_cnt);
	}

    training_data_writable = malloc( training_cnt * sizeof( nn_type* ));
    training_labels_writable = malloc( training_cnt * sizeof( nn_type* ));
    for(i=0; i<training_cnt; i++) {
        training_data_writable[i] = training_data->data;
        training_labels_writable[i] = (nn_type *)malloc( 10 * sizeof( nn_type ) );
        for(j=0; j<10; j++) training_labels_writable[i][j] = (nn_type)((j == training_data->label) ? 1.0 : 0.0);
    }

    testing_data_writable = malloc( testing_cnt * sizeof( nn_type* ));
    testing_labels_writable = malloc( testing_cnt * sizeof( nn_type* ));
    for(i=0; i<testing_cnt; i++) {
        testing_data_writable[i] = testing_data->data;
        testing_labels_writable[i] = (nn_type *)malloc( 10 * sizeof( nn_type ) );
        for(j=0; j<10; j++) testing_labels_writable[i][j] = (nn_type)((j == testing_data->label) ? 1.0 : 0.0);
    }

    printf("Writing training file... ");
    ret = jcky_write_file(training_data_writable, training_labels_writable, training_cnt, 28*28, 10, "training.jockey");
    if (ret) printf("Failed.\n");
    else printf("Success!\n");

    printf("Writing testing file... ");
    jcky_write_file(testing_data_writable, testing_labels_writable, testing_cnt, 28*28, 10, "testing.jockey");
    if (ret) printf("Failed.\n");
    else printf("Success!\n");

    for(i=0; i<training_cnt; i++) {
        free(training_labels_writable[i]);
    }
    for(i=0; i<testing_cnt; i++) {
        free(testing_labels_writable[i]);
    }
    free(training_data_writable);
    free(testing_data_writable);

    return ret;
}
