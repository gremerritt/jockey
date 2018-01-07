#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "constants.h"
#include "helpers.h"


void welcome(jcky_cli *cli, unsigned char master) {
    if (cli->verbose && master) {
        printf("            .''       _            __\n");
        printf("  ._.-.___.' (`\\     (_)___  _____/ /_____  __  __\n");
        printf(" //(        ( `'    / / __ \\/ ___/ //_/ _ \\/ / / /\n");
        printf("'/ )\\ ).__. )      / / /_/ / /__/ ,< /  __/ /_/ /\n");
        printf("' <' `/ ._/'/   __/ /\\____/\\___/_/|_|\\___/\\__, /\n");
        printf("   ` /     /   /___/                     /____/ v%s\n\n", JCKY_VERSION);
    }
}


help_text() {
    printf("Jockey is a small, fast, and scalable neural network which\n");
    printf("runs across an MPI cluster. Version %s.\n", JCKY_VERSION);
    printf("\n");
    printf("This program will write out two files during it's execution:\n");
    printf("  config.jockey - The layout, weights, and biases of the neural network.\n");
    printf("  %s - Detailed timing report of the program. This includes\n", JCKY_TIMING_FILENAME);
    printf("      the following metrics (all in milliseconds):\n");
    printf("          epoch_time:      Total time of the epoch.\n");
    printf("          copy:            Time to make a copy of the neural network used\n");
    printf("                           for processing.\n");
    printf("          shuffle:         Time to shuffle data and sync this information\n");
    printf("                           across the MPI network.\n");
    printf("          training:        Total training time.\n");
    printf("          training_batch:  Average time to create a training batch.\n");
    printf("          training_run:    Average time to push a training batch through the\n");
    printf("                           neural network. This includes both feed forward\n");
    printf("                           and backpropogation time.\n");
    printf("          sync:            Time to syncronize the neural networks from the\n");
    printf("                           various MPI processes.\n");
    printf("          testing:         Total training time.\n");
    printf("          testing_batch:   Average time to create a testing batch.\n");
    printf("          testing_run:     Average time to push a testing batch through the\n");
    printf("                           neural network. This includes both feed forward\n");
    printf("                           time (no backpropogation).\n");
    printf("\n");
    printf("Usage:\n");
    printf("    jockey --help/-h\n");
    printf("    jockey --version\n");
    printf("    jockey [options]\n");
    printf("\n");
    printf("Flags:\n");
    printf("    -v\n");
    printf("        Flag to run with verbose output.\n");
    printf("    --write/-w\n");
    printf("        Flag to call to the 'hooks.write_file' function'.\n");
    printf("    --no-save\n");
    printf("        Flag to ONLY save the neural network directly before the program\n");
    printf("        terminates.\n");
    printf("        Default: Save the neural network after each epoch. This will\n");
    printf("                 create a file 'neural_net.jockey' which \n");
    printf("    --no-timing\n");
    printf("        Flag to ONLY save the timing report directly before the prorgam\n");
    printf("        terminates.\n");
    printf("        NOTE: Compile jockey without '#define JCKY_TIMING' to completly\n");
    printf("              disable timing.\n");
    printf("        Default: Save the timing of the program after each epoch.\n");
    printf("\n");
    printf("Options:\n");
    printf("    --training-filename/--training-file/--train (str)\n");
    printf("        Path to training file. Required (unless running with the --write flag).\n");
    printf("    --testing-filename/--testing-file/--test (str)\n");
    printf("        Path to testing file. Required (unless running with the --write flag).\n");
    printf("    --model-filename/--model-file/--model (str)\n");
    printf("        Path to file to write model into.\n");
    printf("    --init-model-filename/--init-model-file/--init-model (str)\n");
    printf("        Path to model file used to initialize the neural network.\n");
    printf("    --hidden-layers/-hl (int)\n");
    printf("        Number of hidden layers.\n");
    printf("        Default: %i\n", DEFAULT_NUM_HIDDEN_LAYERS);
    printf("    --hidden-nodes/-hn (int)\n");
    printf("        Number of nodes in each hidden layer.\n");
    printf("        Default: %i\n", DEFAULT_NUM_NODES_IN_HIDDEN_LAYERS);
    printf("    --batch-size/-bs (int)\n");
    printf("        Number of samples in each batch.\n");
    printf("        Default: %i\n", DEFAULT_BATCH_SIZE);
    printf("    --learning-rate/-lr (float)\n");
    printf("        Learning rate.\n");
    printf("        Default: %f\n", DEFAULT_LEARNING_RATE);
    printf("    --epochs/-e (int)\n");
    printf("        Number of epochs to run for.\n");
    printf("        Default: %i\n", DEFAULT_EPOCHS);
    printf("    --seed/-s (int)\n");
    printf("        Randon seed used to initialize neural network.\n");
    printf("    --memory-layout/-ml (str)\n");
    printf("        Internal memory layout used to store the neural network. Options are \n");
    printf("        'contiguous' or 'logical'. There should rarely, if ever, be a reason to\n");
    printf("        use this option.\n");
    printf("        Default: %s\n", JCKY_CONTIGUOUS_LAYOUT);
    printf("    --blocks (int)\n");
    printf("        When syncing the neural network across the MPI network, send the data\n");
    printf("        in this many blocks. This only applies when using the 'contiguous'\n");
    printf("        memory layout. Cannot be combined with the 'block-size' argument.\n");
    printf("        Must be between 1 and %i.\n", UCHAR_MAX);
    printf("        Default: Send as much data as possible.\n");
    printf("    --block-size (int)\n");
    printf("        When syncing the neural network across the MPI network, send the data\n");
    printf("        in blocks of this size. This only applies when using the 'contiguous'\n");
    printf("        memory layout. Cannot be combined with the 'blocks' argument.\n");
    printf("        Must be between %i and %lu, and divisible by %i.\n",
        sizeof(nn_type), sizeof(nn_type)*INT_MAX, sizeof(nn_type));
    printf("        Default: Send as much data as possible.\n");
}


unsigned char process_command_line(
    int argc,
    char **argv,
    jcky_cli *cli,
    unsigned char master)
{
	int i;
    unsigned char err = 0;

    cli->action = JCKY_ACTION_RUN;
    cli->batch_size = DEFAULT_BATCH_SIZE;
    cli->block_size = 0;
    cli->epochs = DEFAULT_EPOCHS;
    cli->learning_rate = DEFAULT_LEARNING_RATE;
    cli->memory_layout = (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID;
    cli->num_blocks = 0;
    cli->number_of_hidden_layers = DEFAULT_NUM_HIDDEN_LAYERS;
    cli->number_of_nodes_in_hidden_layers = DEFAULT_NUM_NODES_IN_HIDDEN_LAYERS;
    cli->seed = -1;  // Signal to generate a random seed
    cli->testing_filename[0] = '\0';
    cli->training_filename[0] = '\0';
    cli->init_model_filename[0] = '\0';
    strcpy(cli->model_filename, JCKY_MODEL_FILENAME);
    cli->verbose = 0;
    cli->no_timing = 0;
    cli->no_save = 0;

	for (i=1; i<argc; i++) {
		char *option = argv[i];

        // Flags
        if (strncmp(option, "--write", 7) == 0 ||
            strncmp(option, "-w", 2) == 0) {
            cli->action = JCKY_ACTION_WRITE;
            continue;
        }
        else if (strncmp(option, "--verbose", 9) == 0 ||
                 strncmp(option, "-v", 2) == 0) {
            cli->verbose = 1;
            continue;
        }
        else if (strncmp(option, "--no-timing", 11) == 0) {
            cli->no_timing = 1;
            continue;
        }
        else if (strncmp(option, "--no-save", 9) == 0) {
            cli->no_save = 1;
            continue;
        }
        else if (strncmp(option, "--help", 6) == 0 ||
                 strncmp(option, "-h", 2) == 0) {
            if (master) help_text();
            err = 1;
            return err;
        }
        else if (strncmp(option, "--version", 9) == 0) {
            if (master) printf("%s\n", JCKY_VERSION);
            err = 1;
            return err;
        }

        // Options
        char *val = argv[++i];

		if (strncmp(option, "--hidden-layers", 15) == 0 ||
			strncmp(option, "-hl", 3) == 0) {
			cli->number_of_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if (strncmp(option, "--hidden-nodes", 14) == 0 ||
				 strncmp(option, "-hn", 3) == 0) {
			cli->number_of_nodes_in_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if (strncmp(option, "--batch-size", 12) == 0 ||
				 strncmp(option, "-bs", 3) == 0) {
			cli->batch_size = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if (strncmp(option, "--learning-rate", 15) == 0 ||
				 strncmp(option, "-lr", 3) == 0) {
			cli->learning_rate = (nn_type)strtod( strtok(val, " "), NULL);
		}
        else if (strncmp(option, "--epochs", 8) == 0 ||
				 strncmp(option, "-e", 2) == 0) {
			cli->epochs = (unsigned short int)strtol( strtok(val, " "), NULL, 10);
		}
        else if (strncmp(option, "--seed", 6) == 0 ||
                 strncmp(option, "-s", 2) == 0) {
			cli->seed = (int)strtol( strtok(val, " "), NULL, 10);
		}
        else if (strncmp(option, "--memory-layout", 15) == 0 ||
                 strncmp(option, "-ml", 3) == 0) {
            if (strncmp(val, JCKY_CONTIGUOUS_LAYOUT, strlen(JCKY_CONTIGUOUS_LAYOUT)) == 0) {
                cli->memory_layout = (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID;
            }
            else if (strncmp(val, JCKY_LOGICAL_LAYOUT, strlen(JCKY_LOGICAL_LAYOUT)) == 0) {
                cli->memory_layout = (unsigned char)JCKY_LOGICAL_LAYOUT_ID;
            }
            else {
                printf("Error: Unknown option '%s' for memory-layout.\n", val);
                err = 1;
                break;
            }
        }
        else if (strncmp(option, "--blocks", 8) == 0) {
            unsigned long tmp_num_blocks = strtoul( strtok(val, " "), NULL, 10);
            if ((tmp_num_blocks < 1) || (tmp_num_blocks > UCHAR_MAX)) {
                if (master) {
                    printf("Error: Invalid value %lu for 'blocks'. Must be between 1 and %u.\n",
                            tmp_num_blocks, UCHAR_MAX);
                }
                err = 1;
                break;
            }
            else {
                cli->num_blocks = (unsigned char)tmp_num_blocks;
            }
        }
        else if (strncmp(option, "--block-size", 12) == 0) {
            unsigned long tmp_block_size = strtoul( strtok(val, " "), NULL, 10);
            if (tmp_block_size % sizeof(nn_type) != 0) {
                if (master) {
                    printf("Error: Invalid value %lu for 'blocks'. Must be divisible by %lu.\n",
                            tmp_block_size, sizeof(nn_type));
                }
                err = 1;
                break;
            }
            else if ((tmp_block_size < 1) || ((tmp_block_size / sizeof(nn_type) > INT_MAX))) {
                if (master) {
                    printf("Error: Invalid value %lu for 'blocks'. Must be between 1 and %lu.\n",
                            tmp_block_size, INT_MAX * sizeof(nn_type));
                }
                err = 1;
                break;
            }
            else {
                cli->block_size = (unsigned int)tmp_block_size;
            }
        }
        else if (strncmp(option, "--training-filename", 19) == 0 ||
                 strncmp(option, "--training-file", 15) == 0||
                 strncmp(option, "--train", 7) == 0) {
            if (val == NULL) continue;
            size_t training_filename_len = strlen(val);
            strncpy(cli->training_filename, val, 127);
            cli->training_filename[(training_filename_len > 126) ? 127 : training_filename_len] = '\0';
        }
        else if (strncmp(option, "--testing-filename", 18) == 0 ||
                 strncmp(option, "--testing-file", 14) == 0 ||
                 strncmp(option, "--test", 6) == 0) {
            if (val == NULL) continue;
            size_t testing_filename_len = strlen(val);
            strncpy(cli->testing_filename, val, 127);
            cli->testing_filename[(testing_filename_len > 126) ? 127 : testing_filename_len] = '\0';
        }
        else if (strncmp(option, "--init-model-filename", 21) == 0 ||
                 strncmp(option, "--init-model-file", 17) == 0 ||
                 strncmp(option, "--init-model", 12) == 0) {
            if (val == NULL) continue;
            size_t init_model_filename_len = strlen(val);
            strncpy(cli->init_model_filename, val, 127);
            cli->init_model_filename[(init_model_filename_len > 126) ? 127 : init_model_filename_len] = '\0';
        }
        else if (strncmp(option, "--model-filename", 16) == 0 ||
                 strncmp(option, "--model-file", 12) == 0 ||
                 strncmp(option, "--model", 7) == 0) {
            if (val == NULL) continue;
            size_t model_filename_len = strlen(val);
            strncpy(cli->model_filename, val, 127);
            cli->model_filename[(model_filename_len > 126) ? 127 : model_filename_len] = '\0';
        }
	}

    if (!err) {
        if (master && (cli->memory_layout == JCKY_LOGICAL_LAYOUT_ID) && (cli->num_blocks || cli->block_size)) {
            printf("Warning: 'blocks' and 'block-size' parameters have no effect when using logical memory layout.\n");
        }
        if ((cli->memory_layout == JCKY_CONTIGUOUS_LAYOUT_ID) && cli->num_blocks && cli->block_size) {
            if (master) {
                printf("Error: 'blocks' and 'block-size' parameters are mutually exclusive.\n");
            }
            err = 1;
        }
        if (cli->action == JCKY_ACTION_RUN && (strlen(cli->training_filename) == 0 || strlen(cli->testing_filename) == 0)) {
            if (master) {
                printf("Error: Must provide a training file and a testing file.\n");
            }
            err = 1;
        }
    }

    return err;
}


void print_number(unsigned short int number, unsigned short int len) {
    unsigned short int digits = (unsigned short int)(floor(log10(number)) + 1);
    unsigned short int i;
    unsigned short int buffer = len - digits;

    for(i=0; i<buffer; i++) printf(" ");
    printf("%i", number);
}


unsigned int round_up_multiple(unsigned int number, unsigned int multiple) {
    if (multiple == 0)
        return number;

    unsigned int remainder = number % multiple;
    if (remainder == 0)
        return number;

    return number + multiple - remainder;
}


struct timespec diff_time(struct timespec start, struct timespec end) {
    struct timespec temp;

    if ((end.tv_nsec-start.tv_nsec) < 0) {
        temp.tv_sec = end.tv_sec - start.tv_sec-1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }

    return temp;
}
