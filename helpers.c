#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "constants.h"
#include "helpers.h"


void welcome(unsigned char master) {
    if (master) {
        printf("            .''       _            __\n");
        printf("  ._.-.___.' (`\\     (_)___  _____/ /_____  __  __\n");
        printf(" //(        ( `'    / / __ \\/ ___/ //_/ _ \\/ / / /\n");
        printf("'/ )\\ ).__. )      / / /_/ / /__/ ,< /  __/ /_/ /\n");
        printf("' <' `/ ._/'/   __/ /\\____/\\___/_/|_|\\___/\\__, /\n");
        printf("   ` /     /   /___/                     /____/ v%s\n\n", JCKY_VERSION);
    }
}


unsigned char process_command_line(
    int argc,
    char **argv,
    jcky_cli *cli)
{
	int i;
    unsigned char err = 0;

    cli->number_of_hidden_layers = DEFAULT_NUM_HIDDEN_LAYERS;
    cli->number_of_nodes_in_hidden_layers = DEFAULT_NUM_NODES_IN_HIDDEN_LAYERS;
    cli->batch_size = DEFAULT_BATCH_SIZE;
    cli->learning_rate = DEFAULT_LEARNING_RATE;
    cli->seed = -1;  // Signal to generate a random seed
    cli->memory_layout = (unsigned char)JCKY_CONTIGUOUS_LAYOUT_ID;
    cli->epochs = DEFAULT_EPOCHS;
    cli->num_blocks = 0;
    cli->block_size = 0;
    cli->action = JCKY_ACTION_RUN;
    cli->training_filename[0] = '\0';
    cli->testing_filename[0] = '\0';

	for (i=1; i<argc; i++) {
		char *str   = argv[i];
		char *param = strtok(str, "=");
		char *val   = strtok(NULL, "=");

		if ((strncmp(param, "--hidden-layers", 15) == 0) ||
			(strncmp(param, "--hl", 4) == 0)) {
			cli->number_of_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strncmp(param, "--hidden-nodes", 14) == 0) ||
				 (strncmp(param, "--hn", 4) == 0)) {
			cli->number_of_nodes_in_hidden_layers = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strncmp(param, "--batch-size", 12) == 0) ||
				 (strncmp(param, "--bs", 4) == 0)) {
			cli->batch_size = (int)strtol( strtok(val, " "), NULL, 10);
		}
		else if ((strncmp(param, "--learning-rate", 15) == 0) ||
				 (strncmp(param, "--lr", 4) == 0)) {
			cli->learning_rate = (nn_type)strtod( strtok(val, " "), NULL);
		}
        else if ((strncmp(param, "--epochs", 8) == 0) ||
				 (strncmp(param, "--e", 3) == 0)) {
			cli->epochs = (unsigned short int)strtol( strtok(val, " "), NULL, 10);
		}
        else if (strncmp(param, "--seed", 6) == 0) {
			cli->seed = (int)strtol( strtok(val, " "), NULL, 10);
		}
        else if (strncmp(param, "--memory-layout", 15) == 0) {
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
        else if (strncmp(param, "--blocks", 8) == 0) {
            unsigned long tmp_num_blocks = strtoul( strtok(val, " "), NULL, 10);
            if ((tmp_num_blocks < 1) || (tmp_num_blocks > UCHAR_MAX)) {
                printf("Error: Invalid value %lu for 'blocks'. Must be between 1 and %u.\n", tmp_num_blocks, UCHAR_MAX);
                err = 1;
                break;
            }
            else {
                cli->num_blocks = (unsigned char)tmp_num_blocks;
            }
        }
        else if (strncmp(param, "--block-size", 12) == 0) {
            unsigned long tmp_block_size = strtoul( strtok(val, " "), NULL, 10);
            if (tmp_block_size % sizeof(nn_type) != 0) {
                printf("Error: Invalid value %lu for 'blocks'. Must be divisible by %lu.\n",
                    tmp_block_size, sizeof(nn_type));
                err = 1;
                break;
            }
            else if ((tmp_block_size < 1) || ((tmp_block_size / sizeof(nn_type) > INT_MAX))) {
                printf("Error: Invalid value %lu for 'blocks'. Must be between 1 and %lu.\n",
                    tmp_block_size, INT_MAX * sizeof(nn_type));
                err = 1;
                break;
            }
            else {
                cli->block_size = (unsigned int)tmp_block_size;
            }
        }
        else if (strncmp(param, "--write", 7) == 0) {
            cli->action = JCKY_ACTION_WRITE;
        }
        else if (strncmp(param, "--training-filename", 19) == 0 ||
                 strncmp(param, "--training-file", 15) == 0||
                 strncmp(param, "--train", 7) == 0) {
            if (val == NULL) continue;
            size_t training_filename_len = strlen(val);
            strncpy(cli->training_filename, val, 127);
            cli->training_filename[(training_filename_len > 126) ? 127 : training_filename_len] = '\0';
        }
        else if (strncmp(param, "--testing-filename", 18) == 0 ||
                 strncmp(param, "--testing-file", 14) == 0 ||
                 strncmp(param, "--test", 6) == 0) {
            if (val == NULL) continue;
            size_t testing_filename_len = strlen(val);
            strncpy(cli->testing_filename, val, 127);
            cli->testing_filename[(testing_filename_len > 126) ? 127 : testing_filename_len] = '\0';
        }
	}

    if (!err) {
        if ((cli->memory_layout == JCKY_LOGICAL_LAYOUT_ID) && (cli->num_blocks || cli->block_size)) {
            printf("Warning: 'blocks' and 'block-size' parameters have no effect when using logical memory layout.\n");
        }
        if ((cli->memory_layout == JCKY_CONTIGUOUS_LAYOUT_ID) && cli->num_blocks && cli->block_size) {
            printf("Error: 'blocks' and 'block-size' parameters are mutually exclusive.\n");
            err = 1;
        }
        if (cli->action == JCKY_ACTION_RUN &&
            (strlen(cli->training_filename) == 0 || strlen(cli->testing_filename) == 0)) {
            printf("Error: Must provide a training file and a testing file.\n");
            err = 1;
        }
    }

    return err;
}


unsigned int round_up_multiple(unsigned int number, unsigned int multiple) {
    if (multiple == 0)
        return number;

    unsigned int remainder = number % multiple;
    if (remainder == 0)
        return number;

    return number + multiple - remainder;
}
