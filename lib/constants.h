#ifndef CONSTANTS_H
#define CONSTANTS_H


typedef double nn_type;

#define JCKY_VERSION "0.0.0"

#define JCKY_MASTER 0

#define JCKY_NN_BASE 0
#define JCKY_NN_SCRATCH 1

#define JCKY_NN_INIT_FUNCTIONS 1

#define JCKY_TRAIN JCKY_NN_SCRATCH
#define JCKY_TEST JCKY_NN_BASE

#define JCKY_CONTIGUOUS_LAYOUT "contiguous"
#define JCKY_LOGICAL_LAYOUT "logical"
enum memory_layouts{JCKY_CONTIGUOUS_LAYOUT_ID, JCKY_LOGICAL_LAYOUT_ID};

#define JCKY_DEFAULT_FILE_NAME "data.jockey"
enum type_identifiers{JCKY_FLOAT, JCKY_DOUBLE};

#define JCKY_ACTION_RUN 0
#define JCKY_ACTION_WRITE 1

#define DEFAULT_NUM_HIDDEN_LAYERS 2
#define DEFAULT_NUM_NODES_IN_HIDDEN_LAYERS 60
#define DEFAULT_BATCH_SIZE 5
#define DEFAULT_LEARNING_RATE 1.5
#define DEFAULT_EPOCHS 100

#define JCKY_TIMING
#define JCKY_TIMING_FILENAME "timing.jockey.csv"

#define JCKY_MODEL_FILENAME "model.jockey"

// Colors
#define KNRM "\x1B[0m"
#define KYEL "\x1B[33m"
#define KRED "\x1B[31m"


#endif
