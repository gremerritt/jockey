#ifndef CONSTANTS_H
#define CONSTANTS_H

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

#endif
