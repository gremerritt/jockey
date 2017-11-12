#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "file_helpers.h"
#include "neural_net.h"
#include "batch.h"

#define RECORDS 6
#define DATA_LEN 3
#define TARGETS_LEN 2
#define INCREMENT 0.1
#define BATCH 3
#define FILENAME "test_file.jockey"

int main(int argc, char **argv) {
    printf("Running Tests\n    ");
    assert((RECORDS % BATCH == 0) && "Invalid test data\n");

    char ret;
    nn_type counter = 0.0;
    unsigned int i, j;
    nn_type **test_data, **test_targets;
    nn_type *batch_data, *batch_targets;
    unsigned int *sequence;

    test_data = malloc( RECORDS * sizeof( nn_type* ));
    test_targets = malloc( RECORDS * sizeof( nn_type* ));
    sequence = malloc( RECORDS * sizeof( unsigned int* ));
    batch_data = malloc( DATA_LEN * BATCH * sizeof( nn_type* ));
    batch_targets = malloc( TARGETS_LEN * BATCH * sizeof( nn_type* ));
    jcky_file file;

    for(i=0; i<RECORDS; i++) {
        test_data[i] = (nn_type *)malloc( DATA_LEN * sizeof( nn_type ) );
        test_targets[i] = (nn_type *)malloc( TARGETS_LEN * sizeof( nn_type ) );

        for(j=0; j<DATA_LEN; j++) {
            test_data[i][j] = counter;
            counter += INCREMENT;
        }

        for(j=0; j<TARGETS_LEN; j++) {
            test_targets[i][j] = counter;
            counter += INCREMENT;
        }
    }

    ret = jcky_write_file(test_data, test_targets, RECORDS, DATA_LEN, TARGETS_LEN, FILENAME);
    assert((ret == 0) && "Jockey file failed to write.\n");
    printf(".");

    file = jcky_open_file(FILENAME);
    assert((file.stream != NULL) && "Jockey file failed to open.\n");
    printf(".");
    assert((file.offset == jcky_file_byte_offset()) && "Incorrect file offset.\n");
    printf(".");
    assert((file.datum_size == sizeof(nn_type)) && "Incorrect file datum size.\n");
    printf(".");
    assert((file.bytes_per_record == (sizeof(nn_type) * (DATA_LEN + TARGETS_LEN))) && "Incorrect file bytes per record.\n");
    printf(".");
    assert((file.bytes_per_data == sizeof(nn_type) * (DATA_LEN)) && "Incorrect file bytes per data.\n");
    printf(".");
    assert((file.data_len == DATA_LEN) && "Incorrect file data length.\n");
    printf(".");
    assert((file.targets_len == TARGETS_LEN) && "Incorrect file targets length.\n");
    printf(".");

    sequence[0] = 5;
    sequence[1] = 1;
    sequence[2] = 4;
    sequence[3] = 0;
    sequence[4] = 2;
    sequence[5] = 3;
    for(i=0; i<RECORDS / BATCH; i++) {
        create_batch_with_sequence_file(batch_data, batch_targets, &file, BATCH, i, sequence);
        for(j=0; j<(BATCH * DATA_LEN); j++) {
            assert((batch_data[j] == test_data[sequence[(i * BATCH) + (j % DATA_LEN)]][j / DATA_LEN]) &&
                   "Invalid data batch from sequence\n");
        }
        for(j=0; j<(BATCH * TARGETS_LEN); j++) {
            assert((batch_targets[j] == test_targets[sequence[(i * BATCH) + (j / TARGETS_LEN)]][j % TARGETS_LEN]) &&
                   "Invalid targets batch from sequence\n");
        }
    }
    printf(".");

    for(i=0; i<RECORDS / BATCH; i++) {
        create_batch_no_sequence_file(batch_data, batch_targets, &file, BATCH, i, 0, RECORDS);
        for(j=0; j<(BATCH * DATA_LEN); j++) {
            assert((batch_data[j] == test_data[(i * BATCH) + (j % DATA_LEN)][j / DATA_LEN]) &&
                   "Invalid data batch without sequence\n");
        }
        for(j=0; j<(BATCH * TARGETS_LEN); j++) {
            assert((batch_targets[j] == test_targets[(i * BATCH) + (j / TARGETS_LEN)][j % TARGETS_LEN]) &&
                   "Invalid targets batch without sequence\n");
        }
    }
    printf(".");

    ret = jcky_close_file(file);
    assert((ret == 0) && "Unable to close jockey file.\n");
    printf(".");

    printf("\nAll tests passed!\n");
    remove(FILENAME);

    for(i=0; i<RECORDS; i++) {
        free(test_data[i]);
        free(test_targets[i]);
    }
    free(test_data);
    free(test_targets);
    free(sequence);
    free(batch_data);
    free(batch_targets);

    return 0;
}
