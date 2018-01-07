#include <stdio.h>

#include "constants.h"
#include "timing_helpers.h"


inline void write_headers(FILE *stream) {
    fprintf(stream, "epoch,");
    fprintf(stream, "epoch_time,");
    fprintf(stream, "copy_time,");
    fprintf(stream, "shuffle_time,");
    fprintf(stream, "training_time,");
    fprintf(stream, "training_batch_time,");
    fprintf(stream, "training_run_time,");
    fprintf(stream, "sync_time,");
    fprintf(stream, "testing_time,");
    fprintf(stream, "testing_batch_time,");
    fprintf(stream, "testing_run_time");
    fprintf(stream, "\n");
}


inline void write_record(unsigned short int epoch, jcky_timer *timer, FILE *stream) {
    fprintf(stream, "%i,", epoch+1);
    fprintf(stream, "%i.%i,", timer->epoch.tv_sec, timer->epoch.tv_nsec);
    fprintf(stream, "%i.%i,", timer->copy.tv_sec, timer->copy.tv_nsec);
    fprintf(stream, "%i.%i,", timer->shuffle.tv_sec, timer->shuffle.tv_nsec);
    fprintf(stream, "%i.%i,", timer->training.tv_sec, timer->training.tv_nsec);
    fprintf(stream, "%i.%i,", timer->training_batch.tv_sec, timer->training_batch.tv_nsec);
    fprintf(stream, "%i.%i,", timer->training_run.tv_sec, timer->training_run.tv_nsec);
    fprintf(stream, "%i.%i,", timer->sync.tv_sec, timer->sync.tv_nsec);
    fprintf(stream, "%i.%i,", timer->testing.tv_sec, timer->testing.tv_nsec);
    fprintf(stream, "%i.%i,", timer->testing_batch.tv_sec, timer->testing_batch.tv_nsec);
    fprintf(stream, "%i.%i", timer->testing_run.tv_sec, timer->testing_run.tv_nsec);
    fprintf(stream, "\n");
}


void write_timing(unsigned short int epochs, jcky_timer *timers) {
    FILE *stream;
    unsigned short int i;

    if (epochs > 0) {
        stream = fopen(JCKY_TIMING_FILENAME, "w+");
        if (stream != NULL) {
            write_headers(stream);
            for(i=0; i<epochs; i++) write_record(i, &timers[i], stream);
            fclose(stream);
        }
        else {
            printf("Unabled to write timing file %s\n.", JCKY_TIMING_FILENAME);
        }
    }
}


void write_timing_record(unsigned short int epoch, jcky_timer *timer) {
    char *mode = (epoch == 0) ? "w+" : "a+";
    FILE *stream = fopen(JCKY_TIMING_FILENAME, mode);
    if (stream != NULL) {
        if (epoch == 0) write_headers(stream);
        write_record(epoch, timer, stream);
        fclose(stream);
    }
    else {
        printf("Unabled to write timing file %s\n.", JCKY_TIMING_FILENAME);
    }
}
