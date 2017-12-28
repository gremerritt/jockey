#ifndef TIMINGHELPERS_H
#define TIMINGHELPERS_H


#include <time.h>

#include "constants.h"


struct timespec diff_time(struct timespec start, struct timespec end);

#ifdef JCKY_TIMING
#define INIT_TIMERS jcky_timer *timers = malloc(cli.epochs * sizeof(jcky_timer));
#define GET_TIMER jcky_timer timer;

#define START_TIME_EPOCH clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.epoch_start));
#define END_TIME_EPOCH \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.epoch_end));\
    timer.epoch = diff_time(timer.epoch_start, timer.epoch_end);

#define START_TIME_COPY clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.copy_start));
#define END_TIME_COPY \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.copy_end));\
    timer.copy = diff_time(timer.copy_start, timer.copy_end);

#define START_TIME_SHUFFLE clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.shuffle_start));
#define END_TIME_SHUFFLE \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.shuffle_end));\
    timer.shuffle = diff_time(timer.shuffle_start, timer.shuffle_end);

#define START_TIME_TRAINING clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.training_start));
#define END_TIME_TRAINING \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.training_end));\
    timer.training = diff_time(timer.training_start, timer.training_end);

#define START_TIME_TRAINING_BATCH clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.training_batch_start));
#define END_TIME_TRAINING_BATCH \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.training_batch_end));\
    timer.training_batch = diff_time(timer.training_batch_start, timer.training_batch_end);

#define START_TIME_TRAINING_RUN clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.training_run_start));
#define END_TIME_TRAINING_RUN \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.training_run_end));\
    timer.training_run = diff_time(timer.training_run_start, timer.training_run_end);

#define START_TIME_SYNC clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.sync_start));
#define END_TIME_SYNC \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.sync_end));\
    timer.sync = diff_time(timer.sync_start, timer.sync_end);

#define START_TIME_TESTING clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.testing_start));
#define END_TIME_TESTING \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.testing_end));\
    timer.testing = diff_time(timer.testing_start, timer.testing_end);

#define START_TIME_TESTING_BATCH clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.testing_batch_start));
#define END_TIME_TESTING_BATCH \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.testing_batch_end));\
    timer.testing_batch = diff_time(timer.testing_batch_start, timer.testing_batch_end);

#define START_TIME_TESTING_RUN clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.testing_run_start));
#define END_TIME_TESTING_RUN \
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &(timer.testing_run_end));\
    timer.testing_run = diff_time(timer.testing_run_start, timer.testing_run_end);

#define WRITE_TIME \
    if (mpi_manager.master) { \
        if (!(cli.no_timing)) write_timing_record(epoch, &timer); \
        else timers[epoch] = timer; \
    }
#define WRITE_TIMES \
    if (mpi_manager.master && cli.no_timing) write_timing(cli.epochs, timers);
#define FREE_TIMERS free(timers);

#else
#define INIT_TIMERS
#define GET_TIMER
#define START_TIME_EPOCH
#define END_TIME_EPOCH
#define START_TIME_COPY
#define END_TIME_COPY
#define START_TIME_SHUFFLE
#define END_TIME_SHUFFLE
#define START_TIME_TRAINING
#define END_TIME_TRAINING
#define START_TIME_TRAINING_BATCH
#define END_TIME_TRAINING_BATCH
#define START_TIME_TRAINING_RUN
#define END_TIME_TRAINING_RUN
#define START_TIME_SYNC
#define END_TIME_SYNC
#define START_TIME_TESTING
#define END_TIME_TESTING
#define START_TIME_TESTING_BATCH
#define END_TIME_TESTING_BATCH
#define START_TIME_TESTING_RUN
#define END_TIME_TESTING_RUN
#define RECORD_TIME
#define FREE_TIMERS
#define WRITE_TIME
#define WRITE_TIMES
#endif

typedef struct jcky_timer {
    struct timespec epoch, epoch_start, epoch_end;
    struct timespec copy, copy_start, copy_end;
    struct timespec shuffle, shuffle_start, shuffle_end;
    struct timespec training, training_start, training_end;
    struct timespec training_batch, training_batch_start, training_batch_end;
    struct timespec training_run, training_run_start, training_run_end;
    struct timespec sync, sync_start, sync_end;
    struct timespec testing, testing_start, testing_end;
    struct timespec testing_batch, testing_batch_start, testing_batch_end;
    struct timespec testing_run, testing_run_start, testing_run_end;
} jcky_timer;

void write_timing(unsigned short int epochs, jcky_timer *timers);
void write_timing_record(unsigned short int epoch, jcky_timer *timer);


#endif
