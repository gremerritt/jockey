CC= gcc-4.9
MPICC= mpicc
CFLAGS= -g -O3
OPENMPFLAG= -fopenmp
LIBS=
UNAME = $(shell uname)
ifneq ($(UNAME),Darwin)
  LIBS += -lrt -lm
endif
EXEC = jockey
TEST_EXEC = test_jockey
MODULES = neural_net.o helpers.o matrix_helpers.o randomizing_helpers.o mpi_helper.o file_helpers.o batch.o hooks.o timing_helpers.o model_helpers.o

mpi: main.o $(MODULES)
	$(MPICC) $(CFLAGS) $(LIBS) main.o $(MODULES) -o $(EXEC)

main.o: lib/main.c
	$(MPICC) $(CFLAGS) -c lib/main.c $(LIBS) -o main.o

neural_net.o: lib/neural_net.c lib/neural_net.h
	$(CC) $(CFLAGS) -c lib/neural_net.c $(LIBS) -o neural_net.o

matrix_helpers.o: lib/matrix_helpers.c lib/matrix_helpers.h
	$(CC) $(CFLAGS) -c lib/matrix_helpers.c $(LIBS) -o matrix_helpers.o

randomizing_helpers.o: lib/randomizing_helpers.c lib/randomizing_helpers.h
	$(CC) $(CFLAGS) -c lib/randomizing_helpers.c $(LIBS) -o randomizing_helpers.o

mpi_helper.o: lib/mpi_helper.c lib/mpi_helper.h
	$(CC) $(CFLAGS) -c lib/mpi_helper.c $(LIBS) -o mpi_helper.o

helpers.o: lib/helpers.c lib/helpers.h
	$(CC) $(CFLAGS) -c lib/helpers.c $(LIBS) -o helpers.o

file_helpers.o: lib/file_helpers.c lib/helpers.h
	$(CC) $(CFLAGS) -c lib/file_helpers.c $(LIBS) -o file_helpers.o

batch.o: lib/batch.c lib/batch.h
	$(CC) $(CFLAGS) -c lib/batch.c $(LIBS) -o batch.o

timing_helpers.o: lib/timing_helpers.c lib/timing_helpers.h
	$(CC) $(CFLAGS) -c lib/timing_helpers.c $(LIBS) -o timing_helpers.o

model_helpers.o: lib/model_helpers.c lib/model_helpers.h
	$(CC) $(CFLAGS) -c lib/model_helpers.c $(LIBS) -o model_helpers.o

hooks.o: lib/hooks.c lib/hooks.h
	$(CC) $(CFLAGS) -c lib/hooks.c $(LIBS) -o hooks.o

test: test.o $(MODULES)
	$(MPICC) $(CFLAGS) $(LIBS) test.o $(MODULES) -o $(TEST_EXEC)

test.o: test/test.c
	$(MPICC) $(CFLAGS) -c test/test.c $(LIBS) -o test.o

clean:
	rm -r *.o *.dSYM *.swp $(EXEC) $(TEST_EXEC) 2> /dev/null
