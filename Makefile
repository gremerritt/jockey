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
TEST_EXEC = test
MODULES = neural_net.o helpers.o matrix_helpers.o randomizing_helpers.o mpi_helper.o file_helpers.o batch.o hooks.o

mpi: main.o $(MODULES)
	$(MPICC) $(CFLAGS) $(LIBS) main.o $(MODULES) -o $(EXEC)

main.o: main.c
	$(MPICC) $(CFLAGS) -c main.c $(LIBS) -o main.o

neural_net.o: neural_net.c neural_net.h
	$(CC) $(CFLAGS) -c neural_net.c $(LIBS) -o neural_net.o

matrix_helpers.o: matrix_helpers.c matrix_helpers.h
	$(CC) $(CFLAGS) -c matrix_helpers.c $(LIBS) -o matrix_helpers.o

randomizing_helpers.o: randomizing_helpers.c randomizing_helpers.h
	$(CC) $(CFLAGS) -c randomizing_helpers.c $(LIBS) -o randomizing_helpers.o

mpi_helper.o: mpi_helper.c mpi_helper.h
	$(CC) $(CFLAGS) -c mpi_helper.c $(LIBS) -o mpi_helper.o

helpers.o: helpers.c helpers.h
	$(CC) $(CFLAGS) -c helpers.c $(LIBS) -o helpers.o

file_helpers.o: file_helpers.c helpers.h
	$(CC) $(CFLAGS) -c file_helpers.c $(LIBS) -o file_helpers.o

batch.o: batch.c batch.h
	$(CC) $(CFLAGS) -c batch.c $(LIBS) -o batch.o

test: test.o $(MODULES)
	$(MPICC) $(CFLAGS) $(LIBS) test.o $(MODULES) -o $(TEST_EXEC)

test.o: test.c
	$(MPICC) $(CFLAGS) -c test.c $(LIBS) -o test.o

clean:
	rm -r *.o *.dSYM *.swp $(EXEC) $(TEST_EXEC) 2> /dev/null
