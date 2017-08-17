CC= gcc-4.9
MPICC= mpicc
CFLAGS= -g -O3
OPENMPFLAG= -fopenmp
LIBS=
UNAME = $(shell uname)
ifneq ($(UNAME),Darwin)
  LIBS += -lrt -lm
endif
EXEC = main

main: main.o neural_net.o matrix_helpers.o randomizing_helpers.o
	$(CC) $(CFLAGS) $(OPENMPFLAG) main.o neural_net.o matrix_helpers.o randomizing_helpers.o -o $(EXEC) $(LIBS)

main.o: main.c
	$(CC) $(CFLAGS) $(OPENMPFLAG) -c main.c $(LIBS)

neural_net.o: neural_net.c neural_net.h
	$(CC) $(CFLAGS) $(OPENMPFLAG) -c neural_net.c $(LIBS)

matrix_helpers.o: matrix_helpers.c matrix_helpers.h
	$(CC) $(CFLAGS) $(OPENMPFLAG) -c matrix_helpers.c $(LIBS)

randomizing_helpers.o: randomizing_helpers.c randomizing_helpers.c
	$(CC) $(CFLAGS) $(OPENMPFLAG) -c randomizing_helpers.c $(LIBS)

mpi: main_mpi.o neural_net_mpi.o helpers_mpi.o matrix_helpers_mpi.o randomizing_helpers_mpi.o mpi_helper.o
	$(MPICC) $(CFLAGS) $(LIBS) main_mpi.o neural_net_mpi.o helpers_mpi.o matrix_helpers_mpi.o randomizing_helpers_mpi.o mpi_helper.o -o $(EXEC)

main_mpi.o: main.c
	$(MPICC) $(CFLAGS) -c main.c $(LIBS) -o main_mpi.o

neural_net_mpi.o: neural_net.c neural_net.h
	$(CC) $(CFLAGS) -c neural_net.c $(LIBS) -o neural_net_mpi.o

matrix_helpers_mpi.o: matrix_helpers.c matrix_helpers.h
	$(CC) $(CFLAGS) -c matrix_helpers.c $(LIBS) -o matrix_helpers_mpi.o

randomizing_helpers_mpi.o: randomizing_helpers.c randomizing_helpers.h
	$(CC) $(CFLAGS) -c randomizing_helpers.c $(LIBS) -o randomizing_helpers_mpi.o

mpi_helper_mpi.o: mpi_helper.c mpi_helper.h
	$(CC) $(CFLAGS) -c mpi_helper.c $(LIBS) -o mpi_helper.o

helpers_mpi.o: helpers.c helpers.h
	$(CC) $(CFLAGS) -c helpers.c $(LIBS) -o helpers_mpi.o

clean:
	rm -r *.o *.dSYM $(EXEC) 2> /dev/null
