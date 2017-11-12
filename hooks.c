#include <stdio.h>
#include "file_helpers.h"

// ---------------------------------------------------------------- //
// Description:                                                     //
//      This code will be executed when using the '--write' flag,   //
//      and is used to create jockey files. You should create four  //
//      two-dimensional arrays, each of the same length, as         //
//      follows:                                                    //
//          nn_type (double or float) **training_data               //
//          nn_type (double or float) **training_targets            //
//          nn_type (double or float) **testing_data                //
//          nn_type (double or float) **testing_targets             //
//      Each element of the data array is an input array to the     //
//      neural network, and the corresponding element of the        //
//      targets array is the expected output of the neural network. //
//      You should then call `jcky_write_file` with the following   //
//      parameters:                                                 //
//          data                                                    //
//          targets                                                 //
//          number of records (e.g. length of training_* arrays)    //
//          data length (e.g. length of each array in *_data)       //
//          targets length (e.g. length of each array in *_targets) //
//          filename                                                //
//                                                                  //
//      Below is an example for the MNIST dataset.                  //
// ---------------------------------------------------------------- //
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
char write_file() {
    int i, j;
    char ret;
    mnist_data *mnist_training_data;
    mnist_data *mnist_testing_data;
    unsigned int training_cnt, testing_cnt;
    nn_type **training_data, **training_targets;
    nn_type **testing_data, **testing_targets;

    printf("Loading training image set... ");
	ret = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &mnist_training_data, &training_cnt);
	if (ret) {
		printf("An error occured: %d\n", ret);
		printf("Make sure image files (*-ubyte) are in the current directory.\n");
		return ret;
	}
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", training_cnt);
	}

	printf("\nLoading test image set... ");
	ret = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &mnist_testing_data, &testing_cnt);
	if (ret) {
		printf("An error occured: %d\n", ret);
		printf("Make sure image files (*-ubyte) are in the current directory.\n");
		return ret;
	}
	else {
		printf("Success!\n");
		printf("  Image count: %d\n", testing_cnt);
	}

    training_data = malloc( training_cnt * sizeof( nn_type* ));
    training_targets = malloc( training_cnt * sizeof( nn_type* ));
    for(i=0; i<training_cnt; i++) {
        training_data[i] = mnist_training_data[i].data;
        training_targets[i] = (nn_type *)malloc( 10 * sizeof( nn_type ) );
        for(j=0; j<10; j++) training_targets[i][j] = (nn_type)((j == mnist_training_data[i].label) ? 1.0 : 0.0);
    }

    testing_data = malloc( testing_cnt * sizeof( nn_type* ));
    testing_targets = malloc( testing_cnt * sizeof( nn_type* ));
    for(i=0; i<testing_cnt; i++) {
        testing_data[i] = mnist_testing_data[i].data;
        testing_targets[i] = (nn_type *)malloc( 10 * sizeof( nn_type ) );
        for(j=0; j<10; j++) testing_targets[i][j] = (nn_type)((j == mnist_testing_data[i].label) ? 1.0 : 0.0);
    }

    printf("\nWriting training file... ");
    ret = jcky_write_file(training_data, training_targets, training_cnt, 28*28, 10, "training.jockey");
    if (ret) printf("Failed.\n");
    else printf("Success!\n");

    printf("Writing testing file... ");
    ret = jcky_write_file(testing_data, testing_targets, testing_cnt, 28*28, 10, "testing.jockey");
    if (ret) printf("Failed.\n");
    else printf("Success!\n");

    for(i=0; i<training_cnt; i++) {
        free(training_targets[i]);
    }
    for(i=0; i<testing_cnt; i++) {
        free(testing_targets[i]);
    }
    free(training_data);
    free(testing_data);

    return ret;
}
