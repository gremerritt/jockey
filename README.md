```
            .''       _            __
  ._.-.___.' (`\     (_)___  _____/ /_____  __  __
 //(        ( `'    / / __ \/ ___/ //_/ _ \/ / / /
'/ )\ ).__. )      / / /_/ / /__/ ,< /  __/ /_/ /
' <' `/ ._/'/   __/ /\____/\___/_/|_|\___/\__, /
   ` /     /   /___/                     /____/
```
# Jockey

## A small, fast neural network that rides on top of big data.

The goal of this project is to create a _fast_ neural network that can solve big problems. There are libraries available to get machine learning solutions up and running quickly (think Python with TensorFlow). **This is not the goal**. Using Jockey will likely require some knowledge of multithreaded environments and the internals of how neural networks work. Specifically, the project is:
  - **Written in C**. It should be written as close to the hardware as possible.
  - **Parallelized**. Jockey is currently written with native MPI support, so that this can run _fast_ on large clusters.
  - **Straightforward to interface with**. You need to write your own data loader and send it to the neural network in a defined format.
  - **Easy to customize**. _This has only been partially achieved._ Want to customize the neural network layout? That should be easy. Want a different activation function? That should be easy, too.


## Compiling + Running Jockey

  - Compile by running:
  ```
  $ make mpi
  ```
  By default this assumes there is a compiler called `gcc-4.9`. Update the `CC` variable in the `Makefile` to update the compiler to be used.
  - Edit the `write_file()` and `get_score()` functions in `hooks.c` to handle your data. The file has more details about these methods. This source code comes with functions written to handle the MNIST data set for OCR.
  - Write training and testing data files by running
  ```
  $ jockey --write
  ```
  - Run serially by running
  ```
  $ jockey --train training.jockey --test testing.jockey -v
  ```
  - Otherwise, run using `mpirun` (Jockey ran been tested using Open MPI)
  ```
  $ mpirun -np 4 jockey --train training.jockey --test testing.jockey -v
  ```
  - There are a number of options to use when running Jockey. Use `jockey --help` to see the options.
  - Example output is:
  ```
  $ mpirun -np 4 jockey --train training.jockey --test testing.jockey -e 3 --no-timing -v
                .''       _            __
      ._.-.___.' (`\     (_)___  _____/ /_____  __  __
     //(        ( `'    / / __ \/ ___/ //_/ _ \/ / / /
'/ )\ ).__. )      / / /_/ / /__/ ,< /  __/ /_/ /
' <' `/ ._/'/   __/ /\____/\___/_/|_|\___/\__, /
       ` /     /   /___/                     /____/ v0.0.0

  Reporting from processor Gregs-MacBook-Pro.local, rank 0 of 4
  Reporting from processor Gregs-MacBook-Pro.local, rank 2 of 4
  Reporting from processor Gregs-MacBook-Pro.local, rank 3 of 4
  Reporting from processor Gregs-MacBook-Pro.local, rank 1 of 4

  Creating sample managers:
        Handling 60000 total training samples.
        Process 0 will handle 15000 training samples (3000 batches)
        Handling 10000 total testing samples.
        Process 0 will handle 2500 testing samples (500 batches)
        Process 1 will handle 15000 training samples (3000 batches)
        Process 1 will handle 2500 testing samples (500 batches)
        Process 2 will handle 15000 training samples (3000 batches)
        Process 2 will handle 2500 testing samples (500 batches)
        Process 3 will handle 15000 training samples (3000 batches)
        Process 3 will handle 2500 testing samples (500 batches)

  --------------------------------------
  Configuration:
        Total Layers:           4
        Hidden Layers:          2
        Inputs:                 784
        Outputs:                10
        Nodes in Hidden Layers: 60
        Batch Size:             5
        Learning Rate:          1.500000
        Initialization Seed:    1515351426
        Initialization File:    N/A
        Epochs:                 3
  --------------------------------------

  Epoch 0
        Training - 100%
        Testing  - 100%
        Total Score: 8512.000000
  Epoch 1
        Training - 100%
        Testing  - 100%
        Total Score: 9064.000000
  Epoch 2
        Training - 100%
        Testing  - 100%
        Total Score: 9207.000000
  ```


#### Sample MNIST loader by Nuri Park https://github.com/projectgalateia/mnist
