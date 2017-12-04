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

### THIS PROJECT IS UNDER ACTIVE DEVELOPMENT -- CHECK BACK REGULARLY FOR UPDATES

The goal of this project is to create a lightning-fast neural network that can solve big problems. There are libraries available to get machine learning solutions up and running quickly (think Python with TensorFlow). **This is not the goal**. This will likely require some actual knowledge of multithreaded environments and the internals of how neural networks work. Specifically, the project is:
  - **Written in C**. It should be written as close to the hardware as possible.
  - **Parallelized**. Currently the plan is for MPI, so that this can run *fast* on large clusters.
  - **Easy to interface with**. You need to write your own data loader and send it to the neural network in a defined format.
  - **Easy to customize**. Want to customize the neural network layout? That should be easy. Want a different activation function? That should be easy, too.


### This project is based on https://github.com/gremerritt/neural-net-OCR
This project implements OCR with a Neural Network.

Sample MNIST loader by Nuri Park - https://github.com/projectgalateia/mnist
