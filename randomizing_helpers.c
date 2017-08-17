#include <stdlib.h>
#include <time.h>
#include <math.h>

int generate_random_int() {
  // we're generating two random ints using the Rand() method, then putting
  // the high bits of one of them into the low bits of the other. The reason
  // for this is that the rand() function does a better job of generating
  // statistically random digits in the higher bits.
  int random_upper = rand();
  int random_lower = rand();

  // shift the upper half of the bits to the lower half
  random_lower >>= (int)((sizeof(int) / 2) * 8);

  // zero out the lower bits
  random_upper &= 0xFFFF0000;

  // combine the two
  random_upper |= random_lower;

  return random_upper;
}

int generate_guassian_distribution(double *numbers, int size, int seed) {
    if (seed == -1) {
        seed = time(NULL);
    }

    srand(seed);

    int i;
    for (i=0; i<size; i+=2) {
        // Generate two (pseudo)random variables in uniform distribution (1,0)
        // and use the Box-Muller method to get numbers in a Gaussian distribution.
        int U_upper = generate_random_int();
        int V_upper = generate_random_int();

        double U = (double)U_upper / RAND_MAX;
        double V = (double)V_upper / RAND_MAX;

        double X = sqrt(-2 * log(U)) * cos(2 * M_PI * V);
        double Y = sqrt(-2 * log(U)) * sin(2 * M_PI * V);

        numbers[i]   = X;
        if (i+1 < size) {
            numbers[i+1] = Y;
        }
    }

    return seed;
}

void shuffle(unsigned int *array, int size) {
  // shuffling with Knuth-Fisher-Yates algorithm
  int i;
  for (i = size - 1; i > 0; i--) {
    int n = generate_random_int() % (i+1);
    if (i != n) {
      int tmp = array[i];
      array[i] = array[n];
      array[n] = tmp;
    }
  }
}
