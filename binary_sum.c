#include <threads.h>
#include <time.h>
#define NEURALNETWORK_IMPLEMENTATION
#include "neural_network.h"

#define BITS 5

int main(void) {
  srand(time(0));
  size_t n = (1 << BITS);
  size_t rows = n * n;
  Matrix ti = mat_alloc(rows, 2 * BITS);
  Matrix to = mat_alloc(rows, BITS + 1);
  for (size_t i = 0; i < ti.rows; ++i) {
    size_t x = i / n;
    size_t y = i % n;
    size_t z = x + y;
    size_t overflow = z >= n;
    for (size_t j = 0; j < BITS; ++j) {
      VALUE_AT(ti, i, j) = (x >> j) & 1;
      VALUE_AT(ti, i, j + BITS) = (y >> j) & 1;
      VALUE_AT(to, i, j) = (z >> j) & 1;
    }
    VALUE_AT(to, i, BITS) = overflow;
  }

  size_t arch[] = {2 * BITS, 2 * BITS + 1, BITS + 1};
  NeuralNetwork nn = nn_alloc(arch, ARRAY_LEN(arch));
  NeuralNetwork g = nn_alloc(arch, ARRAY_LEN(arch));
  randomize_nn(nn, 0, 1);
  float rate = 1;
  size_t num_epochs = 1000 * 10;
  printf("Initial Cost: c = %f\n", calculate_cost(nn, ti, to));
  for (size_t i = 0; i < num_epochs; ++i) {
    printf("Training...%zu\n", i);
    printf("\x1b[1F");
    printf("\x1b[1K");
    backprop(nn, g, ti, to);
    learn(nn, g, rate);
    // thrd_sleep(&(struct timespec){.tv_sec = 0.1}, NULL);
  }
  printf("After %zu Epochs: c = %f\n", num_epochs, calculate_cost(nn, ti, to));
  float fails = 0;
  for (size_t x = 0; x < n; ++x) {
    for (size_t y = 0; y < n; ++y) {
      size_t z = x + y;
      for (size_t j = 0; j < BITS; ++j) {
        VALUE_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
        VALUE_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
      }
      forward(nn);
      if (VALUE_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f) {
        if (z < n) {
          printf("%zu + %zu = (OVERFLOW <> %zu)\n", x, y, z);
          fails++;
        }
      } else {
        size_t a = 0;
        for (size_t j = 0; j < BITS; ++j) {
          size_t bit = VALUE_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
          a |= bit << j;
        }
        if (z != a) {
          printf("%zu + %zu = (%zu <> %zu)\n", x, y, z, a);
          fails++;
        }
      }
    }
  }
  if (fails == 0)
    printf("Ayyy No Errors!!\n");
  else {
    float precision = (rows - fails) / rows;
    printf("Precision: %.7f", precision);
  }
  return 0;
}
