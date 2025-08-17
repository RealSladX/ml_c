#include <time.h>
#define NEURALNETWORK_IMPLEMENTATION
#include "neural_network.h"

#define BITS 2

int main(void) {
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

  size_t arch[] = {2 * BITS, 2 * BITS, BITS + 1};
  NeuralNetwork nn = nn_alloc(arch, ARRAY_LEN(arch));
  NeuralNetwork g = nn_alloc(arch, ARRAY_LEN(arch));
  randomize_nn(nn, 0, 1);
  PRETTY_PRINT_NN(nn);
  float rate = 1;
  size_t num_epochs = 1000 * 100;
  printf("Initial Cost: c = %f\n", calculate_cost(nn, ti, to));
  for (size_t i = 0; i < num_epochs; ++i) {
    backprop(nn, g, ti, to);
    learn(nn, g, rate);
  }
  printf("After %zu Epochs: c = %f\n", num_epochs, calculate_cost(nn, ti, to));
  for (size_t x = 0; x < n; ++x) {
    for (size_t y = 0; y < n; ++y) {
      printf("%zu + %zu = ", x, y);
      for (size_t j = 0; j < BITS; ++j) {
        VALUE_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
        VALUE_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
      }
      forward(nn);
      if (VALUE_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f) {
        printf("OVERFLOW\n");
      } else {
        size_t z = 0;
        for (size_t j = 0; j < BITS; ++j) {
          size_t bit = VALUE_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
          z |= bit << j;
        }
        printf("%zu\n", z);
      }
    }
  }
  return 0;
}
