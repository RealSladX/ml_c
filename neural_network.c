#include <time.h>
#define NEURALNETWORK_IMPLEMENTATION
#include "neural_network.h"

float train_xor[] = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0};
float train_or[] = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0};
int main(void) {
  srand(time(0));

  float *train_data = train_xor;
  size_t n = 4;
  size_t stride = 3;
  Matrix train_input = {
      .rows = n,
      .cols = 2,
      .stride = stride,
      .es = train_data,
  };

  Matrix train_output = {
      .rows = n,
      .cols = 1,
      .stride = stride,
      .es = train_data + 2,
  };

  size_t layers[] = {2, 2, 1};
  NeuralNetwork nn = nn_alloc(layers, ARRAY_LEN(layers));
  NeuralNetwork gradient = nn_alloc(layers, ARRAY_LEN(layers));
  randomize_nn(nn, 0, 1);

  float learning_rate = 1;

  printf("Initial cost = %f\n", calculate_cost(nn, train_input, train_output));
  size_t num_epochs = 1000 * 10;
  for (size_t i = 0; i < num_epochs; ++i) {
    backprop(nn, gradient, train_input, train_output);
    learn(nn, gradient, learning_rate);
  }
  printf("After %zu Epochs: cost = %f\n", num_epochs,
         calculate_cost(nn, train_input, train_output));
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      VALUE_AT(NN_INPUT(nn), 0, 0) = i;
      VALUE_AT(NN_INPUT(nn), 0, 1) = j;
      forward(nn);
      printf("%zu ^ %zu = %f\n", i, j, VALUE_AT(NN_OUTPUT(nn), 0, 0));
    }
  }
  return 0;
}
