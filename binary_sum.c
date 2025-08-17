#include <time.h>
#define NEURALNETWORK_IMPLEMENTATION
#include "neural_network.h"

#define BITS 2

// void test_dot_product(Matrix result, Matrix a, Matrix b) {
//   NN_ASSERT(a.cols == b.rows);
//   size_t n = a.cols;
//   NN_ASSERT(result.rows == a.rows);
//   NN_ASSERT(result.cols == b.cols);
//   for (size_t i = 0; i < result.rows; ++i) {
//     for (size_t j = 0; j < result.cols; ++j) {
//       VALUE_AT(result, i, j) = 0;
//       for (size_t k = 0; k < n; ++k) {
//         VALUE_AT(result, i, j) += (VALUE_AT(a, i, k) * VALUE_AT(b, k, j));
//       }
//     }
//   }
// }
//
// void test_forward(NeuralNetwork nn) {
//   for (size_t i = 0; i < nn.num_layers; ++i) {
//     test_dot_product(nn.as[i + 1], nn.as[i], nn.ws[i]);
//     matrix_sum(nn.as[i + 1], nn.bs[i]);
//     sigmoid_activation(nn.as[i + 1]);
//   }
// }
// float test_calculate_cost(NeuralNetwork nn, Matrix train_input,
//                           Matrix train_output) {
//   assert(train_input.rows == train_output.rows);
//   assert(train_output.cols == NN_OUTPUT(nn).cols);
//   size_t num_inputs = train_input.rows;
//
//   float cost = 0;
//   for (size_t i = 0; i < num_inputs; ++i) {
//     Matrix expect_input = get_matrix_row(train_input, i);
//     Matrix expect_output = get_matrix_row(train_output, i);
//
//     matrix_copy(NN_INPUT(nn), expect_input);
//     test_forward(nn);
//     size_t num_outputs = train_output.cols;
//     for (size_t j = 0; j < num_outputs; ++j) {
//       float diff =
//           VALUE_AT(NN_OUTPUT(nn), 0, j) - VALUE_AT(expect_output, 0, j);
//       cost += diff * diff;
//     }
//   test_forward(nn);
//   }
//   return cost / num_inputs;
// }

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
      if (overflow) {
        VALUE_AT(to, i, j) = 0;
      } else {
        VALUE_AT(to, i, j) = (z >> j) & 1;
      }
    }
    VALUE_AT(to, i, BITS) = overflow;
  }

  size_t arch[] = {2 * BITS, BITS + 1};
  NeuralNetwork nn = nn_alloc(arch, ARRAY_LEN(arch));
  NeuralNetwork g = nn_alloc(arch, ARRAY_LEN(arch));
  randomize_nn(nn, 0, 1);
  PRETTY_PRINT_NN(nn);
  printf("c = %f\n", calculate_cost(nn, ti, to));

  return 0;
}
