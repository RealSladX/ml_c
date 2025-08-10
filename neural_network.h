#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <stddef.h>
#include <stdio.h>
#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

// float d_xor[] = {
//  0, 0, 0,
//  0, 1, 1,
//  1, 0, 1,
//  1, 1, 0,
// };

typedef struct {
  size_t rows;
  size_t cols;
  float *es;
} Matrix;
#define VALUE_AT(m, i, j) m.es[(i) * (m).cols + (j)]

float rand_float(void);

Matrix mat_alloc(size_t rows, size_t cols);
void randomize_matrix(Matrix m, float low, float high);
void fill_matrix(Matrix m, float fill);
void dot_product(Matrix result, Matrix a, Matrix b);
void matrix_sum(Matrix result, Matrix a);
void print_matrix(Matrix m);
#endif // NEURAL_NETWORK_H

#ifdef NEURALNETWORK_IMPLEMENTATION
float rand_float(void) { return (float)rand() / (float)RAND_MAX; }
Matrix mat_alloc(size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);
  NN_ASSERT(m.es != NULL);
  return m;
}
void fill_matrix(Matrix m, float fill) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      VALUE_AT(m, i, j) = fill;
    }
  }
}
void randomize_matrix(Matrix m, float low, float high) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      VALUE_AT(m, i, j) = rand_float() * (high - low) + low;
    }
  }
}
void dot_product(Matrix result, Matrix a, Matrix b) {
  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NN_ASSERT(result.rows == a.rows);
  NN_ASSERT(result.cols == a.cols);
  for (size_t i = 0; i < result.rows; ++i) {
    for (size_t j = 0; j < result.cols; ++j) {
      VALUE_AT(result, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        VALUE_AT(result, i, j) += (VALUE_AT(a, i, k) * VALUE_AT(b, k, j));
      }
    }
  }
}
void matrix_sum(Matrix result, Matrix a) {
  NN_ASSERT(result.rows == a.rows);
  NN_ASSERT(result.cols == a.cols);
  for (size_t i = 0; i < result.rows; ++i) {
    for (size_t j = 0; j < result.cols; ++j) {
      VALUE_AT(result, i, j) += VALUE_AT(a, i, j);
    }
  }
}
void print_matrix(Matrix m) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      printf("%f ", VALUE_AT(m, i, j));
    }
    printf("\n");
  }
}

#endif // NEURALNETWORK_IMPLEMENTATION
