#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <math.h>
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

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Matrix;
#define VALUE_AT(m, i, j) m.es[(i) * (m).stride + (j)]

float rand_float(void);
float sigmoidf(float x);
Matrix mat_alloc(size_t rows, size_t cols);
void randomize_matrix(Matrix m, float low, float high);
Matrix get_matrix_row(Matrix m, size_t row);

void matrix_copy(Matrix dst, Matrix src);
void fill_matrix(Matrix m, float fill);
void dot_product(Matrix result, Matrix a, Matrix b);
void matrix_sum(Matrix result, Matrix a);
void print_matrix(Matrix m, const char *name);
void sigmoid_activation(Matrix m);
#define PRETTY_PRINT_MATRIX(m) print_matrix(m, #m)
#endif // NEURAL_NETWORK_H

#ifdef NEURALNETWORK_IMPLEMENTATION
float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
float rand_float(void) { return (float)rand() / (float)RAND_MAX; }
Matrix mat_alloc(size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
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
Matrix get_matrix_row(Matrix m, size_t row) {
  return (Matrix){
      .rows = 1,
      .cols = m.cols,
      .stride = m.stride,
      .es = &VALUE_AT(m, row, 0),
  };
}
void matrix_copy(Matrix dst, Matrix src) {
  NN_ASSERT(dst.rows == src.rows);
  NN_ASSERT(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      VALUE_AT(dst, i, j) = VALUE_AT(src, i, j);
    }
  }
}
void dot_product(Matrix result, Matrix a, Matrix b) {
  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NN_ASSERT(result.rows == a.rows);
  NN_ASSERT(result.cols == b.cols);
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
void sigmoid_activation(Matrix m) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      VALUE_AT(m, i, j) = sigmoidf(VALUE_AT(m, i, j));
    }
  }
}
void print_matrix(Matrix m, const char *name) {
  printf("%s = [\n", name);
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      printf("    %f ", VALUE_AT(m, i, j));
    }
    printf("\n");
  }
  printf("]\n");
}

#endif // NEURALNETWORK_IMPLEMENTATION
