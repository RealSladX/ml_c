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

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])
#define VALUE_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
float rand_float(void);
float sigmoidf(float x);

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Matrix;

Matrix mat_alloc(size_t rows, size_t cols);
void randomize_matrix(Matrix m, float low, float high);
Matrix get_matrix_row(Matrix m, size_t row);
void matrix_copy(Matrix dst, Matrix src);
void fill_matrix(Matrix m, float fill);
void dot_product(Matrix result, Matrix a, Matrix b);
void matrix_sum(Matrix result, Matrix a);
void print_matrix(Matrix m, const char *name, size_t padding);
void sigmoid_activation(Matrix m);
#define PRETTY_PRINT_MATRIX(m) print_matrix(m, #m, 0)

typedef struct {
  size_t num_layers;
  Matrix *ws;
  Matrix *bs;
  Matrix *as;
} NeuralNetwork;
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).num_layers]
NeuralNetwork nn_alloc(size_t *layers, size_t total_num_layers);
void nn_zero(NeuralNetwork nn);
void print_nn(NeuralNetwork nn, const char *name);
#define PRETTY_PRINT_NN(nn) print_nn(nn, #nn)
void randomize_nn(NeuralNetwork nn, float low, float high);
void forward(NeuralNetwork nn);
float calculate_cost(NeuralNetwork nn, Matrix train_input, Matrix train_output);
void finite_diff(NeuralNetwork nn, NeuralNetwork gradient, float epsilon,
                 Matrix train_input, Matrix train_output);
void backprop(NeuralNetwork nn, NeuralNetwork gradient, Matrix train_input,
              Matrix train_output);
void learn(NeuralNetwork nn, NeuralNetwork gradient, float learning_rate);
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
void print_matrix(Matrix m, const char *name, size_t padding) {
  printf("%*s%s = [\n", (int)padding, "", name);
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      printf("%*s    %f ", (int)padding, "", VALUE_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int)padding, "");
}

NeuralNetwork nn_alloc(size_t *layers, size_t total_num_layers) {
  NN_ASSERT(total_num_layers > 0);

  NeuralNetwork nn;
  nn.num_layers = total_num_layers - 1;

  nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.num_layers);
  NN_ASSERT(nn.ws != NULL);
  nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.num_layers);
  NN_ASSERT(nn.bs != NULL);
  nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.num_layers + 1));
  NN_ASSERT(nn.as != NULL);

  nn.as[0] = mat_alloc(1, layers[0]);
  for (size_t i = 1; i < total_num_layers; ++i) {
    nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, layers[i]);
    nn.bs[i - 1] = mat_alloc(1, layers[i]);
    nn.as[i] = mat_alloc(1, layers[i]);
  }
  return nn;
}

void nn_zero(NeuralNetwork nn) {
  for (size_t i = 0; i < nn.num_layers; ++i) {
    fill_matrix(nn.ws[i], 0);
    fill_matrix(nn.bs[i], 0);
    fill_matrix(nn.as[i], 0);
  }
  fill_matrix(nn.as[nn.num_layers], 0);
}

void print_nn(NeuralNetwork nn, const char *name) {
  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.num_layers; ++i) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    print_matrix(nn.ws[i], buf, 4);
    snprintf(buf, sizeof(buf), "bs%zu", i);
    print_matrix(nn.bs[i], buf, 4);
  }
  printf("]\n");
}

void randomize_nn(NeuralNetwork nn, float low, float high) {
  for (size_t i = 0; i < nn.num_layers; ++i) {
    randomize_matrix(nn.ws[i], low, high);
    randomize_matrix(nn.bs[i], low, high);
  }
}
void forward(NeuralNetwork nn) {
  for (size_t i = 0; i < nn.num_layers; ++i) {
    dot_product(nn.as[i + 1], nn.as[i], nn.ws[i]);
    matrix_sum(nn.as[i + 1], nn.bs[i]);
    sigmoid_activation(nn.as[i + 1]);
  }
}

float calculate_cost(NeuralNetwork nn, Matrix train_input,
                     Matrix train_output) {
  assert(train_input.rows == train_output.rows);
  assert(train_output.cols == NN_OUTPUT(nn).cols);
  size_t num_inputs = train_input.rows;

  float cost = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    Matrix expect_input = get_matrix_row(train_input, i);
    Matrix expect_output = get_matrix_row(train_output, i);

    matrix_copy(NN_INPUT(nn), expect_input);
    forward(nn);
    size_t num_outputs = train_output.cols;
    for (size_t j = 0; j < num_outputs; ++j) {
      float diff =
          VALUE_AT(NN_OUTPUT(nn), 0, j) - VALUE_AT(expect_output, 0, j);
      cost += diff * diff;
    }
  }
  return cost / num_inputs;
}
void backprop(NeuralNetwork nn, NeuralNetwork gradient, Matrix train_input,
              Matrix train_output) {
  NN_ASSERT(train_input.rows == train_output.rows);
  size_t n = train_input.rows;
  NN_ASSERT(NN_OUTPUT(nn).cols == train_output.cols);
  nn_zero(gradient);
  for (size_t i = 0; i < n; ++i) {
    matrix_copy(NN_INPUT(nn), get_matrix_row(train_input, i));
    forward(nn);
    for (size_t j = 0; j <= nn.num_layers; ++j) {
      fill_matrix(gradient.as[j], 0);
    }
    for (size_t j = 0; j < train_output.cols; ++j) {
      VALUE_AT(NN_OUTPUT(gradient), 0, j) =
          VALUE_AT(NN_OUTPUT(nn), 0, j) - VALUE_AT(train_output, i, j);
    }
    for (size_t l = nn.num_layers; l > 0; --l) {
      for (size_t a = 0; a < nn.as[l].cols; ++a) {
        float a_l = VALUE_AT(nn.as[l], 0, a);
        float da_l = VALUE_AT(gradient.as[l], 0, a);
        VALUE_AT(gradient.bs[l - 1], 0, a) += 2 * da_l * a_l * (1 - a_l);
        for (size_t pa = 0; pa < nn.as[l - 1].cols; ++pa) {
          float pa_l = VALUE_AT(nn.as[l - 1], 0, pa);
          float w = VALUE_AT(nn.ws[l - 1], pa, a);
          VALUE_AT(gradient.ws[l - 1], pa, a) +=
              2 * da_l * a_l * (1 - a_l) * pa_l;
          VALUE_AT(gradient.as[l - 1], 0, pa) += 2 * da_l * a_l * (1 - a_l) * w;
        }
      }
    }
  }
  for (size_t i = 0; i < gradient.num_layers; ++i) {
    for (size_t j = 0; j < gradient.ws[i].rows; ++j) {
      for (size_t k = 0; k < gradient.ws[i].cols; ++k) {
        VALUE_AT(gradient.ws[i], j, k) /= n;
      }
    }
    for (size_t j = 0; j < gradient.bs[i].rows; ++j) {
      for (size_t k = 0; k < gradient.bs[i].cols; ++k) {
        VALUE_AT(gradient.bs[i], j, k) /= n;
      }
    }
  }
}
void finite_diff(NeuralNetwork nn, NeuralNetwork gradient, float epsilon,
                 Matrix train_input, Matrix train_output) {
  float saved;
  float cost = calculate_cost(nn, train_input, train_output);
  for (size_t i = 0; i < nn.num_layers; ++i) {
    for (size_t j = 0; j < nn.ws[i].rows; ++j) {
      for (size_t k = 0; k < nn.ws[i].cols; ++k) {
        saved = VALUE_AT(nn.ws[i], j, k);
        VALUE_AT(nn.ws[i], j, k) += epsilon;
        VALUE_AT(gradient.ws[i], j, k) =
            (calculate_cost(nn, train_input, train_output) - cost) / epsilon;
        VALUE_AT(nn.ws[i], j, k) = saved;
      }
    }
    for (size_t j = 0; j < nn.bs[i].rows; ++j) {
      for (size_t k = 0; k < nn.bs[i].cols; ++k) {
        saved = VALUE_AT(nn.bs[i], j, k);
        VALUE_AT(nn.bs[i], j, k) += epsilon;
        VALUE_AT(gradient.bs[i], j, k) =
            (calculate_cost(nn, train_input, train_output) - cost) / epsilon;
        VALUE_AT(nn.bs[i], j, k) = saved;
      }
    }
  }
}

void learn(NeuralNetwork nn, NeuralNetwork gradient, float learning_rate) {
  for (size_t i = 0; i < nn.num_layers; ++i) {
    for (size_t j = 0; j < nn.ws[i].rows; ++j) {
      for (size_t k = 0; k < nn.ws[i].cols; ++k) {
        VALUE_AT(nn.ws[i], j, k) -=
            learning_rate * VALUE_AT(gradient.ws[i], j, k);
      }
    }
    for (size_t j = 0; j < nn.bs[i].rows; ++j) {
      for (size_t k = 0; k < nn.bs[i].cols; ++k) {
        VALUE_AT(nn.bs[i], j, k) -=
            learning_rate * VALUE_AT(gradient.bs[i], j, k);
      }
    }
  }
}

#endif // NEURALNETWORK_IMPLEMENTATION
