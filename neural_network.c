#include <time.h>
#define NEURALNETWORK_IMPLEMENTATION
#include "neural_network.h"

int main(void) {
  srand(time(0));
  Matrix a = mat_alloc(2, 2);
  Matrix b = mat_alloc(2, 2);
  fill_matrix(a, 1);
  printf("Matrix A:\n");
  print_matrix(a);
  fill_matrix(b, 1);
  printf("Matrix B:\n");
  print_matrix(b);
  Matrix result = mat_alloc(2, 2);
  dot_product(result, a, b);
  print_matrix(result);
  return 0;
}
