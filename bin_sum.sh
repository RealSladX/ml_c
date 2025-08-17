
#!/bin/sh
set -xe

clang -Wall -Wextra -o bin_sum binary_sum.c -lm -lc
