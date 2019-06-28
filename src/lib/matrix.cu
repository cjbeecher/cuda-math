#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"


struct Matrix *matrix_create(int rows, int columns, int init_zero) {
    struct Matrix *result = (struct Matrix *)malloc(sizeof(struct Matrix));
    result->rows = rows;
    result->columns = columns;
    result->values = (double *)malloc(rows * columns * sizeof(double));
    if (init_zero == 1)
        for (rows = 0; rows < result->rows * result->columns; rows++)
            result->values[rows] = 0;
    return result;
}


struct Matrix *matrix_multiply(struct Matrix *left, struct Matrix *right) {
    struct Matrix *result = matrix_create(left->rows, right->columns, 0);
    return result;
}


void matrix_print(struct Matrix *matrix) {
    printf("this is from the matrix_print function");
}
