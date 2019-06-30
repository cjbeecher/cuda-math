#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"


struct Matrix *matrix_create(int rows, int columns, int init_zero) {
    struct Matrix *result = (struct Matrix *)malloc(sizeof(struct Matrix));
    double *values = (double *)malloc(rows * columns * sizeof(double));
    result->rows = rows;
    result->columns = columns;
    result->total = rows * columns;
    result->vectors = (struct Vector *)malloc(rows * sizeof(struct Vector));
    result->vectors[0].device_values = NULL;
    for (rows = 0; rows < result->rows; rows++) {
        result->vectors[rows].size = result->columns;
        result->vectors[rows].capacity = result->columns;
        result->vectors[rows].values = values;
        if (init_zero == 1) {
            for (columns = 0; columns < result->columns; columns++) {
                values[0] = 0;
                values++;
            }
        }
        else
            values += result->columns;
    }
    return result;
}

void matrix_destroy(struct Matrix *matrix) {
    if (matrix->vectors[0].device_values != NULL)
        cudaFree(matrix->vectors[0].device_values);
    free(matrix->vectors);
    free(matrix);
}

int matrix_copy_from_device(struct Matrix *matrix) {
    cudaError_t cuda_status;
    cuda_status = cudaMemcpy(
            matrix->vectors[0].values,
            matrix->vectors[0].device_values,
            matrix->total * sizeof(double),
            cudaMemcpyDeviceToHost
            );
    return (int)cuda_status;
}

int matrix_copy_to_device(struct Matrix *matrix) {
    cudaError_t cuda_status;
    cuda_status = cudaMemcpy(
            matrix->vectors[0].device_values,
            matrix->vectors[0].values,
            matrix->total * sizeof(double),
            cudaMemcpyHostToDevice
    );
    return (int)cuda_status;
}

void matrix_destroy_device(struct Matrix *matrix) {
    cudaFree(matrix->vectors[0].device_values);
    matrix->vectors[0].device_values = NULL;
}

__global__ void _matrix_transpose(double *output, const double *matrix, int o_columns, int m_columns) {
    int index = threadIdx.x;
    int m_col = index % m_columns;
    int m_row = (index - m_col) / m_columns;
    output[m_col * o_columns + m_row] = matrix[index];
}

int matrix_transpose(struct Matrix **transpose, struct Matrix *matrix) {
    double *transpose_values;
    double *matrix_values;
    struct Matrix *t;
    cudaError_t cuda_status;

    if (*transpose != NULL && ((*transpose)->rows != matrix->columns || (*transpose)->columns != matrix->rows))
        return MATRIX_TRANSPOSE_INVALID_DIMENSIONS;

    cuda_status = cudaMalloc((void **) &transpose_values, matrix->total * sizeof(double));
    if (cuda_status != cudaSuccess)
        return (int)cuda_status;

    if (matrix->vectors[0].device_values == NULL) {
        cuda_status = cudaMalloc((void **) &matrix_values, matrix->total * sizeof(double));
        if (cuda_status != cudaSuccess) {
            cudaFree(transpose_values);
            return (int)cuda_status;
        }
        matrix->vectors[0].device_values = matrix_values;
        cuda_status = (cudaError_t)matrix_copy_to_device(matrix);
        if (cuda_status != cudaSuccess) {
            cudaFree(transpose_values);
            cudaFree(matrix_values);
            matrix->vectors[0].device_values = NULL;
            return cuda_status;
        }
    }

    *transpose = matrix_create(matrix->columns, matrix->rows, 0);
    t = *transpose;
    t->vectors[0].device_values = transpose_values;
    _matrix_transpose<<<1, t->total>>>(transpose_values, matrix->vectors[0].device_values, t->columns, matrix->columns);
    return (int)cudaGetLastError();
}

__global__ void _matrix_multiply(double *output, const double *left, const double *right)
{
    int row = threadIdx.x;
//    int col = 3;
//    int col = row % cols;
    output[0] += 7.0; // left[row] * right[col];
}


int matrix_multiply(struct Matrix *product, struct Matrix *left, struct Matrix *right) {
//    double *d_o;
//    double *d_l;
//    double *d_r;
//    cudaMalloc((void **)&d_o, product->rows * product->columns * sizeof(double));
//    cudaMemcpy(d_o, product->values, product->rows * product->columns * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMalloc((void **)&d_l, left->rows * left->columns * sizeof(double));
//    cudaMemcpy(d_l, left->values, left->rows * left->columns * sizeof(double), cudaMemcpyHostToDevice);
//    cudaMalloc((void **)&d_r, right->rows * right->columns * sizeof(double));
//    cudaMemcpy(d_r, left->values, right->rows * right->columns * sizeof(double), cudaMemcpyHostToDevice);
//    _matrix_multiply<<<1, left->rows * left->columns>>>(d_o, d_l, d_r);
//    cudaMemcpy(product->values, d_o, product->rows * product->columns * sizeof(double), cudaMemcpyDeviceToHost);
//    cudaFree(d_o);
//    cudaFree(d_l);
//    cudaFree(d_r);
    return 0;
}


void matrix_print(struct Matrix *matrix) {
    int row;
    for (row = 0; row < matrix->rows; row++) {
        vector_print(&matrix->vectors[row]);
    }
}
