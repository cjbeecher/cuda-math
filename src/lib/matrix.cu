#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"


cudaError_t _matrix_alloc_device(struct Matrix *matrix) {
    cudaError_t cuda_status;
    double *matrix_values;
    if (matrix->vectors[0].device_values != NULL)
        return cudaSuccess;
    cuda_status = cudaMalloc((void **) &matrix_values, matrix->total * sizeof(double));
    if (cuda_status != cudaSuccess)
        return cuda_status;
    matrix->vectors[0].device_values = matrix_values;
    return cudaSuccess;
}

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
    _matrix_alloc_device(matrix);
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

__global__ void _matrix_transpose(double *output, const double *matrix, const int o_columns, const int m_columns) {
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

    cuda_status = _matrix_alloc_device(matrix);
    if (cuda_status != cudaSuccess)
        return (int)cuda_status;

    if (*transpose == NULL)
        *transpose = matrix_create(matrix->columns, matrix->rows, 0);

    cuda_status = _matrix_alloc_device(*transpose);
    if (cuda_status != cudaSuccess)
        return (int)cuda_status;

    t = *transpose;
    transpose_values = t->vectors[0].device_values;
    matrix_values = matrix->vectors[0].device_values;
    _matrix_transpose<<<1, t->total>>>(transpose_values, matrix_values, t->columns, matrix->columns);
    return (int)cudaGetLastError();
}

__global__ void _matrix_multiply(
        double *output,
        const double *left,
        const double *right,
        const int left_columns,
        const int right_columns
        ) {
    int index;
    double tmp = 0.0;
    left = left + threadIdx.x * left_columns;
    right = right + threadIdx.y;
    for (index = 0; index < left_columns; index++) {
        tmp += left[0] * right[0];
        left += 1;
        right += right_columns;
    }
    output[threadIdx.x * right_columns + threadIdx.y] = tmp;
}

int matrix_multiply(struct Matrix **product, struct Matrix *left, struct Matrix *right) {
    cudaError_t cuda_status;
    *product = matrix_create(left->rows, right->columns, 1);
    cuda_status = _matrix_alloc_device(*product);
    if (cuda_status != cudaSuccess) {
        matrix_destroy(*product);
        *product = NULL;
        return (int)cuda_status;
    }
    _matrix_multiply<<<1, dim3((*product)->rows, (*product)->columns)>>>(
            (*product)->vectors[0].device_values,
            left->vectors[0].device_values,
            right->vectors[0].device_values,
            left->columns,
            right->columns
            );
    return (int)cudaGetLastError();
}


void matrix_print(struct Matrix *matrix) {
    int row;
    for (row = 0; row < matrix->rows; row++) {
        vector_print(&matrix->vectors[row]);
    }
}
