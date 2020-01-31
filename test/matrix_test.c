//
// Created by Carlos Beecher on 6/21/2019.
//

#include <stdio.h>
#include "matrix.h"


#define LEFT_ROWS 2
#define LEFT_COLUMNS 3
#define RIGHT_ROWS 3
#define RIGHT_COLUMNS 2


int main() {
    int index;
    int status;
    struct Matrix *alu = NULL;
    struct Matrix *lu = NULL;
    printf("Creating Left Matrix\n");
    struct Matrix *left = matrix_create(LEFT_ROWS, LEFT_COLUMNS, 0);
    printf("Creating Right Matrix\n");
    struct Matrix *right = matrix_create(RIGHT_ROWS, RIGHT_COLUMNS, 0);
    struct Matrix *output = NULL;
    struct Matrix *transpose = NULL;

    for (index = 0; index < left->total; index++)
        left->vectors[0].values[index] = (double)index + 1;
    for (index = 0; index < right->total; index++)
        right->vectors[0].values[index] = (double)index + 1;

    status = matrix_copy_to_device(left);
    if (status != 0)
        printf("Error copying Left matrix to device\n");
    status = matrix_copy_to_device(right);
    if (status != 0)
        printf("Error copying Right matrix to device\n");

    printf("Multiplying Left and Right matrices\n");
    status = matrix_multiply(&output, left, right);
    if (status != 0)
        printf("Error on matrix multiplication\n");
    status = matrix_copy_from_device(output, false);
    if (status != 0)
        printf("Error copying Product from device\n");

    status = matrix_transpose(&transpose, left);
    if (status != 0)
        printf("Unsuccessful in transposing matrix\n");
    status = matrix_transpose(&transpose, left);
    if (status != 0)
        printf("Unsuccessful in transposing matrix second time\n");

    printf("\nLeft Matrix:\n");
    matrix_print(left);
    if (status == 0) {
        status = matrix_copy_from_device(transpose, false);
        if (status != 0) {
            printf("Error copying transpose from device: %i\n", status);
            printf("Device memory address: %p\n", transpose->vectors[0].device_values);
        }
        else {
            printf("\nTransposed Matrix:\n");
            matrix_print(transpose);
            matrix_destroy(transpose);
        }
    }

    lu = matrix_create(3, 3, false);
    lu->vectors[0].values[0] = 2.0;
    lu->vectors[0].values[1] = -1.0;
    lu->vectors[0].values[2] = -2.0;
    lu->vectors[0].values[3] = -4.0;
    lu->vectors[0].values[4] = 6.0;
    lu->vectors[0].values[5] = 3.0;
    lu->vectors[0].values[6] = -4.0;
    lu->vectors[0].values[7] = -2.0;
    lu->vectors[0].values[8] = 8.0;
    printf("\nLU matrix\n");
    matrix_print(lu);
    matrix_copy(&alu, lu);
    matrix_lu_decomposition(&alu, lu);
    printf("\nMatrix A LU decomposition output\n");
    matrix_print(alu);
    matrix_destroy(alu);

//    printf("\nRight Matrix:\n");
//    matrix_print(right);
//    printf("\n");
//    printf("Product Matrix:\n");
//    matrix_print(output);
//    printf("\n");
    matrix_destroy(left);
    matrix_destroy(right);
    matrix_destroy(output);
}
