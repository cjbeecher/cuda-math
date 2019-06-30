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
    printf("Creating Left Matrix\n");
    struct Matrix *left = matrix_create(LEFT_ROWS, LEFT_COLUMNS, 0);
    printf("Creating Right Matrix\n");
    struct Matrix *right = matrix_create(RIGHT_ROWS, RIGHT_COLUMNS, 0);
    printf("Creating Output Matrix\n\n");
    struct Matrix *output = matrix_create(LEFT_COLUMNS, RIGHT_COLUMNS, 1);
    struct Matrix *transpose = NULL;

    for (index = 0; index < left->total; index++)
        left->vectors[0].values[index] = (double)index + 1;
    for (index = 0; index < right->total; index++)
        right->vectors[0].values[index] = (double)index + 1;

//    matrix_multiply(output, left, right);
    status = matrix_transpose(&transpose, left);
    if (status != 0)
        printf("Unsuccessful in transposing matrix");
    status = matrix_transpose(&transpose, left);
    if (status != 0)
        printf("Unsuccessful in transposing matrix second time");

    matrix_print(left);
    printf("\n");
    if (status == 0) {
        status = matrix_copy_from_device(transpose);
        if (status != 0) {
            printf("Error copying transpose from device: %i\n", status);
            printf("Device memory address: %p\n", transpose->vectors[0].device_values);
        }
        else {
            matrix_print(transpose);
            matrix_destroy(transpose);
        }
    }
//    matrix_print(right);
//    printf("\n");
//    matrix_print(output);
//    printf("\n");
    matrix_destroy(left);
    matrix_destroy(right);
    matrix_destroy(output);
    printf("\n");
}
