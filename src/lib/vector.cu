#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "vector.h"


struct Vector *vector_create(int size, int capacity, int init_zero) {
    int index;
    struct Vector *vector = (struct Vector *)malloc(sizeof(struct Vector));
    vector->values = (double *)malloc(capacity * sizeof(double));
    vector->size = size;
    vector->capacity = capacity;
    if (init_zero == 1)
        for (index = 0; index < size; index++)
            vector->values[index] = 0;
    vector->device_values = NULL;
    return vector;
}

void vector_destroy(struct Vector *vector) {
    free(vector->values);
    free(vector);
}

struct Vector *vector_cross(struct Vector *left, struct Vector *right) {
    return 0;
}

struct Vector *vector_dot(struct Vector *left, struct Vector *right) {
    return 0;
}


void vector_print(struct Vector *vector) {
    int index;
    for (index = 0; index < vector->size; index++)
        printf("%.3f, ", vector->values[index]);
    printf("\n");
}
