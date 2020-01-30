#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "vector.h"

#define MATRIX_INVALID_DIMENSIONS -1

// elements in the member vectors represents a row in a matrix
struct Matrix {
    int rows;
    int columns;
    int total;  // Convenience member for rows * columns
    struct Vector *vectors;
};


struct Matrix *matrix_create(int rows, int columns, int init_zero);
void matrix_destroy(struct Matrix *matrix);
int matrix_copy_from_device(struct Matrix *matrix, bool destroy);
int matrix_copy_to_device(struct Matrix *matrix);
void matrix_destroy_device(struct Matrix *matrix);
// Transpose will initialize all values for argument transpose
// struct Matrix *transpose = malloc(sizeof(struct Matrix));
// status = matrix_transpose(&transpose, matrix);
int matrix_transpose(struct Matrix **transpose, struct Matrix *matrix);
int matrix_multiply(struct Matrix **product, struct Matrix *left, struct Matrix *right);
// Current implementation permits only square matrices
int matrix_lu_decomposition(struct Matrix **l, struct Matrix *matrix);
void matrix_print(struct Matrix *matrix);

#ifdef __cplusplus
}
#endif
