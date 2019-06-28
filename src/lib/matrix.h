#ifdef __cplusplus
extern "C" {
#endif

struct Matrix {
    int rows;
    int columns;
    double *values;
};


struct Matrix *matrix_create(int rows, int columns, int init_zero);
struct Matrix *matrix_multiply(struct Matrix *left, struct Matrix *right);
void matrix_print(struct Matrix *matrix);

#ifdef __cplusplus
}
#endif
