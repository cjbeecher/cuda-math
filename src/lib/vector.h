#ifdef __cplusplus
extern "C" {
#endif

struct Vector {
    int size;
    int capacity;
    double *values;
    double *device_values;
};


struct Vector *vector_create(int size, int capacity, int init_zero);
void vector_destroy(struct Vector *vector);
struct Vector *vector_cross(struct Vector *left, struct Vector *right);
struct Vector *vector_dot(struct Vector *left, struct Vector *right);
void vector_print(struct Vector *vector);

#ifdef __cplusplus
}
#endif
