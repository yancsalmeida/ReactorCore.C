#ifndef REACTOR_CORE_VECTOR_H
#define REACTOR_CORE_VECTOR_H

#include <stddef.h>
#include <stdbool.h>

/**
 * @brief Represents a dynamically allocated numerical state vector in C11.
 * Encapsulates the underlying raw buffer and dimension for memory safety.
 */
typedef struct {
    double *data;
    size_t size;
} Vector;

/**
 * @brief Allocates and initializes a new Vector of specified dimension with zeros.
 * @param size Number of double-precision elements.
 * @return Pointer to newly allocated Vector, or NULL on allocation failure.
 */
Vector* vector_create(size_t size);

/**
 * @brief Safely frees the Vector structure and its internal data buffer.
 * @param vec Pointer to vector to free (NULL-safe).
 */
void vector_free(Vector *vec);

/**
 * @brief Deep clones an existing vector.
 * @param src Source vector to copy.
 * @return New cloned Vector pointer, or NULL on failure.
 */
Vector* vector_clone(const Vector *src);

/**
 * @brief Copies contents from src to dst. Both vectors must have equal size.
 * @param dst Destination vector.
 * @param src Source vector.
 * @return true if successful, false if size mismatch or NULL pointer.
 */
bool vector_copy(Vector *dst, const Vector *src);

/**
 * @brief Fills all elements of the vector with a constant scalar value.
 * @param vec Target vector.
 * @param value Constant value to fill.
 */
void vector_fill(Vector *vec, double value);

/**
 * @brief SAXPY in-place operation: y = alpha * x + y.
 * Standard BLAS Level-1 primitive used for numerical ODE stepping.
 * @param y Destination/accumulator vector.
 * @param alpha Scalar multiplier.
 * @param x Input vector.
 * @return true on success, false on dimension mismatch.
 */
bool vector_saxpy(Vector *y, double alpha, const Vector *x);

#endif /* REACTOR_CORE_VECTOR_H */
