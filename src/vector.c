#include "vector.h"
#include <stdlib.h>
#include <string.h>

Vector* vector_create(size_t size) {
    if (size == 0) {
        return NULL;
    }

    Vector *vec = (Vector*)malloc(sizeof(Vector));
    if (!vec) {
        return NULL;
    }

    vec->data = (double*)calloc(size, sizeof(double));
    if (!vec->data) {
        free(vec);
        return NULL;
    }

    vec->size = size;
    return vec;
}

void vector_free(Vector *vec) {
    if (vec) {
        if (vec->data) {
            free(vec->data);
            vec->data = NULL;
        }
        vec->size = 0;
        free(vec);
    }
}

Vector* vector_clone(const Vector *src) {
    if (!src || !src->data || src->size == 0) {
        return NULL;
    }

    Vector *copy = vector_create(src->size);
    if (!copy) {
        return NULL;
    }

    memcpy(copy->data, src->data, src->size * sizeof(double));
    return copy;
}

bool vector_copy(Vector *dst, const Vector *src) {
    if (!dst || !src || !dst->data || !src->data || dst->size != src->size) {
        return false;
    }

    memcpy(dst->data, src->data, src->size * sizeof(double));
    return true;
}

void vector_fill(Vector *vec, double value) {
    if (!vec || !vec->data) {
        return;
    }

    for (size_t i = 0; i < vec->size; i++) {
        vec->data[i] = value;
    }
}

bool vector_saxpy(Vector *y, double alpha, const Vector *x) {
    if (!y || !x || !y->data || !x->data || y->size != x->size) {
        return false;
    }

    for (size_t i = 0; i < y->size; i++) {
        y->data[i] += alpha * x->data[i];
    }
    return true;
}
