#include "rk4.h"
#include <stdlib.h>

Rk4Workspace* rk4_workspace_create(size_t dim) {
    if (dim == 0) {
        return NULL;
    }

    Rk4Workspace *ws = (Rk4Workspace*)malloc(sizeof(Rk4Workspace));
    if (!ws) {
        return NULL;
    }

    ws->dim = dim;
    ws->k1 = vector_create(dim);
    ws->k2 = vector_create(dim);
    ws->k3 = vector_create(dim);
    ws->k4 = vector_create(dim);
    ws->ytemp = vector_create(dim);

    if (!ws->k1 || !ws->k2 || !ws->k3 || !ws->k4 || !ws->ytemp) {
        rk4_workspace_free(ws);
        return NULL;
    }

    return ws;
}

void rk4_workspace_free(Rk4Workspace *ws) {
    if (ws) {
        vector_free(ws->k1);
        vector_free(ws->k2);
        vector_free(ws->k3);
        vector_free(ws->k4);
        vector_free(ws->ytemp);
        ws->dim = 0;
        free(ws);
    }
}

bool rk4_step(
    Rk4Workspace *ws,
    OdeSystemFunc f,
    double t,
    const Vector *y,
    Vector *y_next,
    double dt,
    void *params
) {
    if (!ws || !f || !y || !y_next || y->size != ws->dim || y_next->size != ws->dim) {
        return false;
    }

    const size_t dim = ws->dim;
    const double half_dt = 0.5 * dt;

    /* Stage 1: k1 = f(t, y) */
    f(t, y, ws->k1, params);

    /* Stage 2: y_temp = y + 0.5 * dt * k1; k2 = f(t + 0.5*dt, y_temp) */
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + half_dt * ws->k1->data[i];
    }
    f(t + half_dt, ws->ytemp, ws->k2, params);

    /* Stage 3: y_temp = y + 0.5 * dt * k2; k3 = f(t + 0.5*dt, y_temp) */
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + half_dt * ws->k2->data[i];
    }
    f(t + half_dt, ws->ytemp, ws->k3, params);

    /* Stage 4: y_temp = y + dt * k3; k4 = f(t + dt, y_temp) */
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + dt * ws->k3->data[i];
    }
    f(t + dt, ws->ytemp, ws->k4, params);

    /* Final Assembly: y_next = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4) */
    const double dt_six = dt / 6.0;
    for (size_t i = 0; i < dim; i++) {
        y_next->data[i] = y->data[i] + dt_six * (
            ws->k1->data[i] +
            2.0 * ws->k2->data[i] +
            2.0 * ws->k3->data[i] +
            ws->k4->data[i]
        );
    }

    return true;
}
