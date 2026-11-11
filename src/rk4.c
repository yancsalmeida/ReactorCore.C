#include "rk4.h"
#include <stdlib.h>
#include <math.h>

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

#include "simd_math.h"

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
#if REACTOR_CORE_HAS_AVX2
    __m256d vhalf_dt = _mm256_set1_pd(half_dt);
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m256d vy = _mm256_loadu_pd(&y->data[i]);
        __m256d vk1 = _mm256_loadu_pd(&ws->k1->data[i]);
        _mm256_storeu_pd(&ws->ytemp->data[i], _mm256_fmadd_pd(vhalf_dt, vk1, vy));
    }
    for (; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + half_dt * ws->k1->data[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + half_dt * ws->k1->data[i];
    }
#endif
    f(t + half_dt, ws->ytemp, ws->k2, params);

    /* Stage 3: y_temp = y + 0.5 * dt * k2; k3 = f(t + 0.5*dt, y_temp) */
#if REACTOR_CORE_HAS_AVX2
    i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m256d vy = _mm256_loadu_pd(&y->data[i]);
        __m256d vk2 = _mm256_loadu_pd(&ws->k2->data[i]);
        _mm256_storeu_pd(&ws->ytemp->data[i], _mm256_fmadd_pd(vhalf_dt, vk2, vy));
    }
    for (; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + half_dt * ws->k2->data[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + half_dt * ws->k2->data[i];
    }
#endif
    f(t + half_dt, ws->ytemp, ws->k3, params);

    /* Stage 4: y_temp = y + dt * k3; k4 = f(t + dt, y_temp) */
#if REACTOR_CORE_HAS_AVX2
    __m256d vdt = _mm256_set1_pd(dt);
    i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m256d vy = _mm256_loadu_pd(&y->data[i]);
        __m256d vk3 = _mm256_loadu_pd(&ws->k3->data[i]);
        _mm256_storeu_pd(&ws->ytemp->data[i], _mm256_fmadd_pd(vdt, vk3, vy));
    }
    for (; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + dt * ws->k3->data[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + dt * ws->k3->data[i];
    }
#endif
    f(t + dt, ws->ytemp, ws->k4, params);

    /* Final Assembly: y_next = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4) */
    const double dt_six = dt / 6.0;
#if REACTOR_CORE_HAS_AVX2
    __m256d vdt_six = _mm256_set1_pd(dt_six);
    __m256d vtwo = _mm256_set1_pd(2.0);
    i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m256d vy = _mm256_loadu_pd(&y->data[i]);
        __m256d vk1 = _mm256_loadu_pd(&ws->k1->data[i]);
        __m256d vk2 = _mm256_loadu_pd(&ws->k2->data[i]);
        __m256d vk3 = _mm256_loadu_pd(&ws->k3->data[i]);
        __m256d vk4 = _mm256_loadu_pd(&ws->k4->data[i]);

        __m256d vk23 = _mm256_add_pd(vk2, vk3);
        __m256d vsum = _mm256_add_pd(vk1, vk4);
        vsum = _mm256_fmadd_pd(vtwo, vk23, vsum);

        _mm256_storeu_pd(&y_next->data[i], _mm256_fmadd_pd(vdt_six, vsum, vy));
    }
    for (; i < dim; i++) {
        y_next->data[i] = y->data[i] + dt_six * (
            ws->k1->data[i] +
            2.0 * ws->k2->data[i] +
            2.0 * ws->k3->data[i] +
            ws->k4->data[i]
        );
    }
#else
    for (size_t i = 0; i < dim; i++) {
        y_next->data[i] = y->data[i] + dt_six * (
            ws->k1->data[i] +
            2.0 * ws->k2->data[i] +
            2.0 * ws->k3->data[i] +
            ws->k4->data[i]
        );
    }
#endif

    return true;
}

Rk45Workspace* rk45_workspace_create(size_t dim) {
    if (dim == 0) {
        return NULL;
    }

    Rk45Workspace *ws = (Rk45Workspace*)malloc(sizeof(Rk45Workspace));
    if (!ws) {
        return NULL;
    }

    ws->dim = dim;
    ws->k1 = vector_create(dim);
    ws->k2 = vector_create(dim);
    ws->k3 = vector_create(dim);
    ws->k4 = vector_create(dim);
    ws->k5 = vector_create(dim);
    ws->k6 = vector_create(dim);
    ws->ytemp = vector_create(dim);
    ws->y4 = vector_create(dim);
    ws->y5 = vector_create(dim);

    if (!ws->k1 || !ws->k2 || !ws->k3 || !ws->k4 || !ws->k5 || !ws->k6 ||
        !ws->ytemp || !ws->y4 || !ws->y5) {
        rk45_workspace_free(ws);
        return NULL;
    }

    return ws;
}

void rk45_workspace_free(Rk45Workspace *ws) {
    if (ws) {
        vector_free(ws->k1);
        vector_free(ws->k2);
        vector_free(ws->k3);
        vector_free(ws->k4);
        vector_free(ws->k5);
        vector_free(ws->k6);
        vector_free(ws->ytemp);
        vector_free(ws->y4);
        vector_free(ws->y5);
        ws->dim = 0;
        free(ws);
    }
}

Rk45Control rk45_default_control(void) {
    Rk45Control c;
    c.rtol = 1e-5;
    c.atol = 1e-7;
    c.h_min = 1e-9;
    c.h_max = 1.0;
    c.safety_factor = 0.90;
    c.min_scale = 0.2;
    c.max_scale = 5.0;
    return c;
}

bool rk45_step(
    Rk45Workspace *ws,
    OdeSystemFunc f,
    double t,
    const Vector *y,
    Vector *y_out,
    double h,
    double rtol,
    double atol,
    double *out_error_norm,
    void *params
) {
    if (!ws || !f || !y || !y_out || y->size != ws->dim || y_out->size != ws->dim || h <= 0.0) {
        return false;
    }

    const size_t dim = ws->dim;

    /* Stage 1: k1 = f(t, y) */
    f(t, y, ws->k1, params);

    /* Stage 2: ytemp = y + h * (1/4 * k1); k2 = f(t + 1/4*h, ytemp) */
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + h * (0.25 * ws->k1->data[i]);
    }
    f(t + 0.25 * h, ws->ytemp, ws->k2, params);

    /* Stage 3: ytemp = y + h * (3/32 * k1 + 9/32 * k2); k3 = f(t + 3/8*h, ytemp) */
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + h * (
            (3.0 / 32.0) * ws->k1->data[i] +
            (9.0 / 32.0) * ws->k2->data[i]
        );
    }
    f(t + 0.375 * h, ws->ytemp, ws->k3, params);

    /* Stage 4: ytemp = y + h * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3); k4 = f(t + 12/13*h, ytemp) */
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + h * (
            (1932.0 / 2197.0) * ws->k1->data[i] -
            (7200.0 / 2197.0) * ws->k2->data[i] +
            (7296.0 / 2197.0) * ws->k3->data[i]
        );
    }
    f(t + (12.0 / 13.0) * h, ws->ytemp, ws->k4, params);

    /* Stage 5: ytemp = y + h * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4); k5 = f(t + h, ytemp) */
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + h * (
            (439.0 / 216.0)  * ws->k1->data[i] -
            8.0              * ws->k2->data[i] +
            (3680.0 / 513.0) * ws->k3->data[i] -
            (845.0 / 4104.0) * ws->k4->data[i]
        );
    }
    f(t + h, ws->ytemp, ws->k5, params);

    /* Stage 6: ytemp = y + h * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5); k6 = f(t + 1/2*h, ytemp) */
    for (size_t i = 0; i < dim; i++) {
        ws->ytemp->data[i] = y->data[i] + h * (
            (-8.0 / 27.0)     * ws->k1->data[i] +
            2.0               * ws->k2->data[i] -
            (3544.0 / 2565.0) * ws->k3->data[i] +
            (1859.0 / 4104.0) * ws->k4->data[i] -
            (11.0 / 40.0)     * ws->k5->data[i]
        );
    }
    f(t + 0.5 * h, ws->ytemp, ws->k6, params);

    /* 4th-order candidate */
    for (size_t i = 0; i < dim; i++) {
        ws->y4->data[i] = y->data[i] + h * (
            (25.0 / 216.0)    * ws->k1->data[i] +
            (1408.0 / 2565.0) * ws->k3->data[i] +
            (2197.0 / 4104.0) * ws->k4->data[i] -
            0.2               * ws->k5->data[i]
        );
    }

    /* 5th-order candidate (propagated) */
    for (size_t i = 0; i < dim; i++) {
        ws->y5->data[i] = y->data[i] + h * (
            (16.0 / 135.0)      * ws->k1->data[i] +
            (6656.0 / 12825.0)  * ws->k3->data[i] +
            (28561.0 / 56430.0) * ws->k4->data[i] -
            (9.0 / 50.0)        * ws->k5->data[i] +
            (2.0 / 55.0)        * ws->k6->data[i]
        );
        y_out->data[i] = ws->y5->data[i];
    }

    /* Weighted RMS error norm */
    if (out_error_norm) {
        double sum_sq = 0.0;
        for (size_t i = 0; i < dim; i++) {
            double err = fabs(ws->y5->data[i] - ws->y4->data[i]);
            double max_val = fmax(fabs(y->data[i]), fabs(ws->y5->data[i]));
            double scale = atol + rtol * max_val;
            if (scale <= 0.0) scale = 1e-15;
            double ratio = err / scale;
            sum_sq += ratio * ratio;
        }
        *out_error_norm = sqrt(sum_sq / (double)dim);
    }

    return true;
}

bool rk45_adaptive_step(
    Rk45Workspace *ws,
    OdeSystemFunc f,
    double *t,
    Vector *y,
    double *h,
    const Rk45Control *ctrl,
    Rk45AdaptiveStats *stats,
    void *params
) {
    if (!ws || !f || !t || !y || !h || !ctrl) {
        return false;
    }

    double cur_h = *h;
    if (cur_h <= 0.0) {
        return false;
    }

    const double h_min = ctrl->h_min > 0.0 ? ctrl->h_min : 1e-9;
    const double h_max = ctrl->h_max > 0.0 ? ctrl->h_max : 1e6;
    const double safety = ctrl->safety_factor > 0.0 ? ctrl->safety_factor : 0.90;
    const double min_scale = ctrl->min_scale > 0.0 ? ctrl->min_scale : 0.2;
    const double max_scale = ctrl->max_scale > 0.0 ? ctrl->max_scale : 5.0;

    while (true) {
        if (cur_h < h_min) {
            cur_h = h_min;
        }

        double error_norm = 0.0;
        bool ok = rk45_step(ws, f, *t, y, ws->y5, cur_h, ctrl->rtol, ctrl->atol, &error_norm, params);
        if (!ok) {
            return false;
        }

        if (stats) {
            if (cur_h < stats->min_h || (stats->steps_accepted + stats->steps_rejected == 0)) {
                stats->min_h = cur_h;
            }
            if (error_norm > stats->max_local_error) {
                stats->max_local_error = error_norm;
            }
        }

        if (error_norm <= 1.0) {
            /* Step accepted */
            *t += cur_h;
            for (size_t i = 0; i < ws->dim; i++) {
                y->data[i] = ws->y5->data[i];
            }

            if (stats) {
                stats->steps_accepted++;
                stats->final_h = cur_h;
            }

            /* Propose next step size */
            double factor;
            if (error_norm == 0.0) {
                factor = max_scale;
            } else {
                factor = safety * pow(error_norm, -0.20);
                if (factor > max_scale) factor = max_scale;
                if (factor < min_scale) factor = min_scale;
            }

            double next_h = cur_h * factor;
            if (next_h > h_max) next_h = h_max;
            if (next_h < h_min) next_h = h_min;
            *h = next_h;

            return true;
        }

        /* Step rejected: retry with reduced h */
        if (stats) {
            stats->steps_rejected++;
        }

        if (cur_h <= h_min) {
            /* Cannot satisfy error bound even at minimum step size */
            return false;
        }

        double factor = safety * pow(error_norm, -0.20);
        if (factor < min_scale) factor = min_scale;
        if (factor > 0.90) factor = 0.90; /* Force shrinkage on rejection */

        cur_h = cur_h * factor;
        if (cur_h < h_min) {
            cur_h = h_min;
        }
        *h = cur_h;
    }
}

bool rk45_integrate_interval(
    Rk45Workspace *ws,
    OdeSystemFunc f,
    double t0,
    double t_final,
    Vector *y,
    double initial_h,
    const Rk45Control *ctrl,
    Rk45AdaptiveStats *stats,
    void *params
) {
    if (!ws || !f || !y || t_final <= t0) {
        return false;
    }

    Rk45Control default_ctrl;
    if (!ctrl) {
        default_ctrl = rk45_default_control();
        ctrl = &default_ctrl;
    }

    if (stats) {
        stats->steps_accepted = 0;
        stats->steps_rejected = 0;
        stats->final_h = 0.0;
        stats->min_h = 1e30;
        stats->max_local_error = 0.0;
    }

    double total_span = t_final - t0;
    double h = initial_h;
    if (h <= 0.0) {
        h = total_span / 20.0;
    }
    if (h > total_span) {
        h = total_span;
    }

    double t = t0;
    const double eps = 1e-12 * fmax(1.0, total_span);

    while (t < t_final - eps) {
        if (t + h > t_final) {
            h = t_final - t;
        }

        double step_h = h;
        bool ok = rk45_adaptive_step(ws, f, &t, y, &step_h, ctrl, stats, params);
        if (!ok) {
            return false;
        }
        h = step_h;
    }

    if (stats && stats->min_h > 1e29) {
        stats->min_h = h;
    }

    return true;
}
