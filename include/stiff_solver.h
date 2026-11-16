#ifndef REACTOR_CORE_STIFF_SOLVER_H
#define REACTOR_CORE_STIFF_SOLVER_H

#include "cstr.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the exact analytical 6x6 Jacobian matrix J = df/dx
 * for the series-parallel CSTR state vector [Ca, Cb, Cc, Cd, T, Tj].
 */
void cstr_series_parallel_jacobian(
    const CstrSeriesParallelParams *params,
    double ca,
    double cb,
    double temp,
    double out_j[6][6]
);

/**
 * @brief Advances the 6-state series-parallel CSTR by a single L-stable
 * 3rd-order Rosenbrock (ROS3L) semi-implicit step using analytical 6x6 Jacobian LU factorization.
 * Remains strictly L-stable even across severe thermal runaway regimes where explicit RK4 diverges.
 *
 * Returns 0 on success, -1 on invalid arguments or singular matrix.
 */
int cstr_series_parallel_step_ros3(
    const CstrSeriesParallelParams *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt
);

/**
 * @brief Adaptive L-stable ROS3L(2) integrator for the 6-state series-parallel CSTR.
 * Sub-steps internally when embedded 2nd-order local error exceeds (rtol, atol).
 *
 * Returns 0 on success, -1 on invalid arguments or step limit overflow.
 */
int cstr_series_parallel_step_ros3_adaptive(
    const CstrSeriesParallelParams *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt,
    double rtol,
    double atol
);

/**
 * @brief Advances the 6-state series-parallel CSTR using adaptive L-stable ROS3L(2)
 * and populates full conversion, yield, selectivity, heat transfer, and runaway diagnostics.
 */
int cstr_series_parallel_step_ros3_detailed(
    const CstrSeriesParallelParams *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt,
    double rtol,
    double atol,
    double runaway_threshold_k,
    CstrSeriesParallelState *out_state
);

/**
 * @brief Advances the classical 2-state CSTR [Ca, T] by a single L-stable
 * 3rd-order Rosenbrock (ROS3L) step using exact analytical 2x2 Jacobian.
 */
int cstr_step_ros3(
    const CstrParameters *params,
    double *ca,
    double *temp_k,
    double dt
);

#ifdef __cplusplus
}
#endif

#endif /* REACTOR_CORE_STIFF_SOLVER_H */
