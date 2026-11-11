#ifndef REACTOR_CORE_RK4_H
#define REACTOR_CORE_RK4_H

#include "vector.h"

/**
 * @brief Function pointer signature for an ODE derivative function: dy/dt = f(t, y, params).
 * @param t Current time.
 * @param y State vector at time t.
 * @param dydt Output vector to store computed derivatives.
 * @param params Generic pointer to model physical parameters (context).
 */
typedef void (*OdeSystemFunc)(double t, const Vector *y, Vector *dydt, void *params);

/**
 * @brief Pre-allocated memory workspace for the Runge-Kutta 4th order integrator.
 * Avoids dynamic heap allocation (malloc/free) inside tight time-integration loops,
 * guaranteeing cache locality and zero memory fragmentation.
 */
typedef struct {
    Vector *k1;
    Vector *k2;
    Vector *k3;
    Vector *k4;
    Vector *ytemp;
    size_t dim;
} Rk4Workspace;

/**
 * @brief Allocates all scratchpad vectors for RK4 integration of dimension dim.
 */
Rk4Workspace* rk4_workspace_create(size_t dim);

/**
 * @brief Frees all scratchpad vectors and the workspace struct.
 */
void rk4_workspace_free(Rk4Workspace *ws);

/**
 * @brief Advances the system state by a single time step dt using classical 4th-order Runge-Kutta:
 *   k1 = f(t, y)
 *   k2 = f(t + dt/2, y + dt/2 * k1)
 *   k3 = f(t + dt/2, y + dt/2 * k2)
 *   k4 = f(t + dt,   y + dt * k3)
 *   y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 *
 * @param ws Pre-allocated RK4 workspace.
 * @param f System derivative evaluation function pointer.
 * @param t Current time.
 * @param y Current state vector.
 * @param y_next Output state vector after step dt.
 * @param dt Time step size.
 * @param params User model parameters passed to f.
 * @return true on success, false if pointers are invalid or dimensions mismatch.
 */
bool rk4_step(
    Rk4Workspace *ws,
    OdeSystemFunc f,
    double t,
    const Vector *y,
    Vector *y_next,
    double dt,
    void *params
);

/**
 * @brief Pre-allocated memory workspace for the Runge-Kutta-Fehlberg 4(5) (RKF45) integrator.
 * Contains 6 stage derivative scratchpad vectors (k1..k6), one temporary state vector (ytemp),
 * and two candidate state vectors (y4, y5) for local truncation error estimation.
 * Guarantees zero heap allocation inside time-stepping loops.
 */
typedef struct {
    Vector *k1;
    Vector *k2;
    Vector *k3;
    Vector *k4;
    Vector *k5;
    Vector *k6;
    Vector *ytemp;
    Vector *y4;
    Vector *y5;
    size_t dim;
} Rk45Workspace;

/**
 * @brief Allocates all 9 scratchpad vectors for RKF45 integration of dimension dim.
 */
Rk45Workspace* rk45_workspace_create(size_t dim);

/**
 * @brief Frees all scratchpad vectors and the Rk45Workspace struct.
 */
void rk45_workspace_free(Rk45Workspace *ws);

/**
 * @brief Configuration parameters for adaptive step size control in RKF45.
 */
typedef struct {
    double rtol;          /* Relative tolerance (e.g. 1e-5) */
    double atol;          /* Absolute tolerance (e.g. 1e-7) */
    double h_min;         /* Minimum allowed step size (e.g. 1e-9) */
    double h_max;         /* Maximum allowed step size (e.g. 1.0) */
    double safety_factor; /* Step adaptation safety multiplier (typically 0.90) */
    double min_scale;     /* Minimum step size change ratio (typically 0.2) */
    double max_scale;     /* Maximum step size change ratio (typically 5.0) */
} Rk45Control;

/**
 * @brief Returns standard default control parameters (rtol=1e-5, atol=1e-7, safety=0.90).
 */
Rk45Control rk45_default_control(void);

/**
 * @brief Runtime diagnostic statistics for an adaptive RKF45 integration run.
 */
typedef struct {
    size_t steps_accepted; /* Count of accepted steps satisfying local error bound */
    size_t steps_rejected; /* Count of rejected steps retried with smaller h */
    double final_h;        /* Step size at conclusion of integration */
    double min_h;          /* Smallest step size utilized during integration */
    double max_local_error;/* Maximum normalized local error observed */
} Rk45AdaptiveStats;

/**
 * @brief Computes a single embedded RKF45 step of fixed length h, evaluating both 4th and 5th
 * order solutions, and returning the weighted RMS local truncation error norm:
 *   err_norm = sqrt( 1/dim * sum( ((y5_i - y4_i) / (atol + rtol * max(|y_i|, |y5_i|)))^2 ) )
 */
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
);

/**
 * @brief Performs an adaptive step from current *t and y.
 * Tries step with *h. If error norm > 1.0, step is rejected, *h is reduced, and step is retried
 * until error norm <= 1.0 or h <= h_min.
 * Upon acceptance:
 *   - *t is advanced by the accepted step size
 *   - y is updated to the new state
 *   - *h is updated to the proposed step size for the next step
 *   - stats (if non-NULL) are updated (steps_accepted, steps_rejected, min_h, max_local_error)
 */
bool rk45_adaptive_step(
    Rk45Workspace *ws,
    OdeSystemFunc f,
    double *t,
    Vector *y,
    double *h,
    const Rk45Control *ctrl,
    Rk45AdaptiveStats *stats,
    void *params
);

/**
 * @brief Integrates an ODE system from t0 to t_final using adaptive RKF45.
 * Dynamically adjusts step size to keep local error below tolerances.
 * Clamps the final step to land exactly on t_final.
 */
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
);

#endif /* REACTOR_CORE_RK4_H */
