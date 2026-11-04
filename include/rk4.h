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

#endif /* REACTOR_CORE_RK4_H */
