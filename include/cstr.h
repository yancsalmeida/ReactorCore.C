#ifndef REACTOR_CORE_CSTR_H
#define REACTOR_CORE_CSTR_H

#include "vector.h"
#include <stdbool.h>

/**
 * @brief Parameters for an exothermic Continuous Stirred-Tank Reactor (CSTR)
 * with an irreversible 1st-order reaction A -> B and a jacket cooling system.
 * Based on classic Rawlings/Fogler non-linear chemical dynamics benchmark.
 */
typedef struct {
    double volumetric_flow_l_per_min;  /* q: Volumetric flow rate (L/min) */
    double volume_liters;              /* V: Reactor fluid volume (L) */
    double k0_per_min;                 /* k0: Arrhenius frequency factor (1/min) */
    double ea_over_r_kelvin;           /* Ea/R: Activation energy / R (Kelvin) */
    double delta_h_j_per_mol;          /* -dH: Heat of reaction (J/mol, positive for exothermic) */
    double rho_cp_j_per_l_k;           /* rho*Cp: Fluid density * heat capacity (J/(L·K)) */
    double ua_j_per_min_k;             /* UA: Heat transfer area * coefficient (J/(min·K)) */
    double ca_feed_mol_per_l;          /* CA0: Feed reactant concentration (mol/L) */
    double t_feed_kelvin;              /* T0: Feed temperature (K) */
    double t_coolant_kelvin;           /* Tc: Jacket coolant temperature (K) */
} CstrParameters;

/**
 * @brief Returns standard literature benchmark parameters for exothermic CSTR.
 */
CstrParameters cstr_default_parameters(void);

/**
 * @brief Calculates the Arrhenius reaction rate: r = k0 * exp(-Ea / (R*T)) * Ca
 * @param params CSTR parameters.
 * @param Ca Concentration of reactant A (mol/L).
 * @param T Temperature in the reactor (K).
 * @return Reaction rate in mol / (L·min).
 */
double cstr_reaction_rate(const CstrParameters *params, double Ca, double T);

/**
 * @brief Evaluates coupled non-linear ODE derivatives for CSTR state [Ca, T]:
 *   dCa/dt = (q/V) * (CA0 - Ca) - r
 *   dT/dt  = (q/V) * (T0 - T) + (-dH / (rho*Cp)) * r - (UA / (V*rho*Cp)) * (T - Tc)
 *
 * Implements the OdeSystemFunc function pointer signature for rk4_step.
 */
void cstr_derivatives(double t, const Vector *y, Vector *dydt, void *params);

/**
 * @brief Evaluates whether the current state indicates a thermal runaway hazard.
 * @param y State vector [Ca, T].
 * @param max_safe_temp_kelvin Maximum temperature threshold before runaway shutdown.
 * @return true if temperature exceeds threshold, false otherwise.
 */
bool cstr_is_thermal_runaway(const Vector *y, double max_safe_temp_kelvin);

/**
 * @brief Pre-allocated workspace for batch / ensemble multi-reactor RK4 simulation.
 * Avoids any heap allocation in the time integration loop.
 */
typedef struct {
    size_t capacity;
    double *k1_ca;
    double *k1_temp;
    double *k2_ca;
    double *k2_temp;
    double *k3_ca;
    double *k3_temp;
    double *k4_ca;
    double *k4_temp;
    double *temp_ca;
    double *temp_temp;
} CstrBatchWorkspace;

/**
 * @brief Allocates scratchpad buffers for a batch of up to capacity reactors.
 */
CstrBatchWorkspace* cstr_batch_workspace_create(size_t capacity);

/**
 * @brief Frees the batch workspace struct and its scratchpad buffers.
 */
void cstr_batch_workspace_free(CstrBatchWorkspace *ws);

/**
 * @brief Evaluates coupled non-linear CSTR derivatives across an ensemble of N reactors:
 *   dCa/dt = (q/V) * (CA0 - Ca) - r
 *   dT/dt  = (q/V) * (T0 - T) + (-dH / (rho*Cp)) * r - (UA / (V*rho*Cp)) * (T - Tc)
 * where r = k0 * exp(-Ea / (R*T)) * Ca.
 *
 * Parallelized with AVX2 256-bit SIMD and FMA instructions (4 double-precision reactors per instruction),
 * with automatic scalar remainder handling.
 *
 * @param n Number of reactors.
 * @param params Common CSTR physical parameters.
 * @param ca Array of reactant concentrations [n].
 * @param temp Array of temperatures [n].
 * @param out_dcadt Array for computed dCa/dt [n].
 * @param out_dtdt Array for computed dT/dt [n].
 */
void cstr_batch_derivatives(
    size_t n,
    const CstrParameters *params,
    const double *ca,
    const double *temp,
    double *out_dcadt,
    double *out_dtdt
);

/**
 * @brief Advances an ensemble of N CSTR reactors by a single RK4 step using pre-allocated workspace.
 * Parallelized via AVX2 and FMA across all 4 Runge-Kutta stages.
 */
bool cstr_batch_step_rk4_ws(
    CstrBatchWorkspace *ws,
    size_t n,
    const CstrParameters *params,
    double *ca,
    double *temp,
    double dt
);

/**
 * @brief Advances an ensemble of N CSTR reactors by a single RK4 step.
 * Uses stack memory for N <= 64 or internal workspace for larger N.
 */
bool cstr_batch_step_rk4(
    size_t n,
    const CstrParameters *params,
    double *ca,
    double *temp,
    double dt
);

/**
 * @brief Simulates dynamic trajectories for an ensemble of N reactors over multiple RK4 steps.
 * Outputs ca and temp at all time steps for all reactors.
 * Output memory layouts:
 *   out_ca[step * n + reactor_idx]
 *   out_temp[step * n + reactor_idx]
 *
 * @param n Number of reactors.
 * @param params CSTR parameters.
 * @param ca0 Initial concentrations [n].
 * @param temp0 Initial temperatures [n].
 * @param dt Step size (min).
 * @param steps Number of integration steps.
 * @param out_time Optional array of size (steps + 1) for time points (can be NULL).
 * @param out_ca Pre-allocated array of size (steps + 1) * n.
 * @param out_temp Pre-allocated array of size (steps + 1) * n.
 * @return true on success, false on invalid parameters.
 */
bool cstr_batch_simulate_trajectory(
    size_t n,
    const CstrParameters *params,
    const double *ca0,
    const double *temp0,
    double dt,
    size_t steps,
    double *out_time,
    double *out_ca,
    double *out_temp
);

#endif /* REACTOR_CORE_CSTR_H */
