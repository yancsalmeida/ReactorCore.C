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

/* ========================================================================= */
/* Complex Series-Parallel Kinetics (A -> B -> C, A -> D) & Dynamic Jacket  */
/* ========================================================================= */

/**
 * @brief Parameters for an exothermic CSTR with complex series-parallel kinetics:
 *   1. A -> B  (Desirable intermediate product, rate r1)
 *   2. B -> C  (Consecutive unwanted degradation by-product, rate r2)
 *   3. A -> D  (Competitive parallel side-reaction by-product, rate r3)
 * coupled to a dynamic non-linear cooling jacket with fluid thermal mass.
 */
typedef struct {
    /* Reactor Hydraulics */
    double volumetric_flow_l_per_min;  /* q: Reactor volumetric feed flow (L/min) */
    double volume_liters;              /* V: Reactor fluid volume (L) */
    double rho_cp_j_per_l_k;           /* rho*Cp: Reactor fluid volumetric heat capacity (J/(L·K)) */

    /* Kinetics 1: A -> B */
    double k0_1_per_min;               /* k0_1: Arrhenius frequency factor (1/min) */
    double ea1_over_r_kelvin;          /* Ea_1 / R: Activation energy (K) */
    double delta_h1_j_per_mol;         /* -dH_1: Heat of reaction (J/mol, >0 for exothermic) */

    /* Kinetics 2: B -> C (degradation by-product) */
    double k0_2_per_min;               /* k0_2: Arrhenius frequency factor (1/min) */
    double ea2_over_r_kelvin;          /* Ea_2 / R: Activation energy (K) */
    double delta_h2_j_per_mol;         /* -dH_2: Heat of reaction (J/mol, >0 for exothermic) */

    /* Kinetics 3: A -> D (parallel side by-product) */
    double k0_3_per_min;               /* k0_3: Arrhenius frequency factor (1/min) */
    double ea3_over_r_kelvin;          /* Ea_3 / R: Activation energy (K) */
    double delta_h3_j_per_mol;         /* -dH_3: Heat of reaction (J/mol, >0 for exothermic) */

    /* Feed Concentrations and Temperature */
    double ca_feed_mol_per_l;          /* CA0: Feed reactant A concentration (mol/L) */
    double cb_feed_mol_per_l;          /* CB0: Feed product B concentration (mol/L) */
    double cc_feed_mol_per_l;          /* CC0: Feed by-product C concentration (mol/L) */
    double cd_feed_mol_per_l;          /* CD0: Feed by-product D concentration (mol/L) */
    double t_feed_kelvin;              /* T0: Feed temperature (K) */

    /* Dynamic Jacket Cooling System */
    double jacket_volume_liters;       /* Vj: Jacket fluid volume (L) */
    double jacket_flow_l_per_min;      /* qj: Coolant volumetric flow rate (L/min) */
    double jacket_rho_cp_j_per_l_k;    /* (rho*Cp)_j: Coolant volumetric heat capacity (J/(L·K)) */
    double jacket_t_in_kelvin;         /* Tj_in: Coolant inlet temperature (K) */
    double ua_j_per_min_k;             /* UA: Heat transfer area * coefficient (J/(min·K)) */
} CstrSeriesParallelParams;

/**
 * @brief Dynamic state telemetry and chemical performance metrics for series-parallel CSTR.
 */
typedef struct {
    double ca_mol_per_l;
    double cb_mol_per_l;
    double cc_mol_per_l;
    double cd_mol_per_l;
    double temp_kelvin;
    double jacket_temp_kelvin;
    double conversion_a_pct;           /* Conversion: (CA0 - CA)/CA0 * 100% */
    double yield_b_pct;                /* Yield: (CB - CB0)/CA0 * 100% */
    double selectivity_b_pct;          /* Selectivity: CB / (CB + CC + CD) * 100% */
    double rate_r1_mol_per_l_min;
    double rate_r2_mol_per_l_min;
    double rate_r3_mol_per_l_min;
    double heat_generated_watts;       /* Total reaction heat release rate (Watts) */
    double heat_removed_watts;         /* Heat removal rate by jacket (Watts) */
    bool is_thermal_runaway;
} CstrSeriesParallelState;

/**
 * @brief Standard default benchmark parameters for series-parallel exothermic CSTR with dynamic jacket.
 */
CstrSeriesParallelParams cstr_series_parallel_default_parameters(void);

/**
 * @brief Calculates the 3 simultaneous Arrhenius reaction rates:
 *   r1 = k0_1 * exp(-Ea1 / (R*T)) * Ca
 *   r2 = k0_2 * exp(-Ea2 / (R*T)) * Cb
 *   r3 = k0_3 * exp(-Ea3 / (R*T)) * Ca
 */
void cstr_series_parallel_rates(
    const CstrSeriesParallelParams *params,
    double Ca,
    double Cb,
    double T,
    double *out_r1,
    double *out_r2,
    double *out_r3
);

/**
 * @brief Evaluates coupled 6-variable ODE system:
 *   dCa/dt = (q/V)*(CA0 - Ca) - r1 - r3
 *   dCb/dt = (q/V)*(CB0 - Cb) + r1 - r2
 *   dCc/dt = (q/V)*(CC0 - Cc) + r2
 *   dCd/dt = (q/V)*(CD0 - Cd) + r3
 *   dT/dt  = (q/V)*(T0 - T) + [(-dH1)*r1 + (-dH2)*r2 + (-dH3)*r3]/(rho*Cp) - (UA/(V*rho*Cp))*(T - Tj)
 *   dTj/dt = (qj/Vj)*(Tj_in - Tj) + (UA / (Vj * (rho*Cp)_j))*(T - Tj)
 */
void cstr_series_parallel_derivatives(double t, const Vector *y, Vector *dydt, void *params);

/**
 * @brief Advances a single series-parallel CSTR by an RK4 step of length dt (min).
 * Updates ca, cb, cc, cd, temp, and temp_j in-place.
 */
int cstr_series_parallel_step_rk4(
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
 * @brief Advances a series-parallel CSTR by an RK4 step and populates full detailed metrics.
 */
int cstr_series_parallel_step_detailed(
    const CstrSeriesParallelParams *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt,
    double runaway_threshold_k,
    CstrSeriesParallelState *out_state
);

/**
 * @brief Pre-allocated workspace for ensemble series-parallel multi-reactor simulation.
 */
typedef struct {
    size_t capacity;
    double *k1_ca, *k1_cb, *k1_cc, *k1_cd, *k1_temp, *k1_tj;
    double *k2_ca, *k2_cb, *k2_cc, *k2_cd, *k2_temp, *k2_tj;
    double *k3_ca, *k3_cb, *k3_cc, *k3_cd, *k3_temp, *k3_tj;
    double *k4_ca, *k4_cb, *k4_cc, *k4_cd, *k4_temp, *k4_tj;
    double *temp_ca, *temp_cb, *temp_cc, *temp_cd, *temp_temp, *temp_tj;
} CstrSeriesParallelWorkspace;

CstrSeriesParallelWorkspace* cstr_series_parallel_workspace_create(size_t capacity);
void cstr_series_parallel_workspace_free(CstrSeriesParallelWorkspace *ws);

/**
 * @brief Parallel evaluation of coupled series-parallel CSTR derivatives across an ensemble of N reactors.
 * Accelerated with SIMD AVX-512 / AVX2 + FMA with scalar fallback.
 */
void cstr_series_parallel_batch_derivatives(
    size_t n,
    const CstrSeriesParallelParams *params,
    const double *ca,
    const double *cb,
    const double *cc,
    const double *cd,
    const double *temp,
    const double *temp_j,
    double *out_dcadt,
    double *out_dcbdt,
    double *out_dccdt,
    double *out_dcddt,
    double *out_dtdt,
    double *out_dtjdt
);

/**
 * @brief Advances an ensemble of N series-parallel CSTR reactors by an RK4 step using pre-allocated workspace.
 */
bool cstr_series_parallel_batch_step_rk4_ws(
    CstrSeriesParallelWorkspace *ws,
    size_t n,
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
 * @brief Advances an ensemble of N series-parallel CSTR reactors by an RK4 step.
 */
bool cstr_series_parallel_batch_step_rk4(
    size_t n,
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
 * @brief Simulates dynamic trajectories for an ensemble of N series-parallel reactors over multiple RK4 steps.
 */
bool cstr_series_parallel_batch_simulate_trajectory(
    size_t n,
    const CstrSeriesParallelParams *params,
    const double *ca0,
    const double *cb0,
    const double *cc0,
    const double *cd0,
    const double *temp0,
    const double *temp_j0,
    double dt,
    size_t steps,
    double *out_time,
    double *out_ca,
    double *out_cb,
    double *out_cc,
    double *out_cd,
    double *out_temp,
    double *out_temp_j
);

#endif /* REACTOR_CORE_CSTR_H */
