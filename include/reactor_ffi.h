#ifndef REACTOR_CORE_FFI_H
#define REACTOR_CORE_FFI_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef BUILDING_REACTOR_CORE_DLL
    #define REACTOR_API __declspec(dllexport)
  #else
    #define REACTOR_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define REACTOR_API __attribute__((visibility("default")))
  #else
    #define REACTOR_API
  #endif
#endif

/**
 * CstrParamsFFI: C-ABI compatible Plain Old Data struct for CSTR simulation parameters.
 * Exactly matches literature benchmark (Fogler / Rawlings continuous exothermic CSTR).
 */
typedef struct {
    double volumetric_flow_l_per_min;  /* q: Volumetric flow rate (L/min) */
    double volume_liters;              /* V: Reactor fluid volume (L) */
    double k0_per_min;                 /* k0: Arrhenius frequency factor (1/min) */
    double ea_over_r_kelvin;           /* Ea/R: Activation energy / R (Kelvin) */
    double delta_h_j_per_mol;          /* -dH: Heat of reaction (J/mol) */
    double rho_cp_j_per_l_k;           /* rho*Cp: Fluid density * heat capacity (J/(L·K)) */
    double ua_j_per_min_k;             /* UA: Heat transfer area * coefficient (J/(min·K)) */
    double ca_feed_mol_per_l;          /* CA0: Feed reactant concentration (mol/L) */
    double t_feed_kelvin;              /* T0: Feed temperature (K) */
    double t_coolant_kelvin;           /* Tc: Jacket coolant temperature (K) */
} CstrParamsFFI;

/**
 * Detailed CSTR state information returned by reactor_cstr_step_detailed.
 */
typedef struct {
    double ca_mol_per_l;
    double temp_kelvin;
    double conversion_pct;
    double reaction_rate_mol_per_l_min;
    bool is_runaway;
} CstrStateFFI;

/**
 * Diagnostic metrics returned from adaptive RKF45 integration.
 * Matches CstrAdaptiveStatsFFI in ChemFlow.NET.
 */
typedef struct {
    size_t steps_accepted;
    size_t steps_rejected;
    double final_h;
    double min_h;
    double max_local_error;
} CstrAdaptiveStatsFFI;

/**
 * Returns version and architecture string for runtime handshake.
 */
REACTOR_API const char* reactor_core_version(void);

/**
 * Populates out_params with standard literature default physical parameters.
 * Returns 0 on success, -1 on NULL pointer.
 */
REACTOR_API int reactor_cstr_default_params(CstrParamsFFI *out_params);

/**
 * Calculates instantaneous Arrhenius reaction rate: r = k0 * exp(-Ea / (R*T)) * Ca.
 */
REACTOR_API double reactor_cstr_reaction_rate(const CstrParamsFFI *params, double Ca, double T);

/**
 * Advances the CSTR state by a single RK4 step of length dt (min).
 * Updates ca and temp_k in-place.
 * Returns 0 on success, -1 on invalid arguments.
 */
REACTOR_API int reactor_cstr_step_rk4(
    const CstrParamsFFI *params,
    double *ca,
    double *temp_k,
    double dt
);

/**
 * Advances the CSTR state and returns full telemetry diagnostics (conversion, reaction rate, runaway).
 * Returns 0 on success, -1 on invalid arguments.
 */
REACTOR_API int reactor_cstr_step_detailed(
    const CstrParamsFFI *params,
    double *ca,
    double *temp_k,
    double dt,
    double runaway_threshold_k,
    CstrStateFFI *out_state
);

/**
 * High-speed batch trajectory simulation: runs 'steps' RK4 iterations entirely in C.
 * Populates pre-allocated arrays out_time, out_ca, out_temp (each size: steps + 1).
 * Returns 0 on success, -1 on invalid arguments.
 */
REACTOR_API int reactor_cstr_simulate_trajectory(
    const CstrParamsFFI *params,
    double ca0,
    double temp0,
    double dt,
    size_t steps,
    double *out_time,
    double *out_ca,
    double *out_temp
);

/**
 * Advances the CSTR state across a time step dt using adaptive RKF45 error control.
 * Sub-steps adaptively to maintain local truncation error within tolerances.
 * Updates ca and temp_k in-place.
 * Returns 0 on success, -1 on invalid arguments or failure.
 */
REACTOR_API int reactor_cstr_step_adaptive(
    const CstrParamsFFI *params,
    double *ca,
    double *temp_k,
    double dt,
    double rtol,
    double atol,
    CstrAdaptiveStatsFFI *out_stats
);

/**
 * Advances the CSTR state with adaptive RKF45 and returns detailed diagnostics.
 * Returns 0 on success, -1 on invalid arguments.
 */
REACTOR_API int reactor_cstr_step_detailed_adaptive(
    const CstrParamsFFI *params,
    double *ca,
    double *temp_k,
    double dt,
    double rtol,
    double atol,
    double runaway_threshold_k,
    CstrStateFFI *out_state,
    CstrAdaptiveStatsFFI *out_stats
);

/**
 * High-speed batch trajectory simulation using adaptive RKF45 error control.
 * Samples trajectory at 'steps' intervals over total_time = dt * steps.
 * Populates pre-allocated arrays out_time, out_ca, out_temp (each size: steps + 1).
 * Returns 0 on success, -1 on invalid arguments.
 */
REACTOR_API int reactor_cstr_simulate_adaptive_trajectory(
    const CstrParamsFFI *params,
    double ca0,
    double temp0,
    double dt,
    size_t steps,
    double rtol,
    double atol,
    double *out_time,
    double *out_ca,
    double *out_temp,
    CstrAdaptiveStatsFFI *out_stats
);

/**
 * Returns 1 if AVX2 and FMA SIMD acceleration is compiled and enabled, 0 otherwise.
 */
REACTOR_API int reactor_core_has_simd(void);

/**
 * Advances an ensemble of N CSTR reactors by a single RK4 step using SIMD AVX2+FMA.
 * Updates ca_array and temp_array in-place.
 * Returns 0 on success, -1 on invalid arguments.
 */
REACTOR_API int reactor_cstr_batch_step_rk4(
    const CstrParamsFFI *params,
    double *ca_array,
    double *temp_array,
    size_t n_reactors,
    double dt
);

/**
 * Advances an ensemble of N CSTR reactors and computes detailed telemetry states for each.
 * Returns 0 on success, -1 on invalid arguments.
 */
REACTOR_API int reactor_cstr_batch_step_detailed(
    const CstrParamsFFI *params,
    double *ca_array,
    double *temp_array,
    size_t n_reactors,
    double dt,
    double runaway_threshold_k,
    CstrStateFFI *out_states
);

/**
 * High-speed batch trajectory simulation for an ensemble of N reactors simultaneously using SIMD AVX2+FMA.
 * Populates out_time (size: steps + 1) and out_ca, out_temp (size: (steps + 1) * n_reactors).
 * Returns 0 on success, -1 on invalid arguments.
 */
REACTOR_API int reactor_cstr_batch_simulate_trajectory(
    const CstrParamsFFI *params,
    const double *ca0_array,
    const double *temp0_array,
    size_t n_reactors,
    double dt,
    size_t steps,
    double *out_time,
    double *out_ca,
    double *out_temp
);

/**
 * Returns the compiled SIMD instruction width: 512 for AVX-512, 256 for AVX2+FMA, 0 for scalar.
 */
REACTOR_API int reactor_core_simd_level(void);

/* ========================================================================= */
/* Complex Series-Parallel Kinetics & Dynamic Jacket FFI Exports             */
/* ========================================================================= */

typedef struct {
    double volumetric_flow_l_per_min;
    double volume_liters;
    double rho_cp_j_per_l_k;

    double k0_1_per_min;
    double ea1_over_r_kelvin;
    double delta_h1_j_per_mol;

    double k0_2_per_min;
    double ea2_over_r_kelvin;
    double delta_h2_j_per_mol;

    double k0_3_per_min;
    double ea3_over_r_kelvin;
    double delta_h3_j_per_mol;

    double ca_feed_mol_per_l;
    double cb_feed_mol_per_l;
    double cc_feed_mol_per_l;
    double cd_feed_mol_per_l;
    double t_feed_kelvin;

    double jacket_volume_liters;
    double jacket_flow_l_per_min;
    double jacket_rho_cp_j_per_l_k;
    double jacket_t_in_kelvin;
    double ua_j_per_min_k;
} CstrSeriesParallelParamsFFI;

typedef struct {
    double ca_mol_per_l;
    double cb_mol_per_l;
    double cc_mol_per_l;
    double cd_mol_per_l;
    double temp_kelvin;
    double jacket_temp_kelvin;
    double conversion_a_pct;
    double yield_b_pct;
    double selectivity_b_pct;
    double rate_r1_mol_per_l_min;
    double rate_r2_mol_per_l_min;
    double rate_r3_mol_per_l_min;
    double heat_generated_watts;
    double heat_removed_watts;
    bool is_thermal_runaway;
} CstrSeriesParallelStateFFI;

REACTOR_API int reactor_cstr_series_parallel_default_params(CstrSeriesParallelParamsFFI *out_params);

REACTOR_API int reactor_cstr_series_parallel_rates(
    const CstrSeriesParallelParamsFFI *params,
    double Ca,
    double Cb,
    double T,
    double *out_r1,
    double *out_r2,
    double *out_r3
);

REACTOR_API int reactor_cstr_series_parallel_step_rk4(
    const CstrSeriesParallelParamsFFI *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt
);

REACTOR_API int reactor_cstr_series_parallel_step_detailed(
    const CstrSeriesParallelParamsFFI *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt,
    double runaway_threshold_k,
    CstrSeriesParallelStateFFI *out_state
);

REACTOR_API int reactor_cstr_series_parallel_batch_step_rk4(
    const CstrSeriesParallelParamsFFI *params,
    double *ca_array,
    double *cb_array,
    double *cc_array,
    double *cd_array,
    double *temp_array,
    double *temp_j_array,
    size_t n_reactors,
    double dt
);

REACTOR_API int reactor_cstr_series_parallel_batch_step_detailed(
    const CstrSeriesParallelParamsFFI *params,
    double *ca_array,
    double *cb_array,
    double *cc_array,
    double *cd_array,
    double *temp_array,
    double *temp_j_array,
    size_t n_reactors,
    double dt,
    double runaway_threshold_k,
    CstrSeriesParallelStateFFI *out_states
);

REACTOR_API int reactor_cstr_series_parallel_batch_simulate_trajectory(
    const CstrSeriesParallelParamsFFI *params,
    const double *ca0,
    const double *cb0,
    const double *cc0,
    const double *cd0,
    const double *temp0,
    const double *temp_j0,
    size_t n_reactors,
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

#ifdef __cplusplus
}
#endif

#endif /* REACTOR_CORE_FFI_H */
