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

#ifdef __cplusplus
}
#endif

#endif /* REACTOR_CORE_FFI_H */
