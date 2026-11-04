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

#endif /* REACTOR_CORE_CSTR_H */
