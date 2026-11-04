#include "cstr.h"
#include <math.h>

CstrParameters cstr_default_parameters(void) {
    CstrParameters p;
    p.volumetric_flow_l_per_min = 100.0;
    p.volume_liters             = 100.0;
    p.k0_per_min                = 7.08e10;
    p.ea_over_r_kelvin          = 8330.1;
    p.delta_h_j_per_mol         = 200000.0; /* Exothermic (+ for heat generation) */
    p.rho_cp_j_per_l_k          = 4000.0;
    p.ua_j_per_min_k            = 40000.0;
    p.ca_feed_mol_per_l         = 1.0;
    p.t_feed_kelvin             = 350.0;
    p.t_coolant_kelvin          = 300.0;
    return p;
}

double cstr_reaction_rate(const CstrParameters *params, double Ca, double T) {
    if (!params || Ca <= 0.0 || T <= 0.0) {
        return 0.0;
    }

    double k = params->k0_per_min * exp(-params->ea_over_r_kelvin / T);
    return k * Ca;
}

void cstr_derivatives(double t, const Vector *y, Vector *dydt, void *params) {
    (void)t; /* Autonomous system: derivatives do not explicitly depend on time */

    if (!y || !dydt || y->size < 2 || dydt->size < 2 || !params) {
        return;
    }

    const CstrParameters *p = (const CstrParameters*)params;

    double Ca = y->data[0];
    double T  = y->data[1];

    /* Physical safety clamps for evaluation */
    if (Ca < 0.0) Ca = 0.0;
    if (T < 50.0) T = 50.0;

    double r = cstr_reaction_rate(p, Ca, T);

    double space_time_inv = p->volumetric_flow_l_per_min / p->volume_liters;
    double heat_gen_coeff = p->delta_h_j_per_mol / p->rho_cp_j_per_l_k;
    double heat_rem_coeff = p->ua_j_per_min_k / (p->volume_liters * p->rho_cp_j_per_l_k);

    /* dCa/dt = (q/V)*(CA0 - Ca) - r */
    dydt->data[0] = space_time_inv * (p->ca_feed_mol_per_l - Ca) - r;

    /* dT/dt = (q/V)*(T0 - T) + (-dH/(rho*Cp))*r - (UA/(V*rho*Cp))*(T - Tc) */
    dydt->data[1] = space_time_inv * (p->t_feed_kelvin - T)
                  + heat_gen_coeff * r
                  - heat_rem_coeff * (T - p->t_coolant_kelvin);
}

bool cstr_is_thermal_runaway(const Vector *y, double max_safe_temp_kelvin) {
    if (!y || y->size < 2) {
        return false;
    }
    return y->data[1] >= max_safe_temp_kelvin;
}
