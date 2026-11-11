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

#include <stdlib.h>
#include <string.h>
#include "simd_math.h"

bool cstr_is_thermal_runaway(const Vector *y, double max_safe_temp_kelvin) {
    if (!y || y->size < 2) {
        return false;
    }
    return y->data[1] >= max_safe_temp_kelvin;
}

CstrBatchWorkspace* cstr_batch_workspace_create(size_t capacity) {
    if (capacity == 0) return NULL;

    CstrBatchWorkspace *ws = (CstrBatchWorkspace*)malloc(sizeof(CstrBatchWorkspace));
    if (!ws) return NULL;

    ws->capacity  = capacity;
    ws->k1_ca     = (double*)malloc(capacity * sizeof(double));
    ws->k1_temp   = (double*)malloc(capacity * sizeof(double));
    ws->k2_ca     = (double*)malloc(capacity * sizeof(double));
    ws->k2_temp   = (double*)malloc(capacity * sizeof(double));
    ws->k3_ca     = (double*)malloc(capacity * sizeof(double));
    ws->k3_temp   = (double*)malloc(capacity * sizeof(double));
    ws->k4_ca     = (double*)malloc(capacity * sizeof(double));
    ws->k4_temp   = (double*)malloc(capacity * sizeof(double));
    ws->temp_ca   = (double*)malloc(capacity * sizeof(double));
    ws->temp_temp = (double*)malloc(capacity * sizeof(double));

    if (!ws->k1_ca || !ws->k1_temp || !ws->k2_ca || !ws->k2_temp ||
        !ws->k3_ca || !ws->k3_temp || !ws->k4_ca || !ws->k4_temp ||
        !ws->temp_ca || !ws->temp_temp) {
        cstr_batch_workspace_free(ws);
        return NULL;
    }

    return ws;
}

void cstr_batch_workspace_free(CstrBatchWorkspace *ws) {
    if (!ws) return;

    free(ws->k1_ca);
    free(ws->k1_temp);
    free(ws->k2_ca);
    free(ws->k2_temp);
    free(ws->k3_ca);
    free(ws->k3_temp);
    free(ws->k4_ca);
    free(ws->k4_temp);
    free(ws->temp_ca);
    free(ws->temp_temp);
    ws->capacity = 0;
    free(ws);
}

void cstr_batch_derivatives(
    size_t n,
    const CstrParameters *params,
    const double *ca,
    const double *temp,
    double *out_dcadt,
    double *out_dtdt
) {
    if (n == 0 || !params || !ca || !temp || !out_dcadt || !out_dtdt) {
        return;
    }

    const double space_time_inv = params->volumetric_flow_l_per_min / params->volume_liters;
    const double heat_gen_coeff = params->delta_h_j_per_mol / params->rho_cp_j_per_l_k;
    const double heat_rem_coeff = params->ua_j_per_min_k / (params->volume_liters * params->rho_cp_j_per_l_k);

#if REACTOR_CORE_HAS_AVX2
    const __m256d v_space_time_inv = _mm256_set1_pd(space_time_inv);
    const __m256d v_heat_gen_coeff = _mm256_set1_pd(heat_gen_coeff);
    const __m256d v_heat_rem_coeff = _mm256_set1_pd(heat_rem_coeff);

    const __m256d v_ca_feed = _mm256_set1_pd(params->ca_feed_mol_per_l);
    const __m256d v_t_feed  = _mm256_set1_pd(params->t_feed_kelvin);
    const __m256d v_t_cool  = _mm256_set1_pd(params->t_coolant_kelvin);
    const __m256d v_k0      = _mm256_set1_pd(params->k0_per_min);
    const __m256d v_ea_r    = _mm256_set1_pd(params->ea_over_r_kelvin);

    const __m256d v_zero  = _mm256_setzero_pd();
    const __m256d v_t_min = _mm256_set1_pd(50.0);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vCa = _mm256_loadu_pd(&ca[i]);
        __m256d vT  = _mm256_loadu_pd(&temp[i]);

        /* Physical safety limits */
        vCa = _mm256_max_pd(vCa, v_zero);
        vT  = _mm256_max_pd(vT, v_t_min);

        /* Reaction rate: r = k0 * exp(-Ea / (R*T)) * Ca */
        __m256d vArg = _mm256_div_pd(_mm256_sub_pd(v_zero, v_ea_r), vT);
        __m256d vExp = simd_exp256_pd(vArg);
        __m256d vr = _mm256_mul_pd(_mm256_mul_pd(v_k0, vExp), vCa);

        /* dCa/dt = space_time_inv * (CA0 - Ca) - r */
        __m256d vdCadt = _mm256_fmsub_pd(v_space_time_inv, _mm256_sub_pd(v_ca_feed, vCa), vr);

        /* dT/dt = space_time_inv * (T0 - T) + heat_gen * r - heat_rem * (T - Tc) */
        __m256d vdTdt = _mm256_mul_pd(v_space_time_inv, _mm256_sub_pd(v_t_feed, vT));
        vdTdt = _mm256_fmadd_pd(v_heat_gen_coeff, vr, vdTdt);
        vdTdt = _mm256_fnmadd_pd(v_heat_rem_coeff, _mm256_sub_pd(vT, v_t_cool), vdTdt);

        _mm256_storeu_pd(&out_dcadt[i], vdCadt);
        _mm256_storeu_pd(&out_dtdt[i], vdTdt);
    }

    /* Scalar remainder handling */
    for (; i < n; i++) {
        double Ca_val = ca[i] < 0.0 ? 0.0 : ca[i];
        double T_val  = temp[i] < 50.0 ? 50.0 : temp[i];

        double r = params->k0_per_min * exp(-params->ea_over_r_kelvin / T_val) * Ca_val;

        out_dcadt[i] = space_time_inv * (params->ca_feed_mol_per_l - Ca_val) - r;
        out_dtdt[i]  = space_time_inv * (params->t_feed_kelvin - T_val)
                     + heat_gen_coeff * r
                     - heat_rem_coeff * (T_val - params->t_coolant_kelvin);
    }
#else
    for (size_t i = 0; i < n; i++) {
        double Ca_val = ca[i] < 0.0 ? 0.0 : ca[i];
        double T_val  = temp[i] < 50.0 ? 50.0 : temp[i];

        double r = params->k0_per_min * exp(-params->ea_over_r_kelvin / T_val) * Ca_val;

        out_dcadt[i] = space_time_inv * (params->ca_feed_mol_per_l - Ca_val) - r;
        out_dtdt[i]  = space_time_inv * (params->t_feed_kelvin - T_val)
                     + heat_gen_coeff * r
                     - heat_rem_coeff * (T_val - params->t_coolant_kelvin);
    }
#endif
}

bool cstr_batch_step_rk4_ws(
    CstrBatchWorkspace *ws,
    size_t n,
    const CstrParameters *params,
    double *ca,
    double *temp,
    double dt
) {
    if (!ws || ws->capacity < n || n == 0 || !params || !ca || !temp) {
        return false;
    }

    const double half_dt = 0.5 * dt;

    /* Stage 1: k1 = f(ca, temp) */
    cstr_batch_derivatives(n, params, ca, temp, ws->k1_ca, ws->k1_temp);

    /* Stage 2 prep: temp_ca = ca + half_dt * k1_ca; temp_temp = temp + half_dt * k1_temp */
#if REACTOR_CORE_HAS_AVX2
    __m256d vhalf_dt = _mm256_set1_pd(half_dt);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v_ca = _mm256_loadu_pd(&ca[i]);
        __m256d v_k1 = _mm256_loadu_pd(&ws->k1_ca[i]);
        _mm256_storeu_pd(&ws->temp_ca[i], _mm256_fmadd_pd(vhalf_dt, v_k1, v_ca));

        __m256d v_t   = _mm256_loadu_pd(&temp[i]);
        __m256d v_tk1 = _mm256_loadu_pd(&ws->k1_temp[i]);
        _mm256_storeu_pd(&ws->temp_temp[i], _mm256_fmadd_pd(vhalf_dt, v_tk1, v_t));
    }
    for (; i < n; i++) {
        ws->temp_ca[i] = ca[i] + half_dt * ws->k1_ca[i];
        ws->temp_temp[i] = temp[i] + half_dt * ws->k1_temp[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        ws->temp_ca[i] = ca[i] + half_dt * ws->k1_ca[i];
        ws->temp_temp[i] = temp[i] + half_dt * ws->k1_temp[i];
    }
#endif

    /* Stage 2: k2 = f(temp_ca, temp_temp) */
    cstr_batch_derivatives(n, params, ws->temp_ca, ws->temp_temp, ws->k2_ca, ws->k2_temp);

    /* Stage 3 prep: temp_ca = ca + half_dt * k2_ca; temp_temp = temp + half_dt * k2_temp */
#if REACTOR_CORE_HAS_AVX2
    i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v_ca = _mm256_loadu_pd(&ca[i]);
        __m256d v_k2 = _mm256_loadu_pd(&ws->k2_ca[i]);
        _mm256_storeu_pd(&ws->temp_ca[i], _mm256_fmadd_pd(vhalf_dt, v_k2, v_ca));

        __m256d v_t   = _mm256_loadu_pd(&temp[i]);
        __m256d v_tk2 = _mm256_loadu_pd(&ws->k2_temp[i]);
        _mm256_storeu_pd(&ws->temp_temp[i], _mm256_fmadd_pd(vhalf_dt, v_tk2, v_t));
    }
    for (; i < n; i++) {
        ws->temp_ca[i] = ca[i] + half_dt * ws->k2_ca[i];
        ws->temp_temp[i] = temp[i] + half_dt * ws->k2_temp[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        ws->temp_ca[i] = ca[i] + half_dt * ws->k2_ca[i];
        ws->temp_temp[i] = temp[i] + half_dt * ws->k2_temp[i];
    }
#endif

    /* Stage 3: k3 = f(temp_ca, temp_temp) */
    cstr_batch_derivatives(n, params, ws->temp_ca, ws->temp_temp, ws->k3_ca, ws->k3_temp);

    /* Stage 4 prep: temp_ca = ca + dt * k3_ca; temp_temp = temp + dt * k3_temp */
#if REACTOR_CORE_HAS_AVX2
    __m256d v_dt = _mm256_set1_pd(dt);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v_ca = _mm256_loadu_pd(&ca[i]);
        __m256d v_k3 = _mm256_loadu_pd(&ws->k3_ca[i]);
        _mm256_storeu_pd(&ws->temp_ca[i], _mm256_fmadd_pd(v_dt, v_k3, v_ca));

        __m256d v_t   = _mm256_loadu_pd(&temp[i]);
        __m256d v_tk3 = _mm256_loadu_pd(&ws->k3_temp[i]);
        _mm256_storeu_pd(&ws->temp_temp[i], _mm256_fmadd_pd(v_dt, v_tk3, v_t));
    }
    for (; i < n; i++) {
        ws->temp_ca[i] = ca[i] + dt * ws->k3_ca[i];
        ws->temp_temp[i] = temp[i] + dt * ws->k3_temp[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        ws->temp_ca[i] = ca[i] + dt * ws->k3_ca[i];
        ws->temp_temp[i] = temp[i] + dt * ws->k3_temp[i];
    }
#endif

    /* Stage 4: k4 = f(temp_ca, temp_temp) */
    cstr_batch_derivatives(n, params, ws->temp_ca, ws->temp_temp, ws->k4_ca, ws->k4_temp);

    /* Final Assembly: y_{n+1} = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4) */
    const double dt_six = dt / 6.0;
#if REACTOR_CORE_HAS_AVX2
    __m256d vdt_six = _mm256_set1_pd(dt_six);
    __m256d vtwo = _mm256_set1_pd(2.0);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        /* Assembly for Ca */
        __m256d v_ca = _mm256_loadu_pd(&ca[i]);
        __m256d vk1  = _mm256_loadu_pd(&ws->k1_ca[i]);
        __m256d vk2  = _mm256_loadu_pd(&ws->k2_ca[i]);
        __m256d vk3  = _mm256_loadu_pd(&ws->k3_ca[i]);
        __m256d vk4  = _mm256_loadu_pd(&ws->k4_ca[i]);
        __m256d vk23 = _mm256_add_pd(vk2, vk3);
        __m256d vsum = _mm256_add_pd(vk1, vk4);
        vsum = _mm256_fmadd_pd(vtwo, vk23, vsum);
        _mm256_storeu_pd(&ca[i], _mm256_fmadd_pd(vdt_six, vsum, v_ca));

        /* Assembly for Temperature */
        __m256d v_t   = _mm256_loadu_pd(&temp[i]);
        __m256d vtk1  = _mm256_loadu_pd(&ws->k1_temp[i]);
        __m256d vtk2  = _mm256_loadu_pd(&ws->k2_temp[i]);
        __m256d vtk3  = _mm256_loadu_pd(&ws->k3_temp[i]);
        __m256d vtk4  = _mm256_loadu_pd(&ws->k4_temp[i]);
        __m256d vtk23 = _mm256_add_pd(vtk2, vtk3);
        __m256d vtsum = _mm256_add_pd(vtk1, vtk4);
        vtsum = _mm256_fmadd_pd(vtwo, vtk23, vtsum);
        _mm256_storeu_pd(&temp[i], _mm256_fmadd_pd(vdt_six, vtsum, v_t));
    }
    for (; i < n; i++) {
        ca[i] += dt_six * (ws->k1_ca[i] + 2.0*ws->k2_ca[i] + 2.0*ws->k3_ca[i] + ws->k4_ca[i]);
        temp[i] += dt_six * (ws->k1_temp[i] + 2.0*ws->k2_temp[i] + 2.0*ws->k3_temp[i] + ws->k4_temp[i]);
    }
#else
    for (size_t i = 0; i < n; i++) {
        ca[i] += dt_six * (ws->k1_ca[i] + 2.0*ws->k2_ca[i] + 2.0*ws->k3_ca[i] + ws->k4_ca[i]);
        temp[i] += dt_six * (ws->k1_temp[i] + 2.0*ws->k2_temp[i] + 2.0*ws->k3_temp[i] + ws->k4_temp[i]);
    }
#endif

    return true;
}

bool cstr_batch_step_rk4(
    size_t n,
    const CstrParameters *params,
    double *ca,
    double *temp,
    double dt
) {
    if (n == 0 || !params || !ca || !temp) return false;

    if (n <= 64) {
        /* Thread stack allocation for zero heap overhead on small/medium batches */
        double k1_ca[64], k1_t[64];
        double k2_ca[64], k2_t[64];
        double k3_ca[64], k3_t[64];
        double k4_ca[64], k4_t[64];
        double tmp_ca[64], tmp_t[64];
        CstrBatchWorkspace ws;
        ws.capacity = 64;
        ws.k1_ca = k1_ca;
        ws.k1_temp = k1_t;
        ws.k2_ca = k2_ca;
        ws.k2_temp = k2_t;
        ws.k3_ca = k3_ca;
        ws.k3_temp = k3_t;
        ws.k4_ca = k4_ca;
        ws.k4_temp = k4_t;
        ws.temp_ca = tmp_ca;
        ws.temp_temp = tmp_t;
        return cstr_batch_step_rk4_ws(&ws, n, params, ca, temp, dt);
    } else {
        CstrBatchWorkspace *ws = cstr_batch_workspace_create(n);
        if (!ws) return false;
        bool ok = cstr_batch_step_rk4_ws(ws, n, params, ca, temp, dt);
        cstr_batch_workspace_free(ws);
        return ok;
    }
}

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
) {
    if (n == 0 || !params || !ca0 || !temp0 || !out_ca || !out_temp || dt <= 0.0) {
        return false;
    }

    CstrBatchWorkspace *ws = cstr_batch_workspace_create(n);
    if (!ws) return false;

    if (out_time) out_time[0] = 0.0;
    for (size_t j = 0; j < n; j++) {
        out_ca[j] = ca0[j];
        out_temp[j] = temp0[j];
    }

    double *curr_ca = (double*)malloc(n * sizeof(double));
    double *curr_temp = (double*)malloc(n * sizeof(double));
    if (!curr_ca || !curr_temp) {
        free(curr_ca);
        free(curr_temp);
        cstr_batch_workspace_free(ws);
        return false;
    }
    memcpy(curr_ca, ca0, n * sizeof(double));
    memcpy(curr_temp, temp0, n * sizeof(double));

    for (size_t s = 1; s <= steps; s++) {
        if (!cstr_batch_step_rk4_ws(ws, n, params, curr_ca, curr_temp, dt)) {
            free(curr_ca);
            free(curr_temp);
            cstr_batch_workspace_free(ws);
            return false;
        }

        if (out_time) out_time[s] = (double)s * dt;
        size_t offset = s * n;
        for (size_t j = 0; j < n; j++) {
            out_ca[offset + j] = curr_ca[j];
            out_temp[offset + j] = curr_temp[j];
        }
    }

    free(curr_ca);
    free(curr_temp);
    cstr_batch_workspace_free(ws);
    return true;
}
