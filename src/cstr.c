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

/* ========================================================================= */
/* Complex Series-Parallel Kinetics (A -> B -> C, A -> D) & Dynamic Jacket  */
/* ========================================================================= */

CstrSeriesParallelParams cstr_series_parallel_default_parameters(void) {
    CstrSeriesParallelParams p;
    /* Reactor Hydraulics */
    p.volumetric_flow_l_per_min = 100.0;
    p.volume_liters             = 100.0;
    p.rho_cp_j_per_l_k          = 4000.0;

    /* Kinetics 1: A -> B (Desirable intermediate product) */
    p.k0_1_per_min              = 7.08e10;
    p.ea1_over_r_kelvin         = 8330.1;
    p.delta_h1_j_per_mol        = 200000.0; /* Exothermic (+ for heat generation) */

    /* Kinetics 2: B -> C (Degradation by-product) */
    p.k0_2_per_min              = 3.50e9;
    p.ea2_over_r_kelvin         = 7500.0;
    p.delta_h2_j_per_mol        = 120000.0;

    /* Kinetics 3: A -> D (Parallel side-reaction by-product) */
    p.k0_3_per_min              = 1.20e9;
    p.ea3_over_r_kelvin         = 9000.0;
    p.delta_h3_j_per_mol        = 150000.0;

    /* Feed Concentrations and Temperature */
    p.ca_feed_mol_per_l         = 1.0;
    p.cb_feed_mol_per_l         = 0.0;
    p.cc_feed_mol_per_l         = 0.0;
    p.cd_feed_mol_per_l         = 0.0;
    p.t_feed_kelvin             = 350.0;

    /* Dynamic Jacket Cooling System */
    p.jacket_volume_liters      = 25.0;
    p.jacket_flow_l_per_min     = 50.0;
    p.jacket_rho_cp_j_per_l_k   = 4184.0;
    p.jacket_t_in_kelvin        = 295.0;
    p.ua_j_per_min_k            = 40000.0;

    return p;
}

void cstr_series_parallel_rates(
    const CstrSeriesParallelParams *params,
    double Ca,
    double Cb,
    double T,
    double *out_r1,
    double *out_r2,
    double *out_r3
) {
    if (!params) return;

    double Ca_clamped = Ca < 0.0 ? 0.0 : Ca;
    double Cb_clamped = Cb < 0.0 ? 0.0 : Cb;
    double T_clamped  = T < 50.0 ? 50.0 : T;

    double k1 = params->k0_1_per_min * exp(-params->ea1_over_r_kelvin / T_clamped);
    double k2 = params->k0_2_per_min * exp(-params->ea2_over_r_kelvin / T_clamped);
    double k3 = params->k0_3_per_min * exp(-params->ea3_over_r_kelvin / T_clamped);

    if (out_r1) *out_r1 = k1 * Ca_clamped;
    if (out_r2) *out_r2 = k2 * Cb_clamped;
    if (out_r3) *out_r3 = k3 * Ca_clamped;
}

void cstr_series_parallel_derivatives(double t, const Vector *y, Vector *dydt, void *params) {
    (void)t;
    if (!y || !dydt || y->size < 6 || dydt->size < 6 || !params) return;

    const CstrSeriesParallelParams *p = (const CstrSeriesParallelParams*)params;

    double Ca = y->data[0] < 0.0 ? 0.0 : y->data[0];
    double Cb = y->data[1] < 0.0 ? 0.0 : y->data[1];
    double Cc = y->data[2] < 0.0 ? 0.0 : y->data[2];
    double Cd = y->data[3] < 0.0 ? 0.0 : y->data[3];
    double T  = y->data[4] < 50.0 ? 50.0 : y->data[4];
    double Tj = y->data[5] < 50.0 ? 50.0 : y->data[5];

    double r1, r2, r3;
    cstr_series_parallel_rates(p, Ca, Cb, T, &r1, &r2, &r3);

    double space_time_inv = p->volumetric_flow_l_per_min / p->volume_liters;
    double qj_over_vj     = p->jacket_flow_l_per_min / p->jacket_volume_liters;

    double heat_gen = (p->delta_h1_j_per_mol * r1 +
                       p->delta_h2_j_per_mol * r2 +
                       p->delta_h3_j_per_mol * r3) / p->rho_cp_j_per_l_k;

    double heat_rem_reactor = (p->ua_j_per_min_k / (p->volume_liters * p->rho_cp_j_per_l_k)) * (T - Tj);
    double heat_rem_jacket  = (p->ua_j_per_min_k / (p->jacket_volume_liters * p->jacket_rho_cp_j_per_l_k)) * (T - Tj);

    /* dCa/dt = (q/V)*(CA0 - Ca) - r1 - r3 */
    dydt->data[0] = space_time_inv * (p->ca_feed_mol_per_l - Ca) - r1 - r3;

    /* dCb/dt = (q/V)*(CB0 - Cb) + r1 - r2 */
    dydt->data[1] = space_time_inv * (p->cb_feed_mol_per_l - Cb) + r1 - r2;

    /* dCc/dt = (q/V)*(CC0 - Cc) + r2 */
    dydt->data[2] = space_time_inv * (p->cc_feed_mol_per_l - Cc) + r2;

    /* dCd/dt = (q/V)*(CD0 - Cd) + r3 */
    dydt->data[3] = space_time_inv * (p->cd_feed_mol_per_l - Cd) + r3;

    /* dT/dt = (q/V)*(T0 - T) + heat_gen - heat_rem_reactor */
    dydt->data[4] = space_time_inv * (p->t_feed_kelvin - T) + heat_gen - heat_rem_reactor;

    /* dTj/dt = (qj/Vj)*(Tj_in - Tj) + heat_rem_jacket */
    dydt->data[5] = qj_over_vj * (p->jacket_t_in_kelvin - Tj) + heat_rem_jacket;
}

int cstr_series_parallel_step_rk4(
    const CstrSeriesParallelParams *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt
) {
    if (!params || !ca || !cb || !cc || !cd || !temp || !temp_j || dt <= 0.0) {
        return -1;
    }

    double y_curr[6] = { *ca, *cb, *cc, *cd, *temp, *temp_j };
    double k1[6], k2[6], k3[6], k4[6], y_stage[6];
    Vector vy, vdydt;
    vdydt.size = 6;
    vy.size = 6;

    double half_dt = 0.5 * dt;

    /* Stage 1: k1 = f(y_curr) */
    vy.data = y_curr;
    vdydt.data = k1;
    cstr_series_parallel_derivatives(0.0, &vy, &vdydt, (void*)params);

    /* Stage 2: k2 = f(y_curr + 0.5*dt*k1) */
    for (int i = 0; i < 6; i++) y_stage[i] = y_curr[i] + half_dt * k1[i];
    vy.data = y_stage;
    vdydt.data = k2;
    cstr_series_parallel_derivatives(0.0, &vy, &vdydt, (void*)params);

    /* Stage 3: k3 = f(y_curr + 0.5*dt*k2) */
    for (int i = 0; i < 6; i++) y_stage[i] = y_curr[i] + half_dt * k2[i];
    vy.data = y_stage;
    vdydt.data = k3;
    cstr_series_parallel_derivatives(0.0, &vy, &vdydt, (void*)params);

    /* Stage 4: k4 = f(y_curr + dt*k3) */
    for (int i = 0; i < 6; i++) y_stage[i] = y_curr[i] + dt * k3[i];
    vy.data = y_stage;
    vdydt.data = k4;
    cstr_series_parallel_derivatives(0.0, &vy, &vdydt, (void*)params);

    /* Simpson's combination */
    const double dt_six = dt / 6.0;
    *ca     = y_curr[0] + dt_six * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
    *cb     = y_curr[1] + dt_six * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
    *cc     = y_curr[2] + dt_six * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
    *cd     = y_curr[3] + dt_six * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]);
    *temp   = y_curr[4] + dt_six * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]);
    *temp_j = y_curr[5] + dt_six * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]);

    /* Physical non-negative clamps */
    if (*ca < 0.0) *ca = 0.0;
    if (*cb < 0.0) *cb = 0.0;
    if (*cc < 0.0) *cc = 0.0;
    if (*cd < 0.0) *cd = 0.0;
    if (*temp < 50.0) *temp = 50.0;
    if (*temp_j < 50.0) *temp_j = 50.0;

    return 0;
}

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
) {
    if (!out_state) return -1;

    int rc = cstr_series_parallel_step_rk4(params, ca, cb, cc, cd, temp, temp_j, dt);
    if (rc != 0) return rc;

    double r1, r2, r3;
    cstr_series_parallel_rates(params, *ca, *cb, *temp, &r1, &r2, &r3);

    out_state->ca_mol_per_l          = *ca;
    out_state->cb_mol_per_l          = *cb;
    out_state->cc_mol_per_l          = *cc;
    out_state->cd_mol_per_l          = *cd;
    out_state->temp_kelvin           = *temp;
    out_state->jacket_temp_kelvin    = *temp_j;

    double ca0 = params->ca_feed_mol_per_l > 1e-12 ? params->ca_feed_mol_per_l : 1.0;
    out_state->conversion_a_pct      = ((params->ca_feed_mol_per_l - *ca) / ca0) * 100.0;
    out_state->yield_b_pct           = ((*cb - params->cb_feed_mol_per_l) / ca0) * 100.0;

    double total_products = *cb + *cc + *cd;
    out_state->selectivity_b_pct     = total_products > 1e-12 ? (*cb / total_products) * 100.0 : 100.0;

    out_state->rate_r1_mol_per_l_min = r1;
    out_state->rate_r2_mol_per_l_min = r2;
    out_state->rate_r3_mol_per_l_min = r3;

    /* Heat rates in Watts (J/s = J/min / 60) */
    double heat_gen_j_per_min = params->volume_liters *
        (params->delta_h1_j_per_mol * r1 +
         params->delta_h2_j_per_mol * r2 +
         params->delta_h3_j_per_mol * r3);
    out_state->heat_generated_watts  = heat_gen_j_per_min / 60.0;

    double heat_rem_j_per_min = params->ua_j_per_min_k * (*temp - *temp_j);
    out_state->heat_removed_watts    = heat_rem_j_per_min / 60.0;

    out_state->is_thermal_runaway    = (*temp >= runaway_threshold_k);

    return 0;
}

CstrSeriesParallelWorkspace* cstr_series_parallel_workspace_create(size_t capacity) {
    if (capacity == 0) return NULL;

    CstrSeriesParallelWorkspace *ws = (CstrSeriesParallelWorkspace*)malloc(sizeof(CstrSeriesParallelWorkspace));
    if (!ws) return NULL;

    ws->capacity = capacity;
    ws->k1_ca = (double*)malloc(capacity * sizeof(double));
    ws->k1_cb = (double*)malloc(capacity * sizeof(double));
    ws->k1_cc = (double*)malloc(capacity * sizeof(double));
    ws->k1_cd = (double*)malloc(capacity * sizeof(double));
    ws->k1_temp = (double*)malloc(capacity * sizeof(double));
    ws->k1_tj = (double*)malloc(capacity * sizeof(double));

    ws->k2_ca = (double*)malloc(capacity * sizeof(double));
    ws->k2_cb = (double*)malloc(capacity * sizeof(double));
    ws->k2_cc = (double*)malloc(capacity * sizeof(double));
    ws->k2_cd = (double*)malloc(capacity * sizeof(double));
    ws->k2_temp = (double*)malloc(capacity * sizeof(double));
    ws->k2_tj = (double*)malloc(capacity * sizeof(double));

    ws->k3_ca = (double*)malloc(capacity * sizeof(double));
    ws->k3_cb = (double*)malloc(capacity * sizeof(double));
    ws->k3_cc = (double*)malloc(capacity * sizeof(double));
    ws->k3_cd = (double*)malloc(capacity * sizeof(double));
    ws->k3_temp = (double*)malloc(capacity * sizeof(double));
    ws->k3_tj = (double*)malloc(capacity * sizeof(double));

    ws->k4_ca = (double*)malloc(capacity * sizeof(double));
    ws->k4_cb = (double*)malloc(capacity * sizeof(double));
    ws->k4_cc = (double*)malloc(capacity * sizeof(double));
    ws->k4_cd = (double*)malloc(capacity * sizeof(double));
    ws->k4_temp = (double*)malloc(capacity * sizeof(double));
    ws->k4_tj = (double*)malloc(capacity * sizeof(double));

    ws->temp_ca = (double*)malloc(capacity * sizeof(double));
    ws->temp_cb = (double*)malloc(capacity * sizeof(double));
    ws->temp_cc = (double*)malloc(capacity * sizeof(double));
    ws->temp_cd = (double*)malloc(capacity * sizeof(double));
    ws->temp_temp = (double*)malloc(capacity * sizeof(double));
    ws->temp_tj = (double*)malloc(capacity * sizeof(double));

    if (!ws->k1_ca || !ws->k1_cb || !ws->k1_cc || !ws->k1_cd || !ws->k1_temp || !ws->k1_tj ||
        !ws->k2_ca || !ws->k2_cb || !ws->k2_cc || !ws->k2_cd || !ws->k2_temp || !ws->k2_tj ||
        !ws->k3_ca || !ws->k3_cb || !ws->k3_cc || !ws->k3_cd || !ws->k3_temp || !ws->k3_tj ||
        !ws->k4_ca || !ws->k4_cb || !ws->k4_cc || !ws->k4_cd || !ws->k4_temp || !ws->k4_tj ||
        !ws->temp_ca || !ws->temp_cb || !ws->temp_cc || !ws->temp_cd || !ws->temp_temp || !ws->temp_tj) {
        cstr_series_parallel_workspace_free(ws);
        return NULL;
    }

    return ws;
}

void cstr_series_parallel_workspace_free(CstrSeriesParallelWorkspace *ws) {
    if (!ws) return;
    free(ws->k1_ca); free(ws->k1_cb); free(ws->k1_cc); free(ws->k1_cd); free(ws->k1_temp); free(ws->k1_tj);
    free(ws->k2_ca); free(ws->k2_cb); free(ws->k2_cc); free(ws->k2_cd); free(ws->k2_temp); free(ws->k2_tj);
    free(ws->k3_ca); free(ws->k3_cb); free(ws->k3_cc); free(ws->k3_cd); free(ws->k3_temp); free(ws->k3_tj);
    free(ws->k4_ca); free(ws->k4_cb); free(ws->k4_cc); free(ws->k4_cd); free(ws->k4_temp); free(ws->k4_tj);
    free(ws->temp_ca); free(ws->temp_cb); free(ws->temp_cc); free(ws->temp_cd); free(ws->temp_temp); free(ws->temp_tj);
    free(ws);
}

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
) {
    if (n == 0 || !params || !ca || !cb || !cc || !cd || !temp || !temp_j ||
        !out_dcadt || !out_dcbdt || !out_dccdt || !out_dcddt || !out_dtdt || !out_dtjdt) {
        return;
    }

    const double space_time_inv  = params->volumetric_flow_l_per_min / params->volume_liters;
    const double qj_over_vj      = params->jacket_flow_l_per_min / params->jacket_volume_liters;
    const double h1_coeff        = params->delta_h1_j_per_mol / params->rho_cp_j_per_l_k;
    const double h2_coeff        = params->delta_h2_j_per_mol / params->rho_cp_j_per_l_k;
    const double h3_coeff        = params->delta_h3_j_per_mol / params->rho_cp_j_per_l_k;
    const double ua_reactor_coeff = params->ua_j_per_min_k / (params->volume_liters * params->rho_cp_j_per_l_k);
    const double ua_jacket_coeff  = params->ua_j_per_min_k / (params->jacket_volume_liters * params->jacket_rho_cp_j_per_l_k);

#if REACTOR_CORE_HAS_AVX512
    const __m512d v_space_inv = _mm512_set1_pd(space_time_inv);
    const __m512d v_qj_vj     = _mm512_set1_pd(qj_over_vj);
    const __m512d v_h1        = _mm512_set1_pd(h1_coeff);
    const __m512d v_h2        = _mm512_set1_pd(h2_coeff);
    const __m512d v_h3        = _mm512_set1_pd(h3_coeff);
    const __m512d v_ua_r      = _mm512_set1_pd(ua_reactor_coeff);
    const __m512d v_ua_j      = _mm512_set1_pd(ua_jacket_coeff);

    const __m512d v_ca0 = _mm512_set1_pd(params->ca_feed_mol_per_l);
    const __m512d v_cb0 = _mm512_set1_pd(params->cb_feed_mol_per_l);
    const __m512d v_cc0 = _mm512_set1_pd(params->cc_feed_mol_per_l);
    const __m512d v_cd0 = _mm512_set1_pd(params->cd_feed_mol_per_l);
    const __m512d v_t0  = _mm512_set1_pd(params->t_feed_kelvin);
    const __m512d v_tj0 = _mm512_set1_pd(params->jacket_t_in_kelvin);

    const __m512d v_k0_1 = _mm512_set1_pd(params->k0_1_per_min);
    const __m512d v_ea1  = _mm512_set1_pd(params->ea1_over_r_kelvin);
    const __m512d v_k0_2 = _mm512_set1_pd(params->k0_2_per_min);
    const __m512d v_ea2  = _mm512_set1_pd(params->ea2_over_r_kelvin);
    const __m512d v_k0_3 = _mm512_set1_pd(params->k0_3_per_min);
    const __m512d v_ea3  = _mm512_set1_pd(params->ea3_over_r_kelvin);

    const __m512d v_zero  = _mm512_setzero_pd();
    const __m512d v_t_min = _mm512_set1_pd(50.0);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d vCa = _mm512_max_pd(_mm512_loadu_pd(&ca[i]), v_zero);
        __m512d vCb = _mm512_max_pd(_mm512_loadu_pd(&cb[i]), v_zero);
        __m512d vCc = _mm512_max_pd(_mm512_loadu_pd(&cc[i]), v_zero);
        __m512d vCd = _mm512_max_pd(_mm512_loadu_pd(&cd[i]), v_zero);
        __m512d vT  = _mm512_max_pd(_mm512_loadu_pd(&temp[i]), v_t_min);
        __m512d vTj = _mm512_max_pd(_mm512_loadu_pd(&temp_j[i]), v_t_min);

        /* r1 = k0_1 * exp(-Ea1 / T) * Ca */
        __m512d vExp1 = simd_exp512_pd(_mm512_div_pd(_mm512_sub_pd(v_zero, v_ea1), vT));
        __m512d vr1   = _mm512_mul_pd(_mm512_mul_pd(v_k0_1, vExp1), vCa);

        /* r2 = k0_2 * exp(-Ea2 / T) * Cb */
        __m512d vExp2 = simd_exp512_pd(_mm512_div_pd(_mm512_sub_pd(v_zero, v_ea2), vT));
        __m512d vr2   = _mm512_mul_pd(_mm512_mul_pd(v_k0_2, vExp2), vCb);

        /* r3 = k0_3 * exp(-Ea3 / T) * Ca */
        __m512d vExp3 = simd_exp512_pd(_mm512_div_pd(_mm512_sub_pd(v_zero, v_ea3), vT));
        __m512d vr3   = _mm512_mul_pd(_mm512_mul_pd(v_k0_3, vExp3), vCa);

        /* dCa/dt = (q/V)*(CA0 - Ca) - r1 - r3 */
        __m512d vdCadt = _mm512_sub_pd(_mm512_fmsub_pd(v_space_inv, _mm512_sub_pd(v_ca0, vCa), vr1), vr3);

        /* dCb/dt = (q/V)*(CB0 - Cb) + r1 - r2 */
        __m512d vdCbdt = _mm512_sub_pd(_mm512_fmadd_pd(v_space_inv, _mm512_sub_pd(v_cb0, vCb), vr1), vr2);

        /* dCc/dt = (q/V)*(CC0 - Cc) + r2 */
        __m512d vdCcdt = _mm512_fmadd_pd(v_space_inv, _mm512_sub_pd(v_cc0, vCc), vr2);

        /* dCd/dt = (q/V)*(CD0 - Cd) + r3 */
        __m512d vdCddt = _mm512_fmadd_pd(v_space_inv, _mm512_sub_pd(v_cd0, vCd), vr3);

        /* dT/dt = (q/V)*(T0 - T) + h1*r1 + h2*r2 + h3*r3 - ua_r*(T - Tj) */
        __m512d vdTdt = _mm512_mul_pd(v_space_inv, _mm512_sub_pd(v_t0, vT));
        vdTdt = _mm512_fmadd_pd(v_h1, vr1, vdTdt);
        vdTdt = _mm512_fmadd_pd(v_h2, vr2, vdTdt);
        vdTdt = _mm512_fmadd_pd(v_h3, vr3, vdTdt);
        vdTdt = _mm512_fnmadd_pd(v_ua_r, _mm512_sub_pd(vT, vTj), vdTdt);

        /* dTj/dt = (qj/Vj)*(Tj_in - Tj) + ua_j*(T - Tj) */
        __m512d vdTjdt = _mm512_fmadd_pd(v_qj_vj, _mm512_sub_pd(v_tj0, vTj),
                                         _mm512_mul_pd(v_ua_j, _mm512_sub_pd(vT, vTj)));

        _mm512_storeu_pd(&out_dcadt[i], vdCadt);
        _mm512_storeu_pd(&out_dcbdt[i], vdCbdt);
        _mm512_storeu_pd(&out_dccdt[i], vdCcdt);
        _mm512_storeu_pd(&out_dcddt[i], vdCddt);
        _mm512_storeu_pd(&out_dtdt[i], vdTdt);
        _mm512_storeu_pd(&out_dtjdt[i], vdTjdt);
    }
#elif REACTOR_CORE_HAS_AVX2
    const __m256d v_space_inv = _mm256_set1_pd(space_time_inv);
    const __m256d v_qj_vj     = _mm256_set1_pd(qj_over_vj);
    const __m256d v_h1        = _mm256_set1_pd(h1_coeff);
    const __m256d v_h2        = _mm256_set1_pd(h2_coeff);
    const __m256d v_h3        = _mm256_set1_pd(h3_coeff);
    const __m256d v_ua_r      = _mm256_set1_pd(ua_reactor_coeff);
    const __m256d v_ua_j      = _mm256_set1_pd(ua_jacket_coeff);

    const __m256d v_ca0 = _mm256_set1_pd(params->ca_feed_mol_per_l);
    const __m256d v_cb0 = _mm256_set1_pd(params->cb_feed_mol_per_l);
    const __m256d v_cc0 = _mm256_set1_pd(params->cc_feed_mol_per_l);
    const __m256d v_cd0 = _mm256_set1_pd(params->cd_feed_mol_per_l);
    const __m256d v_t0  = _mm256_set1_pd(params->t_feed_kelvin);
    const __m256d v_tj0 = _mm256_set1_pd(params->jacket_t_in_kelvin);

    const __m256d v_k0_1 = _mm256_set1_pd(params->k0_1_per_min);
    const __m256d v_ea1  = _mm256_set1_pd(params->ea1_over_r_kelvin);
    const __m256d v_k0_2 = _mm256_set1_pd(params->k0_2_per_min);
    const __m256d v_ea2  = _mm256_set1_pd(params->ea2_over_r_kelvin);
    const __m256d v_k0_3 = _mm256_set1_pd(params->k0_3_per_min);
    const __m256d v_ea3  = _mm256_set1_pd(params->ea3_over_r_kelvin);

    const __m256d v_zero  = _mm256_setzero_pd();
    const __m256d v_t_min = _mm256_set1_pd(50.0);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vCa = _mm256_max_pd(_mm256_loadu_pd(&ca[i]), v_zero);
        __m256d vCb = _mm256_max_pd(_mm256_loadu_pd(&cb[i]), v_zero);
        __m256d vCc = _mm256_max_pd(_mm256_loadu_pd(&cc[i]), v_zero);
        __m256d vCd = _mm256_max_pd(_mm256_loadu_pd(&cd[i]), v_zero);
        __m256d vT  = _mm256_max_pd(_mm256_loadu_pd(&temp[i]), v_t_min);
        __m256d vTj = _mm256_max_pd(_mm256_loadu_pd(&temp_j[i]), v_t_min);

        /* r1 = k0_1 * exp(-Ea1 / T) * Ca */
        __m256d vExp1 = simd_exp256_pd(_mm256_div_pd(_mm256_sub_pd(v_zero, v_ea1), vT));
        __m256d vr1   = _mm256_mul_pd(_mm256_mul_pd(v_k0_1, vExp1), vCa);

        /* r2 = k0_2 * exp(-Ea2 / T) * Cb */
        __m256d vExp2 = simd_exp256_pd(_mm256_div_pd(_mm256_sub_pd(v_zero, v_ea2), vT));
        __m256d vr2   = _mm256_mul_pd(_mm256_mul_pd(v_k0_2, vExp2), vCb);

        /* r3 = k0_3 * exp(-Ea3 / T) * Ca */
        __m256d vExp3 = simd_exp256_pd(_mm256_div_pd(_mm256_sub_pd(v_zero, v_ea3), vT));
        __m256d vr3   = _mm256_mul_pd(_mm256_mul_pd(v_k0_3, vExp3), vCa);

        /* dCa/dt = (q/V)*(CA0 - Ca) - r1 - r3 */
        __m256d vdCadt = _mm256_sub_pd(_mm256_fmsub_pd(v_space_inv, _mm256_sub_pd(v_ca0, vCa), vr1), vr3);

        /* dCb/dt = (q/V)*(CB0 - Cb) + r1 - r2 */
        __m256d vdCbdt = _mm256_sub_pd(_mm256_fmadd_pd(v_space_inv, _mm256_sub_pd(v_cb0, vCb), vr1), vr2);

        /* dCc/dt = (q/V)*(CC0 - Cc) + r2 */
        __m256d vdCcdt = _mm256_fmadd_pd(v_space_inv, _mm256_sub_pd(v_cc0, vCc), vr2);

        /* dCd/dt = (q/V)*(CD0 - Cd) + r3 */
        __m256d vdCddt = _mm256_fmadd_pd(v_space_inv, _mm256_sub_pd(v_cd0, vCd), vr3);

        /* dT/dt = (q/V)*(T0 - T) + h1*r1 + h2*r2 + h3*r3 - ua_r*(T - Tj) */
        __m256d vdTdt = _mm256_mul_pd(v_space_inv, _mm256_sub_pd(v_t0, vT));
        vdTdt = _mm256_fmadd_pd(v_h1, vr1, vdTdt);
        vdTdt = _mm256_fmadd_pd(v_h2, vr2, vdTdt);
        vdTdt = _mm256_fmadd_pd(v_h3, vr3, vdTdt);
        vdTdt = _mm256_fnmadd_pd(v_ua_r, _mm256_sub_pd(vT, vTj), vdTdt);

        /* dTj/dt = (qj/Vj)*(Tj_in - Tj) + ua_j*(T - Tj) */
        __m256d vdTjdt = _mm256_fmadd_pd(v_qj_vj, _mm256_sub_pd(v_tj0, vTj),
                                         _mm256_mul_pd(v_ua_j, _mm256_sub_pd(vT, vTj)));

        _mm256_storeu_pd(&out_dcadt[i], vdCadt);
        _mm256_storeu_pd(&out_dcbdt[i], vdCbdt);
        _mm256_storeu_pd(&out_dccdt[i], vdCcdt);
        _mm256_storeu_pd(&out_dcddt[i], vdCddt);
        _mm256_storeu_pd(&out_dtdt[i], vdTdt);
        _mm256_storeu_pd(&out_dtjdt[i], vdTjdt);
    }
#else
    size_t i = 0;
#endif

    /* Scalar remainder handling */
    for (; i < n; i++) {
        double Ca_val = ca[i] < 0.0 ? 0.0 : ca[i];
        double Cb_val = cb[i] < 0.0 ? 0.0 : cb[i];
        double Cc_val = cc[i] < 0.0 ? 0.0 : cc[i];
        double Cd_val = cd[i] < 0.0 ? 0.0 : cd[i];
        double T_val  = temp[i] < 50.0 ? 50.0 : temp[i];
        double Tj_val = temp_j[i] < 50.0 ? 50.0 : temp_j[i];

        double r1 = params->k0_1_per_min * exp(-params->ea1_over_r_kelvin / T_val) * Ca_val;
        double r2 = params->k0_2_per_min * exp(-params->ea2_over_r_kelvin / T_val) * Cb_val;
        double r3 = params->k0_3_per_min * exp(-params->ea3_over_r_kelvin / T_val) * Ca_val;

        double heat_gen = h1_coeff * r1 + h2_coeff * r2 + h3_coeff * r3;
        double heat_rem_r = ua_reactor_coeff * (T_val - Tj_val);
        double heat_rem_j = ua_jacket_coeff * (T_val - Tj_val);

        out_dcadt[i] = space_time_inv * (params->ca_feed_mol_per_l - Ca_val) - r1 - r3;
        out_dcbdt[i] = space_time_inv * (params->cb_feed_mol_per_l - Cb_val) + r1 - r2;
        out_dccdt[i] = space_time_inv * (params->cc_feed_mol_per_l - Cc_val) + r2;
        out_dcddt[i] = space_time_inv * (params->cd_feed_mol_per_l - Cd_val) + r3;
        out_dtdt[i]  = space_time_inv * (params->t_feed_kelvin - T_val) + heat_gen - heat_rem_r;
        out_dtjdt[i] = qj_over_vj * (params->jacket_t_in_kelvin - Tj_val) + heat_rem_j;
    }
}

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
) {
    if (!ws || ws->capacity < n || n == 0 || !params || !ca || !cb || !cc || !cd || !temp || !temp_j) {
        return false;
    }

    const double half_dt = 0.5 * dt;

    /* Stage 1: k1 = f(y) */
    cstr_series_parallel_batch_derivatives(n, params, ca, cb, cc, cd, temp, temp_j,
        ws->k1_ca, ws->k1_cb, ws->k1_cc, ws->k1_cd, ws->k1_temp, ws->k1_tj);

    /* Stage 2 prep: y_stage = y + half_dt * k1 */
    for (size_t i = 0; i < n; i++) {
        ws->temp_ca[i]   = ca[i] + half_dt * ws->k1_ca[i];
        ws->temp_cb[i]   = cb[i] + half_dt * ws->k1_cb[i];
        ws->temp_cc[i]   = cc[i] + half_dt * ws->k1_cc[i];
        ws->temp_cd[i]   = cd[i] + half_dt * ws->k1_cd[i];
        ws->temp_temp[i] = temp[i] + half_dt * ws->k1_temp[i];
        ws->temp_tj[i]   = temp_j[i] + half_dt * ws->k1_tj[i];
    }
    cstr_series_parallel_batch_derivatives(n, params, ws->temp_ca, ws->temp_cb, ws->temp_cc, ws->temp_cd, ws->temp_temp, ws->temp_tj,
        ws->k2_ca, ws->k2_cb, ws->k2_cc, ws->k2_cd, ws->k2_temp, ws->k2_tj);

    /* Stage 3 prep: y_stage = y + half_dt * k2 */
    for (size_t i = 0; i < n; i++) {
        ws->temp_ca[i]   = ca[i] + half_dt * ws->k2_ca[i];
        ws->temp_cb[i]   = cb[i] + half_dt * ws->k2_cb[i];
        ws->temp_cc[i]   = cc[i] + half_dt * ws->k2_cc[i];
        ws->temp_cd[i]   = cd[i] + half_dt * ws->k2_cd[i];
        ws->temp_temp[i] = temp[i] + half_dt * ws->k2_temp[i];
        ws->temp_tj[i]   = temp_j[i] + half_dt * ws->k2_tj[i];
    }
    cstr_series_parallel_batch_derivatives(n, params, ws->temp_ca, ws->temp_cb, ws->temp_cc, ws->temp_cd, ws->temp_temp, ws->temp_tj,
        ws->k3_ca, ws->k3_cb, ws->k3_cc, ws->k3_cd, ws->k3_temp, ws->k3_tj);

    /* Stage 4 prep: y_stage = y + dt * k3 */
    for (size_t i = 0; i < n; i++) {
        ws->temp_ca[i]   = ca[i] + dt * ws->k3_ca[i];
        ws->temp_cb[i]   = cb[i] + dt * ws->k3_cb[i];
        ws->temp_cc[i]   = cc[i] + dt * ws->k3_cc[i];
        ws->temp_cd[i]   = cd[i] + dt * ws->k3_cd[i];
        ws->temp_temp[i] = temp[i] + dt * ws->k3_temp[i];
        ws->temp_tj[i]   = temp_j[i] + dt * ws->k3_tj[i];
    }
    cstr_series_parallel_batch_derivatives(n, params, ws->temp_ca, ws->temp_cb, ws->temp_cc, ws->temp_cd, ws->temp_temp, ws->temp_tj,
        ws->k4_ca, ws->k4_cb, ws->k4_cc, ws->k4_cd, ws->k4_temp, ws->k4_tj);

    /* Simpson's accumulation */
    const double dt_six = dt / 6.0;
    for (size_t i = 0; i < n; i++) {
        ca[i]     += dt_six * (ws->k1_ca[i] + 2.0 * ws->k2_ca[i] + 2.0 * ws->k3_ca[i] + ws->k4_ca[i]);
        cb[i]     += dt_six * (ws->k1_cb[i] + 2.0 * ws->k2_cb[i] + 2.0 * ws->k3_cb[i] + ws->k4_cb[i]);
        cc[i]     += dt_six * (ws->k1_cc[i] + 2.0 * ws->k2_cc[i] + 2.0 * ws->k3_cc[i] + ws->k4_cc[i]);
        cd[i]     += dt_six * (ws->k1_cd[i] + 2.0 * ws->k2_cd[i] + 2.0 * ws->k3_cd[i] + ws->k4_cd[i]);
        temp[i]   += dt_six * (ws->k1_temp[i] + 2.0 * ws->k2_temp[i] + 2.0 * ws->k3_temp[i] + ws->k4_temp[i]);
        temp_j[i] += dt_six * (ws->k1_tj[i] + 2.0 * ws->k2_tj[i] + 2.0 * ws->k3_tj[i] + ws->k4_tj[i]);

        if (ca[i] < 0.0) ca[i] = 0.0;
        if (cb[i] < 0.0) cb[i] = 0.0;
        if (cc[i] < 0.0) cc[i] = 0.0;
        if (cd[i] < 0.0) cd[i] = 0.0;
        if (temp[i] < 50.0) temp[i] = 50.0;
        if (temp_j[i] < 50.0) temp_j[i] = 50.0;
    }

    return true;
}

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
) {
    if (n == 0 || !params || !ca || !cb || !cc || !cd || !temp || !temp_j) {
        return false;
    }

    CstrSeriesParallelWorkspace *ws = cstr_series_parallel_workspace_create(n);
    if (!ws) return false;

    bool ok = cstr_series_parallel_batch_step_rk4_ws(ws, n, params, ca, cb, cc, cd, temp, temp_j, dt);
    cstr_series_parallel_workspace_free(ws);
    return ok;
}

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
) {
    if (n == 0 || steps == 0 || dt <= 0.0 || !params ||
        !ca0 || !cb0 || !cc0 || !cd0 || !temp0 || !temp_j0 ||
        !out_ca || !out_cb || !out_cc || !out_cd || !out_temp || !out_temp_j) {
        return false;
    }

    CstrSeriesParallelWorkspace *ws = cstr_series_parallel_workspace_create(n);
    if (!ws) return false;

    if (out_time) out_time[0] = 0.0;
    for (size_t j = 0; j < n; j++) {
        out_ca[j]     = ca0[j];
        out_cb[j]     = cb0[j];
        out_cc[j]     = cc0[j];
        out_cd[j]     = cd0[j];
        out_temp[j]   = temp0[j];
        out_temp_j[j] = temp_j0[j];
    }

    double *curr_ca     = (double*)malloc(n * sizeof(double));
    double *curr_cb     = (double*)malloc(n * sizeof(double));
    double *curr_cc     = (double*)malloc(n * sizeof(double));
    double *curr_cd     = (double*)malloc(n * sizeof(double));
    double *curr_temp   = (double*)malloc(n * sizeof(double));
    double *curr_temp_j = (double*)malloc(n * sizeof(double));

    if (!curr_ca || !curr_cb || !curr_cc || !curr_cd || !curr_temp || !curr_temp_j) {
        free(curr_ca); free(curr_cb); free(curr_cc); free(curr_cd); free(curr_temp); free(curr_temp_j);
        cstr_series_parallel_workspace_free(ws);
        return false;
    }

    memcpy(curr_ca, ca0, n * sizeof(double));
    memcpy(curr_cb, cb0, n * sizeof(double));
    memcpy(curr_cc, cc0, n * sizeof(double));
    memcpy(curr_cd, cd0, n * sizeof(double));
    memcpy(curr_temp, temp0, n * sizeof(double));
    memcpy(curr_temp_j, temp_j0, n * sizeof(double));

    for (size_t s = 1; s <= steps; s++) {
        if (!cstr_series_parallel_batch_step_rk4_ws(ws, n, params, curr_ca, curr_cb, curr_cc, curr_cd, curr_temp, curr_temp_j, dt)) {
            free(curr_ca); free(curr_cb); free(curr_cc); free(curr_cd); free(curr_temp); free(curr_temp_j);
            cstr_series_parallel_workspace_free(ws);
            return false;
        }

        if (out_time) out_time[s] = (double)s * dt;
        size_t offset = s * n;
        for (size_t j = 0; j < n; j++) {
            out_ca[offset + j]     = curr_ca[j];
            out_cb[offset + j]     = curr_cb[j];
            out_cc[offset + j]     = curr_cc[j];
            out_cd[offset + j]     = curr_cd[j];
            out_temp[offset + j]   = curr_temp[j];
            out_temp_j[offset + j] = curr_temp_j[j];
        }
    }

    free(curr_ca); free(curr_cb); free(curr_cc); free(curr_cd); free(curr_temp); free(curr_temp_j);
    cstr_series_parallel_workspace_free(ws);
    return true;
}
