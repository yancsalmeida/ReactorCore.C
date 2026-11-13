#include "reactor_ffi.h"
#include "cstr.h"
#include "rk4.h"
#include "vector.h"
#include <stdlib.h>
#include <string.h>

#define REACTOR_CORE_VERSION_STRING "ReactorCore.C 1.1.0-native-ffi (C11 x86_64)"

const char* reactor_core_version(void) {
    return REACTOR_CORE_VERSION_STRING;
}

int reactor_cstr_default_params(CstrParamsFFI *out_params) {
    if (!out_params) {
        return -1;
    }

    CstrParameters def = cstr_default_parameters();
    memcpy(out_params, &def, sizeof(CstrParamsFFI));
    return 0;
}

double reactor_cstr_reaction_rate(const CstrParamsFFI *params, double Ca, double T) {
    if (!params) {
        return 0.0;
    }
    return cstr_reaction_rate((const CstrParameters*)params, Ca, T);
}

int reactor_cstr_step_rk4(
    const CstrParamsFFI *params,
    double *ca,
    double *temp_k,
    double dt
) {
    if (!params || !ca || !temp_k || dt <= 0.0) {
        return -1;
    }

    static _Thread_local Rk4Workspace *ws = NULL;
    static _Thread_local Vector *y = NULL;
    static _Thread_local Vector *y_next = NULL;

    if (!ws) {
        ws = rk4_workspace_create(2);
        if (!ws) return -1;
    }
    if (!y) {
        y = vector_create(2);
        if (!y) return -1;
    }
    if (!y_next) {
        y_next = vector_create(2);
        if (!y_next) return -1;
    }

    y->data[0] = *ca;
    y->data[1] = *temp_k;

    bool ok = rk4_step(ws, cstr_derivatives, 0.0, y, y_next, dt, (void*)params);
    if (!ok) {
        return -1;
    }

    *ca = y_next->data[0];
    *temp_k = y_next->data[1];
    return 0;
}

int reactor_cstr_step_detailed(
    const CstrParamsFFI *params,
    double *ca,
    double *temp_k,
    double dt,
    double runaway_threshold_k,
    CstrStateFFI *out_state
) {
    if (!params || !ca || !temp_k || !out_state || dt <= 0.0) {
        return -1;
    }

    int rc = reactor_cstr_step_rk4(params, ca, temp_k, dt);
    if (rc != 0) {
        return rc;
    }

    double new_ca = *ca;
    double new_temp = *temp_k;

    out_state->ca_mol_per_l = new_ca;
    out_state->temp_kelvin = new_temp;
    out_state->conversion_pct = (1.0 - (new_ca / params->ca_feed_mol_per_l)) * 100.0;
    out_state->reaction_rate_mol_per_l_min = reactor_cstr_reaction_rate(params, new_ca, new_temp);
    out_state->is_runaway = (new_temp >= runaway_threshold_k);

    return 0;
}

int reactor_cstr_simulate_trajectory(
    const CstrParamsFFI *params,
    double ca0,
    double temp0,
    double dt,
    size_t steps,
    double *out_time,
    double *out_ca,
    double *out_temp
) {
    if (!params || !out_time || !out_ca || !out_temp || dt <= 0.0 || steps == 0) {
        return -1;
    }

    Rk4Workspace *ws = rk4_workspace_create(2);
    if (!ws) {
        return -1;
    }

    Vector *y = vector_create(2);
    Vector *y_next = vector_create(2);
    if (!y || !y_next) {
        rk4_workspace_free(ws);
        vector_free(y);
        vector_free(y_next);
        return -1;
    }

    y->data[0] = ca0;
    y->data[1] = temp0;

    out_time[0] = 0.0;
    out_ca[0] = ca0;
    out_temp[0] = temp0;

    double t = 0.0;
    for (size_t i = 1; i <= steps; i++) {
        if (!rk4_step(ws, cstr_derivatives, t, y, y_next, dt, (void*)params)) {
            rk4_workspace_free(ws);
            vector_free(y);
            vector_free(y_next);
            return -1;
        }

        t += dt;
        y->data[0] = y_next->data[0];
        y->data[1] = y_next->data[1];

        out_time[i] = t;
        out_ca[i] = y->data[0];
        out_temp[i] = y->data[1];
    }

    rk4_workspace_free(ws);
    vector_free(y);
    vector_free(y_next);
    return 0;
}

int reactor_cstr_step_adaptive(
    const CstrParamsFFI *params,
    double *ca,
    double *temp_k,
    double dt,
    double rtol,
    double atol,
    CstrAdaptiveStatsFFI *out_stats
) {
    if (!params || !ca || !temp_k || dt <= 0.0) {
        return -1;
    }

    static _Thread_local Rk45Workspace *ws_rk45 = NULL;
    static _Thread_local Vector *y_rk45 = NULL;

    if (!ws_rk45) {
        ws_rk45 = rk45_workspace_create(2);
        if (!ws_rk45) return -1;
    }
    if (!y_rk45) {
        y_rk45 = vector_create(2);
        if (!y_rk45) return -1;
    }

    y_rk45->data[0] = *ca;
    y_rk45->data[1] = *temp_k;

    Rk45Control ctrl = rk45_default_control();
    if (rtol > 0.0) ctrl.rtol = rtol;
    if (atol > 0.0) ctrl.atol = atol;
    ctrl.h_max = dt;

    Rk45AdaptiveStats stats;
    bool ok = rk45_integrate_interval(
        ws_rk45,
        cstr_derivatives,
        0.0,
        dt,
        y_rk45,
        dt / 10.0,
        &ctrl,
        &stats,
        (void*)params
    );

    if (!ok) {
        return -1;
    }

    *ca = y_rk45->data[0];
    *temp_k = y_rk45->data[1];

    if (out_stats) {
        out_stats->steps_accepted = stats.steps_accepted;
        out_stats->steps_rejected = stats.steps_rejected;
        out_stats->final_h = stats.final_h;
        out_stats->min_h = stats.min_h;
        out_stats->max_local_error = stats.max_local_error;
    }

    return 0;
}

int reactor_cstr_step_detailed_adaptive(
    const CstrParamsFFI *params,
    double *ca,
    double *temp_k,
    double dt,
    double rtol,
    double atol,
    double runaway_threshold_k,
    CstrStateFFI *out_state,
    CstrAdaptiveStatsFFI *out_stats
) {
    if (!params || !ca || !temp_k || !out_state || dt <= 0.0) {
        return -1;
    }

    int rc = reactor_cstr_step_adaptive(params, ca, temp_k, dt, rtol, atol, out_stats);
    if (rc != 0) {
        return rc;
    }

    double new_ca = *ca;
    double new_temp = *temp_k;

    out_state->ca_mol_per_l = new_ca;
    out_state->temp_kelvin = new_temp;
    out_state->conversion_pct = (1.0 - (new_ca / params->ca_feed_mol_per_l)) * 100.0;
    out_state->reaction_rate_mol_per_l_min = reactor_cstr_reaction_rate(params, new_ca, new_temp);
    out_state->is_runaway = (new_temp >= runaway_threshold_k);

    return 0;
}

int reactor_cstr_simulate_adaptive_trajectory(
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
) {
    if (!params || !out_time || !out_ca || !out_temp || dt <= 0.0 || steps == 0) {
        return -1;
    }

    Rk45Workspace *ws = rk45_workspace_create(2);
    if (!ws) {
        return -1;
    }

    Vector *y = vector_create(2);
    if (!y) {
        rk45_workspace_free(ws);
        return -1;
    }

    y->data[0] = ca0;
    y->data[1] = temp0;

    out_time[0] = 0.0;
    out_ca[0] = ca0;
    out_temp[0] = temp0;

    Rk45Control ctrl = rk45_default_control();
    if (rtol > 0.0) ctrl.rtol = rtol;
    if (atol > 0.0) ctrl.atol = atol;
    ctrl.h_max = dt;

    size_t total_acc = 0;
    size_t total_rej = 0;
    double global_min_h = 1e30;
    double global_max_err = 0.0;
    double last_h = dt / 10.0;

    double t = 0.0;
    for (size_t i = 1; i <= steps; i++) {
        Rk45AdaptiveStats step_stats;
        bool ok = rk45_integrate_interval(
            ws,
            cstr_derivatives,
            t,
            t + dt,
            y,
            last_h,
            &ctrl,
            &step_stats,
            (void*)params
        );

        if (!ok) {
            rk45_workspace_free(ws);
            vector_free(y);
            return -1;
        }

        t += dt;
        out_time[i] = t;
        out_ca[i] = y->data[0];
        out_temp[i] = y->data[1];

        total_acc += step_stats.steps_accepted;
        total_rej += step_stats.steps_rejected;
        if (step_stats.min_h < global_min_h) {
            global_min_h = step_stats.min_h;
        }
        if (step_stats.max_local_error > global_max_err) {
            global_max_err = step_stats.max_local_error;
        }
        last_h = step_stats.final_h;
    }

    if (out_stats) {
        out_stats->steps_accepted = total_acc;
        out_stats->steps_rejected = total_rej;
        out_stats->final_h = last_h;
        out_stats->min_h = global_min_h;
        out_stats->max_local_error = global_max_err;
    }

    rk45_workspace_free(ws);
    vector_free(y);
    return 0;
}

#include "simd_math.h"

int reactor_core_has_simd(void) {
    return simd_avx2_fma_available();
}

int reactor_cstr_batch_step_rk4(
    const CstrParamsFFI *params,
    double *ca_array,
    double *temp_array,
    size_t n_reactors,
    double dt
) {
    if (!params || !ca_array || !temp_array || n_reactors == 0 || dt <= 0.0) {
        return -1;
    }

    bool ok = cstr_batch_step_rk4(
        n_reactors,
        (const CstrParameters*)params,
        ca_array,
        temp_array,
        dt
    );

    return ok ? 0 : -1;
}

int reactor_cstr_batch_step_detailed(
    const CstrParamsFFI *params,
    double *ca_array,
    double *temp_array,
    size_t n_reactors,
    double dt,
    double runaway_threshold_k,
    CstrStateFFI *out_states
) {
    if (!params || !ca_array || !temp_array || !out_states || n_reactors == 0 || dt <= 0.0) {
        return -1;
    }

    bool ok = cstr_batch_step_rk4(
        n_reactors,
        (const CstrParameters*)params,
        ca_array,
        temp_array,
        dt
    );
    if (!ok) return -1;

    const CstrParameters *p = (const CstrParameters*)params;
    for (size_t i = 0; i < n_reactors; i++) {
        double ca = ca_array[i];
        double temp = temp_array[i];
        double r = cstr_reaction_rate(p, ca, temp);
        double conversion = 0.0;
        if (p->ca_feed_mol_per_l > 1e-12) {
            conversion = ((p->ca_feed_mol_per_l - ca) / p->ca_feed_mol_per_l) * 100.0;
            if (conversion < 0.0) conversion = 0.0;
            if (conversion > 100.0) conversion = 100.0;
        }

        out_states[i].ca_mol_per_l = ca;
        out_states[i].temp_kelvin = temp;
        out_states[i].conversion_pct = conversion;
        out_states[i].reaction_rate_mol_per_l_min = r;
        out_states[i].is_runaway = (temp >= runaway_threshold_k);
    }

    return 0;
}

int reactor_cstr_batch_simulate_trajectory(
    const CstrParamsFFI *params,
    const double *ca0_array,
    const double *temp0_array,
    size_t n_reactors,
    double dt,
    size_t steps,
    double *out_time,
    double *out_ca,
    double *out_temp
) {
    if (!params || !ca0_array || !temp0_array || !out_ca || !out_temp || n_reactors == 0 || dt <= 0.0) {
        return -1;
    }

    bool ok = cstr_batch_simulate_trajectory(
        n_reactors,
        (const CstrParameters*)params,
        ca0_array,
        temp0_array,
        dt,
        steps,
        out_time,
        out_ca,
        out_temp
    );

    return ok ? 0 : -1;
}

int reactor_core_simd_level(void) {
#if REACTOR_CORE_HAS_AVX512
    return 512;
#elif REACTOR_CORE_HAS_AVX2
    return 256;
#else
    return 0;
#endif
}

/* ========================================================================= */
/* Complex Series-Parallel Kinetics & Dynamic Jacket FFI Implementation      */
/* ========================================================================= */

int reactor_cstr_series_parallel_default_params(CstrSeriesParallelParamsFFI *out_params) {
    if (!out_params) return -1;
    CstrSeriesParallelParams def = cstr_series_parallel_default_parameters();
    memcpy(out_params, &def, sizeof(CstrSeriesParallelParamsFFI));
    return 0;
}

int reactor_cstr_series_parallel_rates(
    const CstrSeriesParallelParamsFFI *params,
    double Ca,
    double Cb,
    double T,
    double *out_r1,
    double *out_r2,
    double *out_r3
) {
    if (!params) return -1;
    cstr_series_parallel_rates((const CstrSeriesParallelParams*)params, Ca, Cb, T, out_r1, out_r2, out_r3);
    return 0;
}

int reactor_cstr_series_parallel_step_rk4(
    const CstrSeriesParallelParamsFFI *params,
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
    return cstr_series_parallel_step_rk4(
        (const CstrSeriesParallelParams*)params,
        ca, cb, cc, cd, temp, temp_j, dt
    );
}

int reactor_cstr_series_parallel_step_detailed(
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
) {
    if (!params || !ca || !cb || !cc || !cd || !temp || !temp_j || !out_state || dt <= 0.0) {
        return -1;
    }
    return cstr_series_parallel_step_detailed(
        (const CstrSeriesParallelParams*)params,
        ca, cb, cc, cd, temp, temp_j, dt, runaway_threshold_k,
        (CstrSeriesParallelState*)out_state
    );
}

int reactor_cstr_series_parallel_batch_step_rk4(
    const CstrSeriesParallelParamsFFI *params,
    double *ca_array,
    double *cb_array,
    double *cc_array,
    double *cd_array,
    double *temp_array,
    double *temp_j_array,
    size_t n_reactors,
    double dt
) {
    if (!params || !ca_array || !cb_array || !cc_array || !cd_array ||
        !temp_array || !temp_j_array || n_reactors == 0 || dt <= 0.0) {
        return -1;
    }

    bool ok = cstr_series_parallel_batch_step_rk4(
        n_reactors,
        (const CstrSeriesParallelParams*)params,
        ca_array,
        cb_array,
        cc_array,
        cd_array,
        temp_array,
        temp_j_array,
        dt
    );

    return ok ? 0 : -1;
}

int reactor_cstr_series_parallel_batch_step_detailed(
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
) {
    int rc = reactor_cstr_series_parallel_batch_step_rk4(
        params, ca_array, cb_array, cc_array, cd_array, temp_array, temp_j_array, n_reactors, dt
    );
    if (rc != 0 || !out_states) return rc;

    const CstrSeriesParallelParams *p = (const CstrSeriesParallelParams*)params;
    double ca0 = p->ca_feed_mol_per_l > 1e-12 ? p->ca_feed_mol_per_l : 1.0;

    for (size_t i = 0; i < n_reactors; i++) {
        double ca     = ca_array[i];
        double cb     = cb_array[i];
        double cc     = cc_array[i];
        double cd     = cd_array[i];
        double temp   = temp_array[i];
        double temp_j = temp_j_array[i];

        double r1, r2, r3;
        cstr_series_parallel_rates(p, ca, cb, temp, &r1, &r2, &r3);

        out_states[i].ca_mol_per_l       = ca;
        out_states[i].cb_mol_per_l       = cb;
        out_states[i].cc_mol_per_l       = cc;
        out_states[i].cd_mol_per_l       = cd;
        out_states[i].temp_kelvin        = temp;
        out_states[i].jacket_temp_kelvin = temp_j;

        out_states[i].conversion_a_pct   = ((p->ca_feed_mol_per_l - ca) / ca0) * 100.0;
        out_states[i].yield_b_pct        = ((cb - p->cb_feed_mol_per_l) / ca0) * 100.0;

        double total_p = cb + cc + cd;
        out_states[i].selectivity_b_pct  = total_p > 1e-12 ? (cb / total_p) * 100.0 : 100.0;

        out_states[i].rate_r1_mol_per_l_min = r1;
        out_states[i].rate_r2_mol_per_l_min = r2;
        out_states[i].rate_r3_mol_per_l_min = r3;

        double heat_gen_min = p->volume_liters *
            (p->delta_h1_j_per_mol * r1 + p->delta_h2_j_per_mol * r2 + p->delta_h3_j_per_mol * r3);
        out_states[i].heat_generated_watts = heat_gen_min / 60.0;

        double heat_rem_min = p->ua_j_per_min_k * (temp - temp_j);
        out_states[i].heat_removed_watts   = heat_rem_min / 60.0;

        out_states[i].is_thermal_runaway   = (temp >= runaway_threshold_k);
    }

    return 0;
}

int reactor_cstr_series_parallel_batch_simulate_trajectory(
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
) {
    if (!params || !ca0 || !cb0 || !cc0 || !cd0 || !temp0 || !temp_j0 ||
        !out_ca || !out_cb || !out_cc || !out_cd || !out_temp || !out_temp_j ||
        n_reactors == 0 || dt <= 0.0) {
        return -1;
    }

    bool ok = cstr_series_parallel_batch_simulate_trajectory(
        n_reactors,
        (const CstrSeriesParallelParams*)params,
        ca0, cb0, cc0, cd0, temp0, temp_j0,
        dt, steps,
        out_time,
        out_ca, out_cb, out_cc, out_cd, out_temp, out_temp_j
    );

    return ok ? 0 : -1;
}
