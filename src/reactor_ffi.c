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
