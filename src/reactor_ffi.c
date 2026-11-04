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
