#include "stiff_solver.h"
#include <math.h>
#include <stddef.h>

/*
 * Exact L-stable 3rd-order 3-stage Rosenbrock coefficients (ROS3L)
 * with embedded 2nd-order error estimator and 2 distinct RHS evaluations:
 *   gamma^3 - 3*gamma^2 + (3/2)*gamma - 1/6 = 0  =>  R(-inf) = 0 (L-stable)
 */
static const double ROS3_GAMMA = 0.435866521508458999416;
static const double ROS3_A21   = 2.29428036027904181;
static const double ROS3_C21   = -7.55800273184117177;
static const double ROS3_C31   = 12.22566466169111443;
static const double ROS3_C32   = 5.74019720866211269;
static const double ROS3_M1    = 0.75157658521103743;
static const double ROS3_M2    = -0.57431947473051181;
static const double ROS3_M3    = 0.38238006004650693;
static const double ROS3_E1    = -0.76819871711192689;
static const double ROS3_E2    = -0.23673864922848881;
static const double ROS3_E3    = 0.38238006004650693;

static double clamp_temp(double temp) {
    if (temp < 50.0) return 50.0;
    if (temp > 2500.0) return 2500.0;
    return temp;
}

static double clamp_conc(double c) {
    return c < 0.0 ? 0.0 : c;
}

static void eval_sp_rhs(
    const CstrSeriesParallelParams *p,
    const double x[6],
    double f[6]
) {
    double ca = clamp_conc(x[0]);
    double cb = clamp_conc(x[1]);
    double cc = clamp_conc(x[2]);
    double cd = clamp_conc(x[3]);
    double t  = clamp_temp(x[4]);
    double tj = clamp_temp(x[5]);

    double r1, r2, r3;
    cstr_series_parallel_rates(p, ca, cb, t, &r1, &r2, &r3);

    double s   = p->volumetric_flow_l_per_min / p->volume_liters;
    double sj  = p->jacket_flow_l_per_min / p->jacket_volume_liters;
    double c1  = p->delta_h1_j_per_mol / p->rho_cp_j_per_l_k;
    double c2  = p->delta_h2_j_per_mol / p->rho_cp_j_per_l_k;
    double c3  = p->delta_h3_j_per_mol / p->rho_cp_j_per_l_k;
    double hr  = p->ua_j_per_min_k / (p->volume_liters * p->rho_cp_j_per_l_k);
    double hj  = p->ua_j_per_min_k / (p->jacket_volume_liters * p->jacket_rho_cp_j_per_l_k);

    f[0] = s * (p->ca_feed_mol_per_l - ca) - r1 - r3;
    f[1] = s * (p->cb_feed_mol_per_l - cb) + r1 - r2;
    f[2] = s * (p->cc_feed_mol_per_l - cc) + r2;
    f[3] = s * (p->cd_feed_mol_per_l - cd) + r3;
    f[4] = s * (p->t_feed_kelvin - t) + (c1 * r1 + c2 * r2 + c3 * r3) - hr * (t - tj);
    f[5] = sj * (p->jacket_t_in_kelvin - tj) + hj * (t - tj);
}

void cstr_series_parallel_jacobian(
    const CstrSeriesParallelParams *p,
    double ca,
    double cb,
    double temp,
    double out_j[6][6]
) {
    if (!p || !out_j) return;

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            out_j[i][j] = 0.0;
        }
    }

    double ca_val = clamp_conc(ca);
    double cb_val = clamp_conc(cb);
    double t_val  = clamp_temp(temp);

    double k1 = p->k0_1_per_min * exp(-p->ea1_over_r_kelvin / t_val);
    double k2 = p->k0_2_per_min * exp(-p->ea2_over_r_kelvin / t_val);
    double k3 = p->k0_3_per_min * exp(-p->ea3_over_r_kelvin / t_val);

    double inv_t2 = 1.0 / (t_val * t_val);
    double dr1_dt = ca_val * k1 * p->ea1_over_r_kelvin * inv_t2;
    double dr2_dt = cb_val * k2 * p->ea2_over_r_kelvin * inv_t2;
    double dr3_dt = ca_val * k3 * p->ea3_over_r_kelvin * inv_t2;

    double s  = p->volumetric_flow_l_per_min / p->volume_liters;
    double sj = p->jacket_flow_l_per_min / p->jacket_volume_liters;
    double c1 = p->delta_h1_j_per_mol / p->rho_cp_j_per_l_k;
    double c2 = p->delta_h2_j_per_mol / p->rho_cp_j_per_l_k;
    double c3 = p->delta_h3_j_per_mol / p->rho_cp_j_per_l_k;
    double hr = p->ua_j_per_min_k / (p->volume_liters * p->rho_cp_j_per_l_k);
    double hj = p->ua_j_per_min_k / (p->jacket_volume_liters * p->jacket_rho_cp_j_per_l_k);

    /* Row 0: dCa/dt */
    out_j[0][0] = -s - k1 - k3;
    out_j[0][4] = -(dr1_dt + dr3_dt);

    /* Row 1: dCb/dt */
    out_j[1][0] = k1;
    out_j[1][1] = -s - k2;
    out_j[1][4] = dr1_dt - dr2_dt;

    /* Row 2: dCc/dt */
    out_j[2][1] = k2;
    out_j[2][2] = -s;
    out_j[2][4] = dr2_dt;

    /* Row 3: dCd/dt */
    out_j[3][0] = k3;
    out_j[3][3] = -s;
    out_j[3][4] = dr3_dt;

    /* Row 4: dT/dt */
    out_j[4][0] = c1 * k1 + c3 * k3;
    out_j[4][1] = c2 * k2;
    out_j[4][4] = -s - hr + (c1 * dr1_dt + c2 * dr2_dt + c3 * dr3_dt);
    out_j[4][5] = hr;

    /* Row 5: dTj/dt */
    out_j[5][4] = hj;
    out_j[5][5] = -sj - hj;
}

static int lu_factor_6x6(double a[6][6], int piv[6]) {
    for (int i = 0; i < 6; i++) piv[i] = i;

    for (int k = 0; k < 6; k++) {
        int max_row = k;
        double max_val = fabs(a[k][k]);
        for (int i = k + 1; i < 6; i++) {
            double v = fabs(a[i][k]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }

        if (max_val < 1e-15 || !isfinite(max_val)) {
            return -1;
        }

        if (max_row != k) {
            for (int j = 0; j < 6; j++) {
                double tmp = a[k][j];
                a[k][j] = a[max_row][j];
                a[max_row][j] = tmp;
            }
            int tmp_p = piv[k];
            piv[k] = piv[max_row];
            piv[max_row] = tmp_p;
        }

        double inv_diag = 1.0 / a[k][k];
        for (int i = k + 1; i < 6; i++) {
            double mult = a[i][k] * inv_diag;
            a[i][k] = mult;
            for (int j = k + 1; j < 6; j++) {
                a[i][j] -= mult * a[k][j];
            }
        }
    }

    return 0;
}

static void lu_solve_6x6(double lu[6][6], const int piv[6], const double b[6], double x[6]) {
    for (int i = 0; i < 6; i++) {
        double sum = b[piv[i]];
        for (int j = 0; j < i; j++) {
            sum -= lu[i][j] * x[j];
        }
        x[i] = sum;
    }

    for (int i = 5; i >= 0; i--) {
        double sum = x[i];
        for (int j = i + 1; j < 6; j++) {
            sum -= lu[i][j] * x[j];
        }
        x[i] = sum / lu[i][i];
    }
}

static int sp_ros3_single_step(
    const CstrSeriesParallelParams *p,
    const double x_in[6],
    double dt,
    double x_out[6],
    double err_out[6]
) {
    double jac[6][6];
    cstr_series_parallel_jacobian(p, x_in[0], x_in[1], x_in[4], jac);

    double inv_gh = 1.0 / (ROS3_GAMMA * dt);
    double w[6][6];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            w[i][j] = (i == j ? inv_gh : 0.0) - jac[i][j];
        }
    }

    int piv[6];
    if (lu_factor_6x6(w, piv) != 0) {
        return -1;
    }

    double f1[6], w1[6];
    eval_sp_rhs(p, x_in, f1);
    lu_solve_6x6(w, piv, f1, w1);

    double y2[6];
    for (int i = 0; i < 6; i++) {
        y2[i] = x_in[i] + ROS3_A21 * w1[i];
    }

    double f2[6], rhs2[6], w2[6];
    eval_sp_rhs(p, y2, f2);
    double inv_dt = 1.0 / dt;
    for (int i = 0; i < 6; i++) {
        rhs2[i] = f2[i] + (ROS3_C21 * inv_dt) * w1[i];
    }
    lu_solve_6x6(w, piv, rhs2, w2);

    /* Stage 3 shares y3 = y2 because A31 == A21 and A32 == 0 */
    double rhs3[6], w3[6];
    for (int i = 0; i < 6; i++) {
        rhs3[i] = f2[i] + ((ROS3_C31 * w1[i] + ROS3_C32 * w2[i]) * inv_dt);
    }
    lu_solve_6x6(w, piv, rhs3, w3);

    for (int i = 0; i < 6; i++) {
        double next_val = x_in[i] + ROS3_M1 * w1[i] + ROS3_M2 * w2[i] + ROS3_M3 * w3[i];
        if (!isfinite(next_val)) {
            return -1;
        }
        x_out[i] = next_val;
        if (err_out) {
            err_out[i] = ROS3_E1 * w1[i] + ROS3_E2 * w2[i] + ROS3_E3 * w3[i];
        }
    }

    return 0;
}

int cstr_series_parallel_step_ros3(
    const CstrSeriesParallelParams *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt
) {
    if (!params || !ca || !cb || !cc || !cd || !temp || !temp_j || dt <= 0.0 ||
        params->volume_liters <= 0.0 || params->jacket_volume_liters <= 0.0) {
        return -1;
    }

    double x[6] = { *ca, *cb, *cc, *cd, *temp, *temp_j };
    double x_next[6];
    if (sp_ros3_single_step(params, x, dt, x_next, NULL) != 0) {
        return -1;
    }

    *ca     = clamp_conc(x_next[0]);
    *cb     = clamp_conc(x_next[1]);
    *cc     = clamp_conc(x_next[2]);
    *cd     = clamp_conc(x_next[3]);
    *temp   = clamp_temp(x_next[4]);
    *temp_j = clamp_temp(x_next[5]);
    return 0;
}

int cstr_series_parallel_step_ros3_adaptive(
    const CstrSeriesParallelParams *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt,
    double rtol,
    double atol
) {
    if (!params || !ca || !cb || !cc || !cd || !temp || !temp_j || dt <= 0.0 ||
        params->volume_liters <= 0.0 || params->jacket_volume_liters <= 0.0) {
        return -1;
    }

    double rel_tol = rtol > 0.0 ? rtol : 1e-5;
    double abs_tol = atol > 0.0 ? atol : 1e-7;

    double x[6] = { *ca, *cb, *cc, *cd, *temp, *temp_j };
    double t = 0.0;
    double h = dt;
    const double h_min = 1e-8;
    const int max_substeps = 4096;

    for (int step = 0; step < max_substeps && t < dt - 1e-12; step++) {
        if (t + h > dt) {
            h = dt - t;
        }

        double x_cand[6], err_vec[6];
        if (sp_ros3_single_step(params, x, h, x_cand, err_vec) != 0) {
            h *= 0.25;
            if (h < h_min) return -1;
            continue;
        }

        double err_norm = 0.0;
        for (int i = 0; i < 6; i++) {
            double sc = abs_tol + rel_tol * fmax(fabs(x[i]), fabs(x_cand[i]));
            double ratio = fabs(err_vec[i]) / sc;
            err_norm += ratio * ratio;
        }
        err_norm = sqrt(err_norm / 6.0);

        if (err_norm <= 1.0 || h <= h_min * 1.1) {
            for (int i = 0; i < 6; i++) {
                x[i] = i < 4 ? clamp_conc(x_cand[i]) : clamp_temp(x_cand[i]);
            }
            t += h;

            double scale = err_norm > 1e-12 ? 0.9 * pow(err_norm, -1.0 / 3.0) : 4.0;
            if (scale < 0.2) scale = 0.2;
            if (scale > 4.0) scale = 4.0;
            h *= scale;
        } else {
            double scale = 0.9 * pow(err_norm, -1.0 / 3.0);
            if (scale < 0.1) scale = 0.1;
            if (scale > 0.5) scale = 0.5;
            h = fmax(h_min, h * scale);
        }
    }

    if (t < dt - 1e-9) {
        return -1;
    }

    *ca     = x[0];
    *cb     = x[1];
    *cc     = x[2];
    *cd     = x[3];
    *temp   = x[4];
    *temp_j = x[5];
    return 0;
}

int cstr_series_parallel_step_ros3_detailed(
    const CstrSeriesParallelParams *params,
    double *ca,
    double *cb,
    double *cc,
    double *cd,
    double *temp,
    double *temp_j,
    double dt,
    double rtol,
    double atol,
    double runaway_threshold_k,
    CstrSeriesParallelState *out_state
) {
    if (!out_state) return -1;

    int rc = cstr_series_parallel_step_ros3_adaptive(
        params, ca, cb, cc, cd, temp, temp_j, dt, rtol, atol);
    if (rc != 0) return -1;

    double r1, r2, r3;
    cstr_series_parallel_rates(params, *ca, *cb, *temp, &r1, &r2, &r3);

    out_state->ca_mol_per_l       = *ca;
    out_state->cb_mol_per_l       = *cb;
    out_state->cc_mol_per_l       = *cc;
    out_state->cd_mol_per_l       = *cd;
    out_state->temp_kelvin        = *temp;
    out_state->jacket_temp_kelvin = *temp_j;

    double ca0 = params->ca_feed_mol_per_l > 1e-12 ? params->ca_feed_mol_per_l : 1.0;
    out_state->conversion_a_pct = ((params->ca_feed_mol_per_l - *ca) / ca0) * 100.0;
    out_state->yield_b_pct      = ((*cb - params->cb_feed_mol_per_l) / ca0) * 100.0;

    double total_products = *cb + *cc + *cd;
    out_state->selectivity_b_pct = total_products > 1e-12 ? (*cb / total_products) * 100.0 : 100.0;

    out_state->rate_r1_mol_per_l_min = r1;
    out_state->rate_r2_mol_per_l_min = r2;
    out_state->rate_r3_mol_per_l_min = r3;

    double heat_gen_min = params->volume_liters *
        (params->delta_h1_j_per_mol * r1 + params->delta_h2_j_per_mol * r2 + params->delta_h3_j_per_mol * r3);
    out_state->heat_generated_watts = heat_gen_min / 60.0;

    double heat_rem_min = params->ua_j_per_min_k * (*temp - *temp_j);
    out_state->heat_removed_watts   = heat_rem_min / 60.0;

    out_state->is_thermal_runaway = (*temp >= runaway_threshold_k);
    return 0;
}

static void solve_2x2(double w[2][2], const double b[2], double x[2], int *err) {
    double det = w[0][0] * w[1][1] - w[0][1] * w[1][0];
    if (fabs(det) < 1e-15 || !isfinite(det)) {
        *err = -1;
        return;
    }
    double inv = 1.0 / det;
    x[0] = (w[1][1] * b[0] - w[0][1] * b[1]) * inv;
    x[1] = (-w[1][0] * b[0] + w[0][0] * b[1]) * inv;
    *err = 0;
}

static void eval_cstr2_rhs(const CstrParameters *p, double ca, double t, double f[2]) {
    double ca_v = clamp_conc(ca);
    double t_v  = clamp_temp(t);
    double r    = p->k0_per_min * exp(-p->ea_over_r_kelvin / t_v) * ca_v;
    double s    = p->volumetric_flow_l_per_min / p->volume_liters;
    double cg   = p->delta_h_j_per_mol / p->rho_cp_j_per_l_k;
    double cr   = p->ua_j_per_min_k / (p->volume_liters * p->rho_cp_j_per_l_k);
    f[0] = s * (p->ca_feed_mol_per_l - ca_v) - r;
    f[1] = s * (p->t_feed_kelvin - t_v) + cg * r - cr * (t_v - p->t_coolant_kelvin);
}

int cstr_step_ros3(
    const CstrParameters *p,
    double *ca,
    double *temp_k,
    double dt
) {
    if (!p || !ca || !temp_k || dt <= 0.0 || p->volume_liters <= 0.0) {
        return -1;
    }

    double ca_v = clamp_conc(*ca);
    double t_v  = clamp_temp(*temp_k);
    double k    = p->k0_per_min * exp(-p->ea_over_r_kelvin / t_v);
    double dr_dt = ca_v * k * p->ea_over_r_kelvin / (t_v * t_v);
    double s    = p->volumetric_flow_l_per_min / p->volume_liters;
    double cg   = p->delta_h_j_per_mol / p->rho_cp_j_per_l_k;
    double cr   = p->ua_j_per_min_k / (p->volume_liters * p->rho_cp_j_per_l_k);

    double inv_gh = 1.0 / (ROS3_GAMMA * dt);
    double w[2][2] = {
        { inv_gh - (-s - k),         dr_dt },
        { -(cg * k),                 inv_gh - (-s - cr + cg * dr_dt) }
    };

    double f1[2], w1[2];
    eval_cstr2_rhs(p, ca_v, t_v, f1);
    int err = 0;
    solve_2x2(w, f1, w1, &err);
    if (err != 0) return -1;

    double y2_ca = ca_v + ROS3_A21 * w1[0];
    double y2_t  = t_v  + ROS3_A21 * w1[1];
    double f2[2], rhs2[2], w2[2], rhs3[2], w3[2];
    eval_cstr2_rhs(p, y2_ca, y2_t, f2);

    double inv_dt = 1.0 / dt;
    rhs2[0] = f2[0] + (ROS3_C21 * inv_dt) * w1[0];
    rhs2[1] = f2[1] + (ROS3_C21 * inv_dt) * w1[1];
    solve_2x2(w, rhs2, w2, &err);
    if (err != 0) return -1;

    rhs3[0] = f2[0] + (ROS3_C31 * w1[0] + ROS3_C32 * w2[0]) * inv_dt;
    rhs3[1] = f2[1] + (ROS3_C31 * w1[1] + ROS3_C32 * w2[1]) * inv_dt;
    solve_2x2(w, rhs3, w3, &err);
    if (err != 0) return -1;

    double ca_n = ca_v + ROS3_M1 * w1[0] + ROS3_M2 * w2[0] + ROS3_M3 * w3[0];
    double t_n  = t_v  + ROS3_M1 * w1[1] + ROS3_M2 * w2[1] + ROS3_M3 * w3[1];
    if (!isfinite(ca_n) || !isfinite(t_n)) return -1;

    *ca     = clamp_conc(ca_n);
    *temp_k = clamp_temp(t_n);
    return 0;
}
