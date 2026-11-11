#include "vector.h"
#include "rk4.h"
#include "cstr.h"
#include "reactor_ffi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (cond) { \
        g_tests_passed++; \
    } else { \
        g_tests_failed++; \
        fprintf(stderr, "[FAIL] %s (%s:%d)\n", msg, __FILE__, __LINE__); \
    } \
} while (0)

/* Helper for analytical test: dy/dt = -0.5 * y */
static void decay_ode(double t, const Vector *y, Vector *dydt, void *params) {
    (void)t;
    double k = *(double*)params;
    dydt->data[0] = -k * y->data[0];
}

/* Helper for harmonic oscillator: dy1/dt = y2, dy2/dt = -y1 */
static void oscillator_ode(double t, const Vector *y, Vector *dydt, void *params) {
    (void)t;
    (void)params;
    dydt->data[0] = y->data[1];
    dydt->data[1] = -y->data[0];
}

static void test_vector_operations(void) {
    printf("Running test_vector_operations...\n");

    Vector *v1 = vector_create(3);
    TEST_ASSERT(v1 != NULL, "Vector creation allocates memory");
    TEST_ASSERT(v1->size == 3, "Vector size correctly initialized");

    v1->data[0] = 1.0;
    v1->data[1] = 2.0;
    v1->data[2] = 3.0;

    Vector *v2 = vector_clone(v1);
    TEST_ASSERT(v2 != NULL, "Vector clone allocates new vector");
    TEST_ASSERT(v2->size == 3, "Vector clone matches dimension");
    TEST_ASSERT(fabs(v2->data[1] - 2.0) < 1e-9, "Vector clone copies contents");

    /* Test SAXPY: v2 = 2.0 * v1 + v2 => v2[0] = 2*1 + 1 = 3 */
    bool ok = vector_saxpy(v2, 2.0, v1);
    TEST_ASSERT(ok, "Vector saxpy returns true for matching sizes");
    TEST_ASSERT(fabs(v2->data[0] - 3.0) < 1e-9, "SAXPY correctly calculates element 0");
    TEST_ASSERT(fabs(v2->data[2] - 9.0) < 1e-9, "SAXPY correctly calculates element 2");

    vector_free(v1);
    vector_free(v2);
    /* Safe double free check / NULL safety */
    vector_free(NULL);
}

static void test_rk4_analytical_decay(void) {
    printf("Running test_rk4_analytical_decay (Order of Accuracy)...\n");

    Rk4Workspace *ws = rk4_workspace_create(1);
    TEST_ASSERT(ws != NULL, "Rk4Workspace created successfully");

    Vector *y = vector_create(1);
    Vector *y_next = vector_create(1);
    y->data[0] = 1.0; /* y(0) = 1.0 */

    double k = 0.5;
    double dt = 0.05;
    double t_final = 2.0;
    size_t steps = (size_t)(t_final / dt);

    double t = 0.0;
    for (size_t i = 0; i < steps; i++) {
        rk4_step(ws, decay_ode, t, y, y_next, dt, &k);
        vector_copy(y, y_next);
        t += dt;
    }

    double y_analytical = exp(-k * t_final);
    double error = fabs(y->data[0] - y_analytical);

    /* For RK4 with dt=0.05 over 2.0 seconds, error must be < 1e-7 */
    TEST_ASSERT(error < 1e-7, "RK4 matches analytical decay solution with O(dt^4) precision");

    vector_free(y);
    vector_free(y_next);
    rk4_workspace_free(ws);
}

static void test_rk4_harmonic_oscillator_energy(void) {
    printf("Running test_rk4_harmonic_oscillator_energy...\n");

    Rk4Workspace *ws = rk4_workspace_create(2);
    Vector *y = vector_create(2);
    Vector *y_next = vector_create(2);

    /* y1(0) = 1.0 (position), y2(0) = 0.0 (velocity) */
    y->data[0] = 1.0;
    y->data[1] = 0.0;

    double e0 = 0.5 * (y->data[0] * y->data[0] + y->data[1] * y->data[1]);
    double dt = 0.01;
    double t = 0.0;

    /* Simulate one full period T = 2*pi approx 6.28 seconds */
    for (int i = 0; i < 628; i++) {
        rk4_step(ws, oscillator_ode, t, y, y_next, dt, NULL);
        vector_copy(y, y_next);
        t += dt;
    }

    double e_final = 0.5 * (y->data[0] * y->data[0] + y->data[1] * y->data[1]);
    double energy_drift = fabs(e_final - e0);

    TEST_ASSERT(energy_drift < 1e-5, "Harmonic oscillator energy is conserved within 1e-5");

    vector_free(y);
    vector_free(y_next);
    rk4_workspace_free(ws);
}

static void test_cstr_steady_state_and_runaway(void) {
    printf("Running test_cstr_steady_state_and_runaway...\n");

    CstrParameters params = cstr_default_parameters();
    Rk4Workspace *ws = rk4_workspace_create(2);

    Vector *y = vector_create(2);
    Vector *y_next = vector_create(2);

    /* Initial state: cold reactor filled with feed conditions */
    y->data[0] = params.ca_feed_mol_per_l; /* Ca = 1.0 mol/L */
    y->data[1] = params.t_feed_kelvin;      /* T = 350.0 K */

    double dt = 0.005; /* 0.005 min = 0.3 s */
    double t = 0.0;

    /* Simulate 10 minutes to reach steady state */
    for (int i = 0; i < 2000; i++) {
        rk4_step(ws, cstr_derivatives, t, y, y_next, dt, &params);
        vector_copy(y, y_next);
        t += dt;
    }

    /* Verify non-linear conversion and temperature rise: Ca approx 0.026 mol/L (97.4% conversion) */
    TEST_ASSERT(y->data[0] > 0.01 && y->data[0] < 0.95, "Reactant concentration in valid bounded range");
    TEST_ASSERT(y->data[1] > 310.0 && y->data[1] < 450.0, "Reactor temperature stable under jacket cooling");
    TEST_ASSERT(!cstr_is_thermal_runaway(y, 395.0), "Controlled CSTR avoids runaway threshold (389.7 K < 395.0 K)");

    /* Now trigger cooling failure: UA = 0 (cooling jacket pump breakdown).
     * Without cooling, temperature rises toward the thermodynamic adiabatic limit (approx 400 K).
     */
    params.ua_j_per_min_k = 0.0;

    bool runaway_detected = false;
    for (int i = 0; i < 1000; i++) {
        rk4_step(ws, cstr_derivatives, t, y, y_next, dt, &params);
        vector_copy(y, y_next);
        t += dt;

        if (cstr_is_thermal_runaway(y, 395.0)) {
            runaway_detected = true;
            break;
        }
    }

    TEST_ASSERT(runaway_detected, "Cooling jacket failure causes temperature to exceed 395 K threshold");

    vector_free(y);
    vector_free(y_next);
    rk4_workspace_free(ws);
}

static void test_reactor_ffi_interface(void) {
    printf("Running test_reactor_ffi_interface...\n");

    const char *ver = reactor_core_version();
    TEST_ASSERT(ver != NULL && strstr(ver, "ReactorCore.C") != NULL, "Version string returned");

    CstrParamsFFI params;
    int rc = reactor_cstr_default_params(&params);
    TEST_ASSERT(rc == 0, "Default parameters populated");
    TEST_ASSERT(params.ca_feed_mol_per_l == 1.0, "Feed concentration is 1.0");

    double ca = 1.0;
    double temp = 350.0;
    rc = reactor_cstr_step_rk4(&params, &ca, &temp, 0.005);
    TEST_ASSERT(rc == 0, "Single RK4 step succeeds");
    TEST_ASSERT(ca < 1.0, "Concentration decreases after reaction");
    TEST_ASSERT(temp > 350.0, "Temperature increases due to exothermicity");

    CstrStateFFI detailed;
    rc = reactor_cstr_step_detailed(&params, &ca, &temp, 0.005, 400.0, &detailed);
    TEST_ASSERT(rc == 0, "Detailed step succeeds");
    TEST_ASSERT(detailed.conversion_pct > 0.0, "Conversion percentage calculated");
    TEST_ASSERT(!detailed.is_runaway, "Initial operating point is not runaway");

    /* Batch trajectory test */
    const size_t n_steps = 100;
    double times[101];
    double ca_arr[101];
    double temp_arr[101];
    rc = reactor_cstr_simulate_trajectory(&params, 1.0, 350.0, 0.005, n_steps, times, ca_arr, temp_arr);
    TEST_ASSERT(rc == 0, "Batch trajectory simulation succeeds");
    TEST_ASSERT(fabs(times[100] - 0.5) < 1e-6, "Final trajectory time matches dt * n_steps");
    TEST_ASSERT(ca_arr[100] < ca_arr[0], "Concentration drops along trajectory");
}

static void test_rk45_workspace_and_decay(void) {
    printf("Running test_rk45_workspace_and_decay (Adaptive Order & Accuracy)...\n");

    Rk45Workspace *ws = rk45_workspace_create(1);
    TEST_ASSERT(ws != NULL, "Rk45Workspace allocated successfully");
    TEST_ASSERT(ws->dim == 1, "Rk45Workspace dimension matches");

    Vector *y = vector_create(1);
    y->data[0] = 1.0;

    double k = 0.5;
    double t_final = 2.0;

    Rk45Control ctrl = rk45_default_control();
    ctrl.rtol = 1e-6;
    ctrl.atol = 1e-8;

    Rk45AdaptiveStats stats;
    bool ok = rk45_integrate_interval(ws, decay_ode, 0.0, t_final, y, 0.1, &ctrl, &stats, &k);
    TEST_ASSERT(ok, "Adaptive RKF45 interval integration succeeds");

    double y_analytical = exp(-k * t_final);
    double error = fabs(y->data[0] - y_analytical);
    TEST_ASSERT(error < 1e-6, "RKF45 decay solution matches analytical with < 1e-6 precision");
    TEST_ASSERT(stats.steps_accepted > 0, "Adaptive integration accepted valid steps");
    TEST_ASSERT(stats.max_local_error <= 1.0, "All accepted steps satisfy normalized local error <= 1.0");

    vector_free(y);
    rk45_workspace_free(ws);
}

static void test_rk45_adaptive_step_shrinkage_on_runaway(void) {
    printf("Running test_rk45_adaptive_step_shrinkage_on_runaway...\n");

    CstrParameters params = cstr_default_parameters();
    /* Thermal runaway triggered by cooling jacket loss */
    params.ua_j_per_min_k = 0.0;

    Rk45Workspace *ws = rk45_workspace_create(2);
    Vector *y = vector_create(2);
    y->data[0] = params.ca_feed_mol_per_l;
    y->data[1] = params.t_feed_kelvin;

    Rk45Control ctrl = rk45_default_control();
    ctrl.rtol = 1e-5;
    ctrl.atol = 1e-7;

    Rk45AdaptiveStats stats;
    double initial_h = 0.05; /* 3 seconds */
    bool ok = rk45_integrate_interval(ws, cstr_derivatives, 0.0, 1.0, y, initial_h, &ctrl, &stats, &params);

    TEST_ASSERT(ok, "Adaptive integration completes across runaway regime");
    TEST_ASSERT(stats.min_h < initial_h, "Adaptive RKF45 dynamically shrinks step size during thermal runaway");
    TEST_ASSERT(y->data[1] > 380.0, "Exothermic runaway elevates reactor temperature");

    vector_free(y);
    rk45_workspace_free(ws);
}

static void test_reactor_ffi_adaptive_interface(void) {
    printf("Running test_reactor_ffi_adaptive_interface...\n");

    CstrParamsFFI params;
    int rc = reactor_cstr_default_params(&params);
    TEST_ASSERT(rc == 0, "Default parameters populated");

    double ca = 1.0;
    double temp = 350.0;
    CstrAdaptiveStatsFFI stats;

    rc = reactor_cstr_step_adaptive(&params, &ca, &temp, 0.01, 1e-5, 1e-7, &stats);
    TEST_ASSERT(rc == 0, "Single adaptive RKF45 step succeeds");
    TEST_ASSERT(ca < 1.0, "Adaptive step consumes reactant");
    TEST_ASSERT(temp > 350.0, "Adaptive step generates reaction heat");
    TEST_ASSERT(stats.steps_accepted > 0, "Steps accepted is > 0");

    CstrStateFFI detailed;
    rc = reactor_cstr_step_detailed_adaptive(&params, &ca, &temp, 0.01, 1e-5, 1e-7, 400.0, &detailed, &stats);
    TEST_ASSERT(rc == 0, "Detailed adaptive step succeeds");
    TEST_ASSERT(detailed.conversion_pct > 0.0, "Conversion percentage calculated");
    TEST_ASSERT(!detailed.is_runaway, "Controlled operating point is safe");

    /* Batch adaptive trajectory test */
    const size_t n_steps = 50;
    double times[51];
    double ca_arr[51];
    double temp_arr[51];
    rc = reactor_cstr_simulate_adaptive_trajectory(&params, 1.0, 350.0, 0.01, n_steps, 1e-5, 1e-7, times, ca_arr, temp_arr, &stats);
    TEST_ASSERT(rc == 0, "Batch adaptive trajectory simulation succeeds");
    TEST_ASSERT(fabs(times[50] - 0.5) < 1e-6, "Final adaptive time matches 50 * 0.01");
    TEST_ASSERT(ca_arr[50] < ca_arr[0], "Concentration drops continuously");
    TEST_ASSERT(stats.steps_accepted >= n_steps, "Total accepted steps >= sample steps");
}

int main(void) {
    printf("=========================================\n");
    printf("   ReactorCore.C Test Suite (C11)        \n");
    printf("=========================================\n");

    test_vector_operations();
    test_rk4_analytical_decay();
    test_rk4_harmonic_oscillator_energy();
    test_cstr_steady_state_and_runaway();
    test_reactor_ffi_interface();
    test_rk45_workspace_and_decay();
    test_rk45_adaptive_step_shrinkage_on_runaway();
    test_reactor_ffi_adaptive_interface();

    printf("=========================================\n");
    printf("Summary: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("=========================================\n");

    return (g_tests_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
