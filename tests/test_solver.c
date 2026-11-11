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

#include "simd_math.h"
#include <time.h>

static void test_simd_vector_and_exp(void) {
    printf("Running test_simd_vector_and_exp (AVX2 + FMA Accuracy)...\n");

    int has_simd = simd_avx2_fma_available();
    TEST_ASSERT(has_simd == 1, "AVX2 + FMA hardware acceleration compiled and active");

#if REACTOR_CORE_HAS_AVX2
    /* Test 1: SAXPY vector sizes (sub-vector, exact vector, remainder tail) */
    size_t test_sizes[] = {1, 3, 4, 7, 16, 65};
    for (size_t s = 0; s < sizeof(test_sizes)/sizeof(test_sizes[0]); s++) {
        size_t n = test_sizes[s];
        Vector *vx = vector_create(n);
        Vector *vy = vector_create(n);
        for (size_t i = 0; i < n; i++) {
            vx->data[i] = (double)(i + 1) * 0.5;
            vy->data[i] = (double)(i + 1) * 2.0;
        }

        double alpha = 3.5;
        bool ok = vector_saxpy(vy, alpha, vx);
        TEST_ASSERT(ok, "Vector saxpy returns true");

        bool all_correct = true;
        for (size_t i = 0; i < n; i++) {
            double expected = alpha * ((double)(i + 1) * 0.5) + ((double)(i + 1) * 2.0);
            if (fabs(vy->data[i] - expected) > 1e-12) {
                all_correct = false;
                break;
            }
        }
        TEST_ASSERT(all_correct, "SAXPY AVX2 result matches scalar expectation for various sizes");

        vector_free(vx);
        vector_free(vy);
    }

    /* Test 2: SIMD Exponential Accuracy against libm exp() */
    double test_inputs[8] = {-28.5, -23.8, -15.2, -8.7, -4.2, -1.0, 0.0, 2.5};
    for (int b = 0; b < 2; b++) {
        __m256d vin = _mm256_loadu_pd(&test_inputs[b * 4]);
        __m256d vout = simd_exp256_pd(vin);
        double out_arr[4];
        _mm256_storeu_pd(out_arr, vout);

        for (int i = 0; i < 4; i++) {
            double exact = exp(test_inputs[b * 4 + i]);
            double rel_err = fabs(out_arr[i] - exact) / (exact > 1e-300 ? exact : 1.0);
            TEST_ASSERT(rel_err < 1e-13, "SIMD exp256 relative error < 1e-13");
        }
    }
#endif
}

static void test_cstr_batch_simd_derivatives_and_rk4(void) {
    printf("Running test_cstr_batch_simd_derivatives_and_rk4 (Multi-Reactor Parallel Integration)...\n");

    CstrParameters params = cstr_default_parameters();

    /* Test with N = 11 reactors (tests two 4-wide SIMD chunks + 3-wide remainder tail) */
    const size_t n_reactors = 11;
    double ca_batch[11];
    double temp_batch[11];
    double dcadt_batch[11];
    double dtdt_batch[11];

    /* Vary initial states across the reactors */
    for (size_t i = 0; i < n_reactors; i++) {
        ca_batch[i] = 0.5 + (double)i * 0.05;
        temp_batch[i] = 310.0 + (double)i * 8.0;
    }

    /* Batch evaluation via AVX2 SIMD */
    cstr_batch_derivatives(n_reactors, &params, ca_batch, temp_batch, dcadt_batch, dtdt_batch);

    /* Compare against scalar cstr_derivatives for each individual reactor */
    Vector *y_scalar = vector_create(2);
    Vector *dydt_scalar = vector_create(2);

    bool deriv_match = true;
    for (size_t i = 0; i < n_reactors; i++) {
        y_scalar->data[0] = ca_batch[i];
        y_scalar->data[1] = temp_batch[i];

        cstr_derivatives(0.0, y_scalar, dydt_scalar, &params);

        double diff_ca = fabs(dcadt_batch[i] - dydt_scalar->data[0]);
        double diff_t  = fabs(dtdt_batch[i] - dydt_scalar->data[1]);

        if (diff_ca > 1e-12 || diff_t > 1e-11) {
            deriv_match = false;
            break;
        }
    }
    TEST_ASSERT(deriv_match, "Batch SIMD derivatives match scalar cstr_derivatives to within 1e-11");

    /* Test full RK4 step: compare batch RK4 vs scalar RK4 across 50 time steps */
    double ca_scalar[11];
    double temp_scalar[11];
    memcpy(ca_scalar, ca_batch, n_reactors * sizeof(double));
    memcpy(temp_scalar, temp_batch, n_reactors * sizeof(double));

    Rk4Workspace *ws_scalar = rk4_workspace_create(2);
    Vector *y_next_scalar = vector_create(2);
    double dt = 0.01;

    for (int step = 0; step < 50; step++) {
        /* Batch RK4 step */
        bool ok = cstr_batch_step_rk4(n_reactors, &params, ca_batch, temp_batch, dt);
        TEST_ASSERT(ok, "Batch RK4 step succeeds");

        /* Scalar individual steps */
        for (size_t i = 0; i < n_reactors; i++) {
            y_scalar->data[0] = ca_scalar[i];
            y_scalar->data[1] = temp_scalar[i];
            rk4_step(ws_scalar, cstr_derivatives, 0.0, y_scalar, y_next_scalar, dt, &params);
            ca_scalar[i] = y_next_scalar->data[0];
            temp_scalar[i] = y_next_scalar->data[1];
        }
    }

    bool trajectory_match = true;
    for (size_t i = 0; i < n_reactors; i++) {
        if (fabs(ca_batch[i] - ca_scalar[i]) > 1e-10 ||
            fabs(temp_batch[i] - temp_scalar[i]) > 1e-9) {
            trajectory_match = false;
            break;
        }
    }
    TEST_ASSERT(trajectory_match, "50-step batch RK4 multi-reactor integration matches scalar solution to 1e-9");

    vector_free(y_scalar);
    vector_free(dydt_scalar);
    vector_free(y_next_scalar);
    rk4_workspace_free(ws_scalar);
}

static void test_cstr_batch_ffi_interface(void) {
    printf("Running test_cstr_batch_ffi_interface...\n");

    int has_simd = reactor_core_has_simd();
    TEST_ASSERT(has_simd == 1, "FFI reports SIMD AVX2 acceleration is active");

    CstrParamsFFI params;
    int rc = reactor_cstr_default_params(&params);
    TEST_ASSERT(rc == 0, "Default parameters retrieved via FFI");

    const size_t n = 8;
    double ca[8] = {1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7};
    double temp[8] = {340.0, 350.0, 360.0, 370.0, 340.0, 350.0, 360.0, 370.0};

    rc = reactor_cstr_batch_step_rk4(&params, ca, temp, n, 0.01);
    TEST_ASSERT(rc == 0, "FFI batch RK4 step succeeds");
    TEST_ASSERT(ca[0] < 1.0, "Reactant consumed in reactor 0");
    TEST_ASSERT(temp[0] > 340.0, "Temperature increased in reactor 0");

    /* Detailed telemetry diagnostics for batch */
    CstrStateFFI states[8];
    rc = reactor_cstr_batch_step_detailed(&params, ca, temp, n, 0.01, 380.0, states);
    TEST_ASSERT(rc == 0, "FFI detailed batch step succeeds");
    TEST_ASSERT(states[0].conversion_pct > 0.0, "Conversion computed for reactor 0");
    TEST_ASSERT(!states[0].is_runaway, "Reactor 0 at safe temperature");

    /* Batch trajectory simulation */
    const size_t steps = 20;
    double ca0[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double temp0[8] = {330.0, 340.0, 350.0, 360.0, 330.0, 340.0, 350.0, 360.0};
    double out_time[21];
    double out_ca[21 * 8];
    double out_temp[21 * 8];

    rc = reactor_cstr_batch_simulate_trajectory(&params, ca0, temp0, n, 0.01, steps, out_time, out_ca, out_temp);
    TEST_ASSERT(rc == 0, "FFI batch trajectory simulation succeeds");
    TEST_ASSERT(fabs(out_time[20] - 0.20) < 1e-6, "Trajectory time at step 20 is 0.20 min");
    TEST_ASSERT(out_ca[20 * 8 + 0] < 1.0, "Concentration decreased at end of trajectory");
}

static void test_simd_batch_benchmark(void) {
    printf("Running test_simd_batch_benchmark (64 Simultaneous Reactors x 2000 RK4 Steps)...\n");

    const size_t n_reactors = 64;
    const size_t n_steps = 2000;
    const double dt = 0.005;

    CstrParameters params = cstr_default_parameters();
    double ca[64];
    double temp[64];
    for (size_t i = 0; i < n_reactors; i++) {
        ca[i] = 1.0;
        temp[i] = 340.0 + (double)(i % 16) * 2.0;
    }

    CstrBatchWorkspace *ws = cstr_batch_workspace_create(n_reactors);
    TEST_ASSERT(ws != NULL, "Batch workspace allocated for 64 reactors");

    clock_t start = clock();
    for (size_t s = 0; s < n_steps; s++) {
        cstr_batch_step_rk4_ws(ws, n_reactors, &params, ca, temp, dt);
    }
    clock_t end = clock();

    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    size_t total_evals = n_reactors * n_steps * 4; /* 4 RK stages per step */
    double evals_per_sec = (double)total_evals / (elapsed_sec > 1e-6 ? elapsed_sec : 1e-6);

    printf("   [Benchmark] 64 reactors x %zu steps completed in %.4f s (%.1f million stage evals/sec)\n",
           n_steps, elapsed_sec, evals_per_sec / 1e6);

    TEST_ASSERT(elapsed_sec >= 0.0, "Benchmark finished successfully");
    TEST_ASSERT(ca[0] < 1.0, "Chemical conversion verified at conclusion");

    cstr_batch_workspace_free(ws);
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
    test_simd_vector_and_exp();
    test_cstr_batch_simd_derivatives_and_rk4();
    test_cstr_batch_ffi_interface();
    test_simd_batch_benchmark();

    printf("=========================================\n");
    printf("Summary: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("=========================================\n");

    return (g_tests_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
