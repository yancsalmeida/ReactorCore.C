#include "vector.h"
#include "rk4.h"
#include "cstr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    printf("====================================================================\n");
    printf("   ReactorCore.C — Exothermic CSTR Dynamic Simulation Engine        \n");
    printf("   C11 Systems Architecture · RK4 Numerical Integrator · Arrhenius  \n");
    printf("====================================================================\n\n");

    const char *csv_filename = "cstr_simulation.csv";
    if (argc > 1) {
        csv_filename = argv[1];
    }

    FILE *fp = fopen(csv_filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open output file %s\n", csv_filename);
        return EXIT_FAILURE;
    }
    fprintf(fp, "time_min,Ca_mol_per_L,temp_K,conversion,reaction_rate_mol_per_L_min,status\n");

    CstrParameters params = cstr_default_parameters();
    Rk4Workspace *ws = rk4_workspace_create(2);
    if (!ws) {
        fprintf(stderr, "Error allocating RK4 workspace\n");
        fclose(fp);
        return EXIT_FAILURE;
    }

    Vector *y = vector_create(2);
    Vector *y_next = vector_create(2);

    /* Initial state at t = 0 */
    y->data[0] = params.ca_feed_mol_per_l; /* Ca = 1.0 mol/L */
    y->data[1] = params.t_feed_kelvin;      /* T = 350.0 K */

    const double dt = 0.005;        /* 0.005 min = 0.3 seconds */
    const double t_total = 15.0;    /* 15 minutes total simulation */
    const double t_disrupt = 10.0;  /* Coolant disruption occurs at t = 10 min */
    const double runaway_threshold = 440.0;

    printf("[INFO] Simulating 0.0 to %.1f minutes with dt = %.4f min...\n", t_total, dt);
    printf("[INFO] Planned disruption: Coolant temperature increases at t = %.1f min (Cooling reduction)\n\n", t_disrupt);

    printf("%-8s | %-12s | %-10s | %-10s | %-14s | %-10s\n",
           "Time(min)", "Ca (mol/L)", "Temp (K)", "Conv (%)", "Rate (mol/L/m)", "Status");
    printf("----------------------------------------------------------------------------\n");

    double t = 0.0;
    size_t print_interval = (size_t)(0.5 / dt); /* Print to stdout every 0.5 min */
    size_t step = 0;

    while (t <= t_total) {
        /* Inject disruption at t = 10.0 min: coolant jacket water warms up from 300K to 325K */
        if (t >= t_disrupt && params.t_coolant_kelvin < 325.0) {
            params.t_coolant_kelvin = 325.0;
            printf(">> [EVENT @ t = %.2f min] Coolant jacket temperature increased to 325.0 K!\n", t);
        }

        double Ca = y->data[0];
        double T = y->data[1];
        double conversion = (1.0 - (Ca / params.ca_feed_mol_per_l)) * 100.0;
        double r = cstr_reaction_rate(&params, Ca, T);

        const char *status = "NORMAL";
        if (T >= runaway_threshold) {
            status = "RUNAWAY";
        } else if (T >= 390.0) {
            status = "WARNING";
        }

        /* Write to CSV */
        fprintf(fp, "%.4f,%.6f,%.2f,%.2f,%.6f,%s\n", t, Ca, T, conversion, r, status);

        /* Print periodic status row */
        if (step % print_interval == 0) {
            printf("%-8.2f | %-12.4f | %-10.2f | %-10.1f | %-14.4f | %-10s\n",
                   t, Ca, T, conversion, r, status);
        }

        rk4_step(ws, cstr_derivatives, t, y, y_next, dt, &params);
        vector_copy(y, y_next);
        t += dt;
        step++;
    }

    fclose(fp);
    printf("----------------------------------------------------------------------------\n");
    printf("[SUCCESS] Simulation completed. Telemetry exported to: %s\n", csv_filename);

    vector_free(y);
    vector_free(y_next);
    rk4_workspace_free(ws);

    return EXIT_SUCCESS;
}
