/**
 * @file closed_loop_driver.cpp
 * @brief Stateful interactive controller driver for closed-loop simulation.
 *
 * Protocol
 * ────────
 * Each stdin line:   q  omega  dt  [setpoint]
 * Each stdout line:  u_total,u_pid,u_p,u_i,u_d,u_stribeck,u_nn,integral
 *
 * Special commands (single-word lines):
 *   DISABLE_NN  — zero u_nn every step from this point forward
 *   RESET       — reset PID state (integrator + derivative)
 *   QUIT        — exit cleanly
 *
 * stdout is flushed after every line so Python readline() does not stall.
 *
 * Build (from firmware/):
 *   g++ -std=c++17 -O2 -I lib/PIRL -I lib/ArduinoStub \
 *       test/closed_loop/closed_loop_driver.cpp \
 *       lib/PIRL/pid.cpp lib/PIRL/stribeck.cpp lib/PIRL/pirl_inference.cpp \
 *       -o closed_loop_binary -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "pid.h"
#include "stribeck.h"
#include "pirl_inference.h"

// ============================================================================
//  Configuration — must match simulate_closed_loop.py
// ============================================================================
static constexpr float KP         = 10.0f;
static constexpr float KI         =  5.0f;
static constexpr float KD         =  0.1f;
static constexpr float V_MAX      =  12.0f;
static constexpr float V_MIN      = -12.0f;
static constexpr float SP_DEFAULT =  1.0f;

static const StribeckParams STRIBECK(0.15f, 0.35f, 0.10f, 2.0f, 0.01f);

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ============================================================================
//  Main
// ============================================================================
int main()
{
    PIDController pid(KP, KI, KD, V_MIN, V_MAX, SP_DEFAULT);
    bool nn_enabled = true;

    // Header — Python reads and discards this line
    printf("u_total,u_pid,u_p,u_i,u_d,u_stribeck,u_nn,integral\n");
    fflush(stdout);

    char  line[256];
    float q, omega, dt, setpoint;

    while (fgets(line, sizeof(line), stdin) != nullptr) {

        // Strip trailing newline / carriage-return
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        if (len == 0) continue;

        // ── Special commands ──────────────────────────────────────────
        if (strcmp(line, "DISABLE_NN") == 0) {
            nn_enabled = false;
            fprintf(stderr, "[closed_loop] NN disabled -- u_nn=0 every step\n");
            printf("DISABLE_NN_OK\n");
            fflush(stdout);
            continue;
        }
        if (strcmp(line, "RESET") == 0) {
            pid.reset();
            printf("RESET_OK\n");
            fflush(stdout);
            continue;
        }
        if (strcmp(line, "QUIT") == 0) {
            break;
        }

        // ── Parse state line ──────────────────────────────────────────
        int n = sscanf(line, "%f %f %f %f", &q, &omega, &dt, &setpoint);
        if (n == 3) {
            setpoint = SP_DEFAULT;
        } else if (n != 4) {
            fprintf(stderr, "[closed_loop] Parse error: '%s'\n", line);
            printf("ERROR\n");
            fflush(stdout);
            continue;
        }

        if (dt <= 0.0f) {
            fprintf(stderr, "[closed_loop] dt<=0, skipping\n");
            printf("ERROR\n");
            fflush(stdout);
            continue;
        }

        pid.setSetpoint(setpoint);

        const float u_stribeck = stribeck(omega, STRIBECK);
        const float u_ff       = u_stribeck;
        const float u_nn       = nn_enabled ? pirl_infer(omega) : 0.0f;
        const float u_pid      = pid.compute(q, dt, u_nn, u_ff);
        const float u_total    = clampf(u_pid + u_nn + u_ff, V_MIN, V_MAX);

        printf("%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
               u_total,
               u_pid,
               pid.getProportional(),
               pid.getIntegralTerm(),
               pid.getDerivativeTerm(),
               u_stribeck,
               u_nn,
               pid.getIntegral());
        fflush(stdout);
    }

    return 0;
}
