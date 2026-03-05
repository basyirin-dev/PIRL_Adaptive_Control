/**
 * @file cross_val_driver.cpp
 * @brief Cross-validation driver for Python vs C++ numerical comparison.
 *
 * Reads (q, dq, dt, [setpoint]) lines from stdin.
 * Writes CSV rows with full control term decomposition to stdout.
 *
 * Flags:
 *   --disable-nn   Zero u_nn every step (use when Python cannot run the
 *                  real network so both sides produce identical u_nn = 0).
 *
 * Build (from firmware/):
 *   g++ -std=c++17 -O2 -I lib/PIRL -I lib/ArduinoStub \
 *       test/cross_validation/cross_val_driver.cpp \
 *       lib/PIRL/pid.cpp lib/PIRL/stribeck.cpp lib/PIRL/pirl_inference.cpp \
 *       -o cross_val_binary -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pid.h"
#include "stribeck.h"
#include "pirl_inference.h"

// ============================================================================
//  Configuration — must match validate_firmware.py exactly
// ============================================================================
static constexpr float KP              = 10.0f;
static constexpr float KI              =  5.0f;
static constexpr float KD              =  0.1f;
static constexpr float V_MAX           =  OUTPUT_MAX;
static constexpr float V_MIN           =  OUTPUT_MIN;
static constexpr float SETPOINT_DEFAULT = 1.0f;

static const StribeckParams STRIBECK(0.15f, 0.35f, 0.10f, 2.0f, 0.01f);

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static bool parse_line(const char* line,
                        float& q, float& dq, float& dt, float& setpoint)
{
    int n = sscanf(line, "%f %f %f %f", &q, &dq, &dt, &setpoint);
    if (n == 3) { setpoint = SETPOINT_DEFAULT; return true; }
    return (n == 4);
}

// ============================================================================
//  Main
// ============================================================================
int main(int argc, char* argv[])
{
    // ------------------------------------------------------------------
    //  --disable-nn flag: zeroes u_nn every step.
    //  Use this when pirl_model is not importable in Python so both
    //  sides produce u_nn = 0 and the comparison is meaningful.
    // ------------------------------------------------------------------
    bool nn_enabled = true;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--disable-nn") == 0) {
            nn_enabled = false;
            fprintf(stderr, "[cross_val] NN disabled -- u_nn=0 every step\n");
        }
    }

    PIDController pid(KP, KI, KD, V_MIN, V_MAX, SETPOINT_DEFAULT);

    printf("step,q,dq,dt,setpoint,"
           "u_pid,u_p,u_i,u_d,"
           "u_stribeck,u_nn,u_ff,"
           "u_total,integral,deriv_filtered\n");

    char  line[512];
    int   step     = 0;
    float q        = 0.0f;
    float dq       = 0.0f;
    float dt       = 0.0f;
    float setpoint = SETPOINT_DEFAULT;

    while (fgets(line, sizeof(line), stdin) != nullptr) {

        if (line[0] == '\n' || line[0] == '#' || line[0] == '\r') continue;

        if (!parse_line(line, q, dq, dt, setpoint)) {
            fprintf(stderr, "[cross_val] Parse error at step %d: %s\n",
                    step, line);
            continue;
        }
        if (dt <= 0.0f) { continue; }

        pid.setSetpoint(setpoint);

        const float u_stribeck = stribeck(dq, STRIBECK);
        const float u_ff       = u_stribeck;
        const float u_nn       = nn_enabled ? pirl_infer(dq) : 0.0f;
        const float u_pid      = pid.compute(q, dt, u_nn, u_ff);
        const float u_total    = clampf(u_pid + u_nn + u_ff, V_MIN, V_MAX);

        printf("%d,%.8f,%.8f,%.8f,%.8f,"
               "%.8f,%.8f,%.8f,%.8f,"
               "%.8f,%.8f,%.8f,"
               "%.8f,%.8f,%.8f\n",
               step, q, dq, dt, setpoint,
               u_pid,
               pid.getProportional(),
               pid.getIntegralTerm(),
               pid.getDerivativeTerm(),
               u_stribeck,
               u_nn,
               u_ff,
               u_total,
               pid.getIntegral(),
               pid.getDerivative());

        ++step;
    }

    return 0;
}
