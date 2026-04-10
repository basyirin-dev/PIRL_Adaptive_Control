/**
 * @file main.cpp
 * @brief PIRL Adaptive Friction Compensation — Top-level Control Loop.
 *
 * Architecture:
 *   u_total = u_pid  (PIDController, NN-aware anti-windup)
 *           + u_ff   (inertia feed-forward)
 *           + u_nn   (PIRL residual friction compensation)
 *
 * Timing:
 *   IntervalTimer fires controlISR() at exactly CONTROL_PERIOD_US.
 *   The ISR sets a flag; all computation runs in loop() — ISR is kept
 *   minimal to avoid nesting issues with Serial.printf().
 *   DO NOT use delay() or millis()-polling — jitter degrades derivative
 *   estimation and corrupts the dt passed to PIDController::compute().
 *
 * Logging:
 *   CSV over Serial at 2 Mbaud.  Column order:
 *     t_ms, q, dq, u_pid, u_ff, u_nn, u_total, loop_us
 *   t_elapsed (loop_us) is the smoke-test latency measurement —
 *   values consistently > 900 µs at 1 kHz indicate a timing problem.
 *
 * Hardware stubs:
 *   readEncoder(), estimateVelocity(), readCurrentSensor(), setMotorVoltage()
 *   are declared here as stubs.  Replace with actual driver calls when
 *   hardware is available.  This file compiles cleanly without hardware.
 *
 * Target: Teensy 4.1 (IMXRT1062, Cortex-M7 @ 600 MHz)
 */

#include <Arduino.h>
#include "pid.h"
#include "stribeck.h"
#include "pirl_inference.h"   /* also pulls in pirl_weights.h + self-test */


// ============================================================================
//  Timing configuration
// ============================================================================

#define CONTROL_HZ          100                          ///< Control loop rate
#define CONTROL_PERIOD_US   (1000000UL / CONTROL_HZ)    ///< 10 000 µs = 10 ms
#define DT_S                (1.0f / CONTROL_HZ)         ///< dt in seconds (0.01)


// ============================================================================
//  Physical constants
// ============================================================================

/** Estimated rotor inertia [kg·m²] — replace with identified value. */
static constexpr float J_EST        = 0.0012f;

/** Velocity deadband: PIRL inference suppressed below this [rad/s]. */
static constexpr float VEL_DEADBAND = PIRL_DEADBAND_RAD_S;

/** Motor voltage limits [V] — must match OUTPUT_MAX / OUTPUT_MIN in pid.h */
static constexpr float V_MAX        = OUTPUT_MAX;
static constexpr float V_MIN        = OUTPUT_MIN;

/** Saturation clamp helper. */
inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}


// ============================================================================
//  Reference trajectory
//  Replace with trajectory generator or serial-commanded setpoints.
// ============================================================================

static float target_q   = 1.0f;    ///< Position setpoint [rad]
static float target_dq  = 0.0f;    ///< Velocity setpoint [rad/s]
static float target_ddq = 0.0f;    ///< Acceleration setpoint [rad/s²]


// ============================================================================
//  Controller instance
//  Gains from Phase 1 simulation (controllers.py defaults).
//  Tune via serial command in a future revision.
// ============================================================================

static PIDController pid(
    /* kp */ 10.0f,
    /* ki */  5.0f,
    /* kd */  0.1f,
    /* output_min */ V_MIN,
    /* output_max */ V_MAX,
    /* setpoint   */ target_q
);


// ============================================================================
//  ISR flag  (volatile: written in ISR, read in loop())
// ============================================================================

static IntervalTimer controlTimer;
static volatile bool controlFlag = false;

void controlISR() {
    controlFlag = true;
}


// ============================================================================
//  Hardware interface stubs
//  ─────────────────────────
//  These compile and return deterministic values so the firmware builds and
//  the control loop logic can be verified without physical hardware.
//  Replace the stub bodies with real encoder/ADC driver calls before flash.
// ============================================================================

/** @brief Read encoder position.  @return Joint angle [rad]. */
static float readEncoder() {
    // STUB: return a fixed position for compile-time verification.
    // Replace with: return encoder.getPosition() * ENCODER_TO_RAD;
    return 0.0f;
}

/** @brief Estimate joint velocity (e.g. from encoder delta or observer).
 *  @return Joint velocity [rad/s]. */
static float estimateVelocity() {
    // STUB: replace with velocity observer or filtered encoder delta.
    return 0.0f;
}

/** @brief Read phase current (q-axis).  @return i_q [A]. */
static float readCurrentSensor() {
    // STUB: replace with ADC read + current sense amplifier scaling.
    return 0.0f;
}

/**
 * @brief Write voltage command to H-bridge / motor driver.
 * @param v  Commanded voltage [V], clamped to [V_MIN, V_MAX].
 */
static void setMotorVoltage(float v) {
    // STUB: replace with PWM duty cycle conversion and analogWrite().
    // Example: analogWrite(MOTOR_PWM_PIN, voltageToDuty(v));
    (void)v;  // suppress unused-parameter warning in stub
}


// ============================================================================
//  Setup
// ============================================================================

void setup() {
    Serial.begin(115200);   // 2 Mbaud — low-latency logging

    // Wait up to 2 s for Serial monitor (USB CDC).
    // Skip wait on power-up without host connected.
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 2000) { /* spin */ }

    Serial.println("[PIRL] Firmware initialising...");
    Serial.printf("[PIRL] Control rate: %d Hz  dt: %.4f s\n",
                  CONTROL_HZ, DT_S);

    // NN weight self-test (golden value check, see pirl_self_test.h)
    PIRL_SELF_TEST();

    // Print CSV header
    Serial.println("t_ms,q,dq,u_pid,u_ff,u_nn,u_total,loop_us");

    // Arm the interval timer LAST — nothing should fire before setup completes
    controlTimer.begin(controlISR, CONTROL_PERIOD_US);

    Serial.println("[PIRL] Control loop armed.");
}


// ============================================================================
//  Main control loop
// ============================================================================

void loop() {
    if (!controlFlag) return;
    controlFlag = false;

    // -----------------------------------------------------------------------
    //  Latency measurement start
    // -----------------------------------------------------------------------
    const uint32_t t_start = micros();

    // -----------------------------------------------------------------------
    //  1. Sensor reads
    // -----------------------------------------------------------------------
    const float q   = readEncoder();
    const float dq  = estimateVelocity();
    // const float i_q = readCurrentSensor();   // available for current-loop

    // -----------------------------------------------------------------------
    //  2. Feed-forward torque  (inertia * reference acceleration)
    // -----------------------------------------------------------------------
    const float u_ff = J_EST * target_ddq;

    // -----------------------------------------------------------------------
    //  3. PIRL residual friction compensation
    //     Suppressed inside deadband — avoids dithering at rest.
    // -----------------------------------------------------------------------
    const float u_nn = pirl_infer(dq);

    // -----------------------------------------------------------------------
    //  4. PID with NN-aware anti-windup
    //     compute(measurement, dt, u_nn, u_ff) — integrator freeze checks
    //     u_pid + u_nn + u_ff against [V_MIN, V_MAX], not just u_pid alone.
    // -----------------------------------------------------------------------
    pid.setSetpoint(target_q);
    const float u_pid = pid.compute(q, DT_S, u_nn, u_ff);

    // -----------------------------------------------------------------------
    //  5. Combine and saturate
    // -----------------------------------------------------------------------
    const float u_total = clampf(u_pid + u_ff + u_nn, V_MIN, V_MAX);

    // -----------------------------------------------------------------------
    //  6. Output
    // -----------------------------------------------------------------------
    setMotorVoltage(u_total);

    // -----------------------------------------------------------------------
    //  7. Telemetry  (CSV, 2 Mbaud Serial)
    //     t_elapsed is the smoke-test latency metric — no separate test needed.
    // -----------------------------------------------------------------------
    const uint32_t t_elapsed = micros() - t_start;
    const uint32_t t_ms      = millis();

    Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%lu\n",
                  t_ms,
                  q, dq,
                  u_pid, u_ff, u_nn, u_total,
                  t_elapsed);
}
