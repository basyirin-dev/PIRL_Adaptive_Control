/**
 * @file test_pid.cpp
 * @brief Unity test suite for PIDController.
 *
 * Run with: pio test -e native_test
 *
 * Test groups:
 *   A. Term isolation       — P, I, D in isolation to verify each gain path
 *   B. First-call guard     — derivative spike suppression on first compute()
 *   C. Anti-windup FREEZE   — the critical PIRL failure mode (dossier §5)
 *   D. Anti-windup PERMIT   — unwinding case must not be blocked
 *   E. NN-aware differential — shows standard AW would fail where PIRL AW passes
 *   F. dt guard             — degenerate time step safety
 *   G. State management     — reset(), setIntegralState(), decomposition invariant
 *   H. Derivative filter    — low-pass coefficient affects filtered output
 *   I. Output decomposition — u_p + u_i + u_d == returned u_pid
 */

#include <unity.h>
#include "pid.h"

// ---------------------------------------------------------------------------
//  Tolerances
//  LOOSE  (1e-4f): acceptable for float32 accumulation over many steps
//  TIGHT  (1e-5f): expected for single-step arithmetic
//  EXACT  (1e-6f): required for values that should be identically zero
// ---------------------------------------------------------------------------
static constexpr float TOL_LOOSE = 1e-4f;
static constexpr float TOL_TIGHT = 1e-5f;
static constexpr float TOL_EXACT = 1e-6f;


// ============================================================================
//  A. Term isolation
// ============================================================================

/**
 * P-only step: kp=10, setpoint=1, measurement=0 → error=1 → u_pid=10.
 * Verifies the proportional path end-to-end.
 */
void test_A1_p_only_step_response() {
    PIDController pid(10.0f, 0.0f, 0.0f, -100.0f, 100.0f, 1.0f);
    float out = pid.compute(0.0f, 0.01f);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 10.0f, out);
}

/**
 * P-only: output tracks error linearly.
 * Three different measurements, same gain — output must be kp*error.
 */
void test_A2_p_only_linear_tracking() {
    PIDController pid(5.0f, 0.0f, 0.0f, -100.0f, 100.0f, 2.0f);
    // setpoint=2, measurement=0.5 → error=1.5 → u=7.5
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 7.5f,  pid.compute(0.5f, 0.01f));
    pid.reset();
    // setpoint=2, measurement=2.0 → error=0 → u=0
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f,  pid.compute(2.0f, 0.01f));
    pid.reset();
    // setpoint=2, measurement=3.0 → error=-1 → u=-5
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, -5.0f, pid.compute(3.0f, 0.01f));
}

/**
 * I-only: integrator accumulates correctly over three steps.
 * ki=1, error=1, dt=0.01 → integral grows by 0.01 each step.
 * Wide limits so anti-windup never fires.
 */
void test_A3_i_only_accumulation() {
    PIDController pid(0.0f, 1.0f, 0.0f, -100.0f, 100.0f, 1.0f);
    pid.compute(0.0f, 0.01f);   // integral → 0.01,  u_i = 0.01
    pid.compute(0.0f, 0.01f);   // integral → 0.02,  u_i = 0.02
    float out = pid.compute(0.0f, 0.01f);   // integral → 0.03, u_i = 0.03
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 0.03f, out);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 0.03f, pid.getIntegral());
}

/**
 * D-only: on the second call, derivative responds to a step change in error.
 * Alpha=1 (raw finite difference, no filtering) so the math is exact:
 *   call 1: measurement=0.5 → error=0.5, first_call → deriv=0, u_d=0
 *   call 2: measurement=0.0 → error=1.0, deriv_raw=(1.0-0.5)/0.01=50, u_d=50
 */
void test_A4_d_only_step_derivative() {
    PIDController pid(0.0f, 0.0f, 1.0f, -1000.0f, 1000.0f, 1.0f);
    pid.setDerivativeFilter(1.0f);          // raw finite difference
    pid.compute(0.5f, 0.01f);              // first call — deriv suppressed
    float out = pid.compute(0.0f, 0.01f); // second call — deriv fires
    // error went from 0.5 to 1.0 over dt=0.01: deriv=(0.5)/0.01=50
    TEST_ASSERT_FLOAT_WITHIN(TOL_LOOSE, 50.0f, out);
    TEST_ASSERT_FLOAT_WITHIN(TOL_LOOSE, 50.0f, pid.getDerivativeTerm());
}


// ============================================================================
//  B. First-call guard (derivative spike suppression)
// ============================================================================

/**
 * On the very first compute() call, the derivative term must be zero
 * regardless of the measurement value.
 * Without this guard, (error - 0) / dt would produce a large spike.
 */
void test_B1_first_call_no_derivative_spike() {
    // kd=100, large gain to amplify any spike
    PIDController pid(0.0f, 0.0f, 100.0f, -10000.0f, 10000.0f, 1.0f);
    pid.setDerivativeFilter(1.0f);
    float out = pid.compute(0.0f, 0.01f);  // first call, error=1
    // u_d must be 0, not kd * error / dt = 100 * 1 / 0.01 = 10000
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getDerivativeTerm());
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, out);
}

/**
 * After reset(), the first-call guard is reinstated.
 * A second fresh start must again suppress the derivative.
 */
void test_B2_reset_reinstates_first_call_guard() {
    PIDController pid(0.0f, 0.0f, 100.0f, -10000.0f, 10000.0f, 1.0f);
    pid.setDerivativeFilter(1.0f);
    pid.compute(0.0f, 0.01f);   // first call — deriv=0
    pid.compute(0.0f, 0.01f);   // second call — deriv fires, state built up
    pid.reset();
    float out = pid.compute(0.0f, 0.01f);  // must behave like first call again
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getDerivativeTerm());
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, out);
}


// ============================================================================
//  C. Anti-windup FREEZE (the critical PIRL failure mode — dossier §5)
// ============================================================================

/**
 * CORE TEST: integrator must freeze when u_total is saturated AND
 * error would deepen saturation.
 *
 * Setup:
 *   integral = 11.5 (positive, pre-loaded)
 *   u_pid_current = ki * integral = 11.5
 *   u_nn = 1.0
 *   u_total_est = 11.5 + 1.0 = 12.5  >  OUTPUT_MAX=12.0  → saturated
 *   error = 1.0 (positive, same sign as integral)             → same_sign
 *   → FREEZE expected
 */
void test_C1_antiwindup_freezes_saturated_same_sign() {
    PIDController pid(0.0f, 1.0f, 0.0f, -12.0f, 12.0f, 1.0f);
    pid.setIntegralState(11.5f, 0.0f, 0.0f);   // integral = 11.5
    float integral_before = pid.getIntegral();

    // measurement=0, setpoint=1 → error=+1.0 (positive, same sign as integral)
    // u_nn=1.0 pushes u_total over OUTPUT_MAX
    pid.compute(0.0f, 0.01f, 1.0f, 0.0f);

    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, integral_before, pid.getIntegral());
}

/**
 * Freeze must also work in the negative direction.
 * integral=-11.5, error=-1.0, u_nn=-1.0 → u_total < OUTPUT_MIN → FREEZE.
 */
void test_C2_antiwindup_freezes_negative_saturation() {
    PIDController pid(0.0f, 1.0f, 0.0f, -12.0f, 12.0f, -1.0f); // setpoint=-1
    pid.setIntegralState(-11.5f, 0.0f, 0.0f);  // integral=-11.5
    float integral_before = pid.getIntegral();

    // measurement=0, setpoint=-1 → error=-1.0 (negative, same sign as integral)
    // u_nn=-1.0 pushes u_total below OUTPUT_MIN
    pid.compute(0.0f, 0.01f, -1.0f, 0.0f);

    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, integral_before, pid.getIntegral());
}

/**
 * Standard (non-NN-aware) anti-windup would pass this: u_pid alone is NOT
 * saturated, so it would integrate freely. But u_pid + u_nn IS saturated.
 * The NN-aware check must catch it.
 *
 * Setup:
 *   integral=10.5, u_pid=10.5 (below OUTPUT_MAX=12 — standard AW would pass)
 *   u_nn=2.0  →  u_total=12.5 > 12  →  NN-aware AW must FREEZE
 */
void test_C3_nn_aware_catches_what_standard_aw_misses() {
    PIDController pid(0.0f, 1.0f, 0.0f, -12.0f, 12.0f, 1.0f);
    pid.setIntegralState(10.5f, 0.0f, 0.0f);  // u_pid=10.5 < 12 alone
    float integral_before = pid.getIntegral();

    // u_nn=2.0: standard AW (checking u_pid only) would permit integration.
    // NN-aware AW (checking u_pid+u_nn=12.5) must FREEZE.
    pid.compute(0.0f, 0.01f, 2.0f, 0.0f);

    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, integral_before, pid.getIntegral());
}

/**
 * Feed-forward contributes to saturation check too.
 * integral=10.5, u_ff=2.0 → u_total=12.5 > 12 → FREEZE.
 */
void test_C4_feedforward_contributes_to_antiwindup() {
    PIDController pid(0.0f, 1.0f, 0.0f, -12.0f, 12.0f, 1.0f);
    pid.setIntegralState(10.5f, 0.0f, 0.0f);
    float integral_before = pid.getIntegral();

    pid.compute(0.0f, 0.01f, 0.0f, 2.0f);  // u_ff=2.0 causes saturation

    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, integral_before, pid.getIntegral());
}


// ============================================================================
//  D. Anti-windup PERMIT (unwinding must not be blocked)
// ============================================================================

/**
 * When output is saturated but error opposes the integral direction,
 * the integrator must be allowed to unwind.
 *
 * Setup:
 *   integral=11.5 (positive), u_nn=1.0 → saturated
 *   error=-1.0 (negative) → same_sign=false → PERMIT
 *   integral must decrease after the call.
 */
void test_D1_antiwindup_permits_unwinding_positive_integral() {
    PIDController pid(0.0f, 1.0f, 0.0f, -12.0f, 12.0f, 0.0f); // setpoint=0
    pid.setIntegralState(11.5f, 0.0f, 0.0f);

    // measurement=1.0, setpoint=0 → error=-1.0 (opposing positive integral)
    pid.compute(1.0f, 0.01f, 1.0f, 0.0f);

    // integral must have decreased: 11.5 + (-1.0)*0.01 = 11.49
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 11.49f, pid.getIntegral());
}

/**
 * Symmetric case: negative integral unwinding with positive error.
 */
void test_D2_antiwindup_permits_unwinding_negative_integral() {
    PIDController pid(0.0f, 1.0f, 0.0f, -12.0f, 12.0f, 1.0f); // setpoint=1
    pid.setIntegralState(-11.5f, 0.0f, 0.0f);

    // measurement=0, setpoint=1 → error=+1.0 (opposing negative integral)
    pid.compute(0.0f, 0.01f, -1.0f, 0.0f);

    // integral: -11.5 + 1.0*0.01 = -11.49
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, -11.49f, pid.getIntegral());
}

/**
 * Well inside limits: anti-windup must never block integration.
 * Wide limits (±100) ensure saturation can never be reached here.
 */
void test_D3_antiwindup_never_blocks_when_unsaturated() {
    PIDController pid(0.0f, 1.0f, 0.0f, -100.0f, 100.0f, 1.0f);
    // 10 steps — integrator must grow every step regardless of u_nn
    float integral_prev = 0.0f;
    for (int i = 0; i < 10; i++) {
        pid.compute(0.0f, 0.01f, 5.0f, 5.0f);  // non-zero u_nn and u_ff
        float integral_now = pid.getIntegral();
        TEST_ASSERT_GREATER_THAN_FLOAT(integral_prev, integral_now);
        integral_prev = integral_now;
    }
}


// ============================================================================
//  E. NN-aware anti-windup differential
//     (most important test for the PIRL architecture)
// ============================================================================

/**
 * This test encodes the exact failure mode described in dossier §5.
 *
 * Two controllers, identical gains, same integral pre-load.
 * Only difference: one is called with u_nn=0 (standard AW scenario),
 * one with u_nn=2.0 (NN-compensating, combined output saturated).
 *
 * Expected:
 *   Without u_nn: u_pid alone < OUTPUT_MAX → AW permits → integral grows.
 *   With    u_nn: u_total > OUTPUT_MAX     → AW freezes → integral unchanged.
 */
void test_E1_nn_aware_differential_freeze_vs_permit() {
    // Controller A: u_nn=0 (as if NN is disabled)
    PIDController pid_a(0.0f, 1.0f, 0.0f, -12.0f, 12.0f, 1.0f);
    pid_a.setIntegralState(10.5f, 0.0f, 0.0f);
    float int_a_before = pid_a.getIntegral();
    pid_a.compute(0.0f, 0.01f, 0.0f, 0.0f);   // u_nn=0: u_total=10.5 < 12
    float int_a_after = pid_a.getIntegral();

    // Controller B: u_nn=2.0 (NN contributing at saturation boundary)
    PIDController pid_b(0.0f, 1.0f, 0.0f, -12.0f, 12.0f, 1.0f);
    pid_b.setIntegralState(10.5f, 0.0f, 0.0f);
    float int_b_before = pid_b.getIntegral();
    pid_b.compute(0.0f, 0.01f, 2.0f, 0.0f);   // u_nn=2: u_total=12.5 > 12

    float int_b_after = pid_b.getIntegral();

    // A should have integrated (grew)
    TEST_ASSERT_GREATER_THAN_FLOAT(int_a_before, int_a_after);
    // B must be frozen (unchanged)
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, int_b_before, int_b_after);
}


// ============================================================================
//  F. dt guard
// ============================================================================

/**
 * dt=0 must return the last valid output without modifying any state.
 * Prevents NaN from divide-by-zero in the derivative term.
 */
void test_F1_zero_dt_returns_last_output() {
    PIDController pid(10.0f, 0.0f, 0.0f, -100.0f, 100.0f, 1.0f);
    float out1 = pid.compute(0.0f, 0.01f);    // valid: error=1, u_pid=10
    float out2 = pid.compute(0.5f, 0.0f);     // dt=0: must return out1=10
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, out1, out2);
}

/**
 * Negative dt must also be rejected (e.g. timer wraparound).
 */
void test_F2_negative_dt_returns_last_output() {
    PIDController pid(10.0f, 0.0f, 0.0f, -100.0f, 100.0f, 1.0f);
    float out1 = pid.compute(0.0f, 0.01f);
    float out2 = pid.compute(0.5f, -0.005f);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, out1, out2);
}

/**
 * dt=0 must not corrupt the integrator state.
 * After a zero-dt call, a subsequent valid call must produce the correct result.
 */
void test_F3_zero_dt_does_not_corrupt_integrator() {
    PIDController pid(0.0f, 1.0f, 0.0f, -100.0f, 100.0f, 1.0f);
    pid.compute(0.0f, 0.01f);             // integral → 0.01
    float int_before = pid.getIntegral();

    pid.compute(0.0f, 0.0f);              // dt=0: integrator must not change
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, int_before, pid.getIntegral());

    pid.compute(0.0f, 0.01f);             // valid call: integral → 0.02
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 0.02f, pid.getIntegral());
}


// ============================================================================
//  G. State management
// ============================================================================

/**
 * reset() must zero all state: integral, derivative, cached terms,
 * and reinstate the first-call guard.
 */
void test_G1_reset_clears_all_state() {
    PIDController pid(5.0f, 2.0f, 1.0f, -100.0f, 100.0f, 1.0f);
    for (int i = 0; i < 20; i++) pid.compute(0.0f, 0.01f);

    pid.reset();

    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getIntegral());
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getDerivative());
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getProportional());
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getIntegralTerm());
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getDerivativeTerm());
    // First-call guard reinstated: next compute must produce zero derivative
    pid.compute(0.0f, 0.01f);
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getDerivativeTerm());
}

/**
 * setIntegralState() bumpless transfer:
 * The integral must be loaded so that ki*integral == (u_prior - u_nn - u_ff).
 *
 * With ki=2, u_prior=8.0, u_nn=0, u_ff=0:
 *   integral = (8.0 - 0 - 0 - u_p - u_d) / 2.0 = 4.0  (fresh: u_p=u_d=0)
 */
void test_G2_bumpless_transfer_integral_preload() {
    PIDController pid(0.0f, 2.0f, 0.0f, -100.0f, 100.0f, 1.0f);
    pid.setIntegralState(8.0f, 0.0f, 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 4.0f, pid.getIntegral());
    // Verify the integral term matches: u_i = ki*integral = 2*4 = 8
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 8.0f, pid.getIntegralTerm());
}

/**
 * setIntegralState() with u_nn offset:
 * With ki=1, u_prior=10, u_nn=3:
 *   u_pid_target = 10 - 3 = 7 → integral = 7.0
 */
void test_G3_bumpless_transfer_with_nn_offset() {
    PIDController pid(0.0f, 1.0f, 0.0f, -100.0f, 100.0f, 0.0f);
    pid.setIntegralState(10.0f, 3.0f, 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 7.0f, pid.getIntegral());
}

/**
 * setIntegralState() with ki=0 (pure PD controller) must not divide by zero.
 * integral must remain 0.
 */
void test_G4_bumpless_transfer_zero_ki_no_divide_by_zero() {
    PIDController pid(5.0f, 0.0f, 0.0f, -100.0f, 100.0f, 1.0f);
    pid.setIntegralState(6.0f, 0.0f, 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, pid.getIntegral());
}

/**
 * setpoint can be updated mid-run without reset.
 */
void test_G5_setpoint_update_mid_run() {
    PIDController pid(10.0f, 0.0f, 0.0f, -100.0f, 100.0f, 0.0f);
    float out1 = pid.compute(0.0f, 0.01f);   // setpoint=0, error=0, out=0
    pid.setSetpoint(2.0f);
    float out2 = pid.compute(0.0f, 0.01f);   // setpoint=2, error=2, out=20
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, out1);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 20.0f, out2);
}


// ============================================================================
//  H. Derivative filter
// ============================================================================

/**
 * With alpha=1.0 (no filtering), the derivative is the raw finite difference.
 * With alpha=0.1 (heavy filtering), the derivative is attenuated.
 * Both must have the same sign for the same step direction.
 */
void test_H1_derivative_filter_attenuates() {
    // Raw (alpha=1.0)
    PIDController pid_raw(0.0f, 0.0f, 1.0f, -1000.0f, 1000.0f, 1.0f);
    pid_raw.setDerivativeFilter(1.0f);
    pid_raw.compute(1.0f, 0.01f);  // first call, deriv=0
    pid_raw.compute(0.0f, 0.01f);  // step: error goes from 0 to 1
    float d_raw = pid_raw.getDerivative();

    // Filtered (alpha=0.1)
    PIDController pid_filt(0.0f, 0.0f, 1.0f, -1000.0f, 1000.0f, 1.0f);
    pid_filt.setDerivativeFilter(0.1f);
    pid_filt.compute(1.0f, 0.01f);
    pid_filt.compute(0.0f, 0.01f);
    float d_filt = pid_filt.getDerivative();

    // Filtered must be smaller in magnitude than raw
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, fabsf(d_filt));   // non-zero
    TEST_ASSERT_LESS_THAN_FLOAT(fabsf(d_raw), fabsf(d_filt)); // attenuated
    // Same sign
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, d_raw  * d_filt);
}


// ============================================================================
//  I. Output decomposition invariant
// ============================================================================

/**
 * compute() must return exactly u_p + u_i + u_d.
 * Verified across multiple steps with all terms active.
 */
void test_I1_output_equals_sum_of_terms() {
    PIDController pid(5.0f, 2.0f, 0.5f, -100.0f, 100.0f, 1.0f);
    float measurements[] = {0.0f, 0.2f, 0.5f, 0.8f, 1.0f, 0.9f, 1.1f};
    for (float m : measurements) {
        float out = pid.compute(m, 0.01f);
        float sum = pid.getProportional()
                  + pid.getIntegralTerm()
                  + pid.getDerivativeTerm();
        TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, sum, out);
    }
}


// ============================================================================
//  Unity runner
// ============================================================================

void setUp()    {}
void tearDown() {}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    UNITY_BEGIN();

    // A. Term isolation
    RUN_TEST(test_A1_p_only_step_response);
    RUN_TEST(test_A2_p_only_linear_tracking);
    RUN_TEST(test_A3_i_only_accumulation);
    RUN_TEST(test_A4_d_only_step_derivative);

    // B. First-call guard
    RUN_TEST(test_B1_first_call_no_derivative_spike);
    RUN_TEST(test_B2_reset_reinstates_first_call_guard);

    // C. Anti-windup FREEZE
    RUN_TEST(test_C1_antiwindup_freezes_saturated_same_sign);
    RUN_TEST(test_C2_antiwindup_freezes_negative_saturation);
    RUN_TEST(test_C3_nn_aware_catches_what_standard_aw_misses);
    RUN_TEST(test_C4_feedforward_contributes_to_antiwindup);

    // D. Anti-windup PERMIT
    RUN_TEST(test_D1_antiwindup_permits_unwinding_positive_integral);
    RUN_TEST(test_D2_antiwindup_permits_unwinding_negative_integral);
    RUN_TEST(test_D3_antiwindup_never_blocks_when_unsaturated);

    // E. NN-aware differential
    RUN_TEST(test_E1_nn_aware_differential_freeze_vs_permit);

    // F. dt guard
    RUN_TEST(test_F1_zero_dt_returns_last_output);
    RUN_TEST(test_F2_negative_dt_returns_last_output);
    RUN_TEST(test_F3_zero_dt_does_not_corrupt_integrator);

    // G. State management
    RUN_TEST(test_G1_reset_clears_all_state);
    RUN_TEST(test_G2_bumpless_transfer_integral_preload);
    RUN_TEST(test_G3_bumpless_transfer_with_nn_offset);
    RUN_TEST(test_G4_bumpless_transfer_zero_ki_no_divide_by_zero);
    RUN_TEST(test_G5_setpoint_update_mid_run);

    // H. Derivative filter
    RUN_TEST(test_H1_derivative_filter_attenuates);

    // I. Decomposition invariant
    RUN_TEST(test_I1_output_equals_sum_of_terms);

    return UNITY_END();
}
