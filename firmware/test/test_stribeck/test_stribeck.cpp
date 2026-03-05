/**
 * @file test_stribeck.cpp
 * @brief Unity test suite for the Stribeck friction model.
 *
 * Run with: pio test -e native_test
 *
 * Test groups:
 *   A. Edge cases       — zero velocity, very high velocity, near-zero
 *   B. Symmetry         — antisymmetry (odd function), sign correctness
 *   C. Golden values    — exact numerical results vs Python reference
 *   D. Parameter sweep  — delta=1 (linear), delta=2 (Gaussian), pure Coulomb
 *   E. Struct overload  — convenience wrapper produces identical results
 *   F. Jacobian         — analytical derivative correctness
 *   G. Physics checks   — monotonicity, Stribeck effect, viscous dominance
 *
 * Golden values computed in Python (float64) and rounded to float32:
 *   Standard params: Fc=0.15, Fs=0.35, vs=0.1, delta=2.0, sigma=0.01
 *
 *   v=0.0  → 0.00000000
 *   v=0.25 → 0.15288609  (verified: exp(-6.25)*0.2 + 0.15 + 0.01*0.25)
 *   v=10.0 → 0.25000000  (Fc + sigma*v = 0.15 + 0.1, Stribeck term ≈ 0)
 */

#include <unity.h>
#include "stribeck.h"
#include <math.h>

// Standard test parameters — match Phase 1 simulation defaults
static constexpr float FC    = 0.15f;
static constexpr float FS    = 0.35f;
static constexpr float VS    = 0.10f;
static constexpr float DELTA = 2.0f;
static constexpr float SIGMA = 0.01f;

static constexpr float TOL_LOOSE = 1e-4f;
static constexpr float TOL_TIGHT = 1e-5f;
static constexpr float TOL_EXACT = 1e-6f;


// ============================================================================
//  A. Edge cases
// ============================================================================

/**
 * At v=0, sign(0)=0, so the directional friction term vanishes entirely.
 * Only sigma*v=0 remains → output must be exactly 0.
 *
 * This is the numpy.sign() convention: no torque command at rest.
 * A non-zero result here would command a DC offset at standstill.
 */
void test_A1_zero_velocity_returns_zero() {
    float out = stribeck(0.0f, FC, FS, VS, DELTA, SIGMA);
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, 0.0f, out);
}

/**
 * At very high velocity, the Stribeck exponential term vanishes:
 *   exp(-(v/vs)^delta) → 0  for v >> vs
 * Remaining: sign(v)*(Fc + 0) + sigma*v = Fc + sigma*v
 *
 * At v=10.0: Fc=0.15, sigma*v=0.1 → expected=0.25
 * (exp(-(10/0.1)^2) = exp(-10000) is effectively 0 in float32)
 */
void test_A2_high_velocity_coulomb_plus_viscous_only() {
    float out = stribeck(10.0f, FC, FS, VS, DELTA, SIGMA);
    float expected = FC + SIGMA * 10.0f;   // 0.15 + 0.10 = 0.25
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, expected, out);
}

/**
 * Negative high velocity: same magnitude, opposite sign.
 */
void test_A3_high_velocity_negative() {
    float out = stribeck(-10.0f, FC, FS, VS, DELTA, SIGMA);
    float expected = -(FC + SIGMA * 10.0f);  // -0.25
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, expected, out);
}

/**
 * Near-zero positive velocity: output must be positive (friction opposes
 * any motion, even infinitesimally small velocity).
 */
void test_A4_near_zero_positive_velocity_positive_output() {
    float out = stribeck(1e-4f, FC, FS, VS, DELTA, SIGMA);
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, out);
}

/**
 * Near-zero negative velocity: output must be negative.
 */
void test_A5_near_zero_negative_velocity_negative_output() {
    float out = stribeck(-1e-4f, FC, FS, VS, DELTA, SIGMA);
    TEST_ASSERT_LESS_THAN_FLOAT(0.0f, out);
}


// ============================================================================
//  B. Symmetry (odd function)
// ============================================================================

/**
 * The Stribeck model is an odd function: F(-v) = -F(v).
 * This must hold exactly in float32.
 *
 * Violation means the sign logic or viscous term has an asymmetry bug —
 * the motor would command different friction magnitudes for identical
 * positive and negative velocities, which would corrupt the Symmetric
 * Harvesting Protocol's balanced training distribution.
 */
void test_B1_antisymmetry_at_stribeck_velocity() {
    float pos = stribeck( VS, FC, FS, VS, DELTA, SIGMA);
    float neg = stribeck(-VS, FC, FS, VS, DELTA, SIGMA);
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, pos, -neg);
}

void test_B2_antisymmetry_below_stribeck_velocity() {
    float v = VS * 0.3f;
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT,
        stribeck(v, FC, FS, VS, DELTA, SIGMA),
       -stribeck(-v, FC, FS, VS, DELTA, SIGMA));
}

void test_B3_antisymmetry_above_stribeck_velocity() {
    float v = VS * 5.0f;
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT,
        stribeck(v, FC, FS, VS, DELTA, SIGMA),
       -stribeck(-v, FC, FS, VS, DELTA, SIGMA));
}

void test_B4_antisymmetry_multiple_velocities() {
    float velocities[] = {0.01f, 0.05f, 0.1f, 0.5f, 1.0f, 5.0f};
    for (float v : velocities) {
        float pos = stribeck( v, FC, FS, VS, DELTA, SIGMA);
        float neg = stribeck(-v, FC, FS, VS, DELTA, SIGMA);
        TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, pos, -neg);
    }
}


// ============================================================================
//  C. Golden values (Python cross-validation)
// ============================================================================

/**
 * These values were computed in Python with float64 precision using the
 * identical formula and rounded to float32. A mismatch here indicates
 * a translation error in the C++ implementation.
 *
 * Python reference:
 *   import math
 *   def stribeck(v, Fc, Fs, vs, delta, sigma):
 *       sign_v = (1 if v>0 else -1 if v<0 else 0)
 *       abs_v  = abs(v)
 *       exp_t  = math.exp(-pow(abs_v/vs, delta))
 *       return sign_v*(Fc + (Fs-Fc)*exp_t) + sigma*v
 */
void test_C1_golden_value_v025() {
    // v=0.25: exp(-(2.5)^2)=exp(-6.25)=0.001930454
    // friction = 0.15 + 0.2*0.001930454 = 0.150386
    // output   = 0.150386 + 0.01*0.25   = 0.152886
    float out = stribeck(0.25f, FC, FS, VS, DELTA, SIGMA);
    TEST_ASSERT_FLOAT_WITHIN(TOL_LOOSE, 0.15288609f, out);
}

void test_C2_golden_value_v025_negative() {
    float out = stribeck(-0.25f, FC, FS, VS, DELTA, SIGMA);
    TEST_ASSERT_FLOAT_WITHIN(TOL_LOOSE, -0.15288609f, out);
}

void test_C3_golden_value_v005() {
    // v=0.05: (0.05/0.1)^2=0.25, exp(-0.25)=0.778801
    // friction = 0.15 + 0.2*0.778801 = 0.305760
    // output   = 0.305760 + 0.0005    = 0.306260
    float out = stribeck(0.05f, FC, FS, VS, DELTA, SIGMA);
    TEST_ASSERT_FLOAT_WITHIN(TOL_LOOSE, 0.30626016f, out);
}

void test_C4_golden_value_high_velocity() {
    float out = stribeck(10.0f, FC, FS, VS, DELTA, SIGMA);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, 0.25f, out);
}

/**
 * At exactly v=vs with delta=1 (linear Stribeck):
 *   exp(-(vs/vs)^1) = exp(-1) = 0.367879441
 *   friction = 0.15 + 0.2*0.367879441 = 0.223575888
 *   output   = 0.223575888 + 0.01*0.1 = 0.224575888
 */
void test_C5_golden_value_at_stribeck_velocity_delta1() {
    float out = stribeck(VS, FC, FS, VS, 1.0f, SIGMA);
    TEST_ASSERT_FLOAT_WITHIN(TOL_LOOSE, 0.22457589f, out);
}


// ============================================================================
//  D. Parameter sweep
// ============================================================================

/**
 * delta=1 (exponential/linear Stribeck): output at v=VS should be
 * between Fc and Fs. Verified against golden value in test_C5.
 */
void test_D1_delta1_output_between_fc_and_fs_at_vs() {
    float out = stribeck(VS, FC, FS, VS, 1.0f, SIGMA);
    TEST_ASSERT_GREATER_THAN_FLOAT(FC, out);    // must exceed Coulomb
    TEST_ASSERT_LESS_THAN_FLOAT(FS, out);       // must be below static
}

/**
 * delta=2 (Gaussian Stribeck): steeper transition. At v=VS the
 * Stribeck contribution is much smaller than delta=1.
 */
void test_D2_delta2_steeper_than_delta1_above_vs() {
    // At v == vs the ratio |v|/vs == 1, so 1^delta == 1 for any delta —
    // both curves are identical at exactly that point.  The steeper decay
    // of delta=2 is only visible at v > vs.  Evaluate at 2*vs:
    //   delta=1: exp(-2)   = 0.1353  → output ≈ 0.179
    //   delta=2: exp(-4)   = 0.0183  → output ≈ 0.156
    // delta=2 decays faster → lower output above vs.
    float v     = VS * 2.0f;
    float out_d1 = stribeck(v, FC, FS, VS, 1.0f, SIGMA);
    float out_d2 = stribeck(v, FC, FS, VS, 2.0f, SIGMA);
    TEST_ASSERT_GREATER_THAN_FLOAT(out_d2, out_d1);  // out_d1 > out_d2
}

/**
 * Pure Coulomb: Fs==Fc collapses the Stribeck term regardless of velocity.
 * Output must equal sign(v)*Fc + sigma*v exactly.
 */
void test_D3_pure_coulomb_when_fs_equals_fc() {
    float v = 0.5f;
    float out = stribeck(v, FC, FC, VS, DELTA, SIGMA);  // Fs=Fc
    float expected = FC + SIGMA * v;
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, expected, out);
}

/**
 * Pure viscous: Fc=Fs=0 → only viscous damping term remains.
 * Output = sigma*v.
 */
void test_D4_pure_viscous_when_coulomb_zero() {
    float v = 2.0f;
    float out = stribeck(v, 0.0f, 0.0f, VS, DELTA, SIGMA);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, SIGMA * v, out);
}

/**
 * Zero viscous: sigma=0 → output is purely directional friction.
 * F(v) == F(-v) in magnitude (before sign application).
 */
void test_D5_zero_viscous_exact_antisymmetry() {
    float pos = stribeck( 0.3f, FC, FS, VS, DELTA, 0.0f);
    float neg = stribeck(-0.3f, FC, FS, VS, DELTA, 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(TOL_TIGHT, pos, -neg);
}


// ============================================================================
//  E. Struct overload
// ============================================================================

/**
 * The struct overload stribeck(v, params) must produce bit-identical
 * results to the explicit-parameter version for all test velocities.
 */
void test_E1_struct_overload_matches_explicit() {
    StribeckParams p(FC, FS, VS, DELTA, SIGMA);
    float velocities[] = {-1.0f, -0.1f, -0.01f, 0.0f, 0.01f, 0.1f, 1.0f};
    for (float v : velocities) {
        float explicit_result = stribeck(v, FC, FS, VS, DELTA, SIGMA);
        float struct_result   = stribeck(v, p);
        TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, explicit_result, struct_result);
    }
}

/**
 * StribeckParams default constructor must produce the same result as
 * manually specifying the default values.
 */
void test_E2_default_params_match_named_defaults() {
    StribeckParams p_default;
    StribeckParams p_explicit(STRIBECK_FC_DEFAULT,
                              STRIBECK_FS_DEFAULT,
                              STRIBECK_VS_DEFAULT,
                              STRIBECK_DELTA_DEFAULT,
                              STRIBECK_SIGMA_DEFAULT);
    float v = 0.3f;
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT,
        stribeck(v, p_default),
        stribeck(v, p_explicit));
}


// ============================================================================
//  F. Jacobian
// ============================================================================

/**
 * At v=0, the Jacobian is undefined (non-smooth kink).
 * The implementation returns sigma as the safe approximation.
 */
void test_F1_jacobian_at_zero_returns_sigma() {
    StribeckParams p(FC, FS, VS, DELTA, SIGMA);
    float jac = stribeck_jacobian(0.0f, p);
    TEST_ASSERT_FLOAT_WITHIN(TOL_EXACT, SIGMA, jac);
}

/**
 * At v=vs with delta=2, the analytical Jacobian is negative:
 * friction is decreasing with velocity (Stribeck effect).
 *   u=1, dg/du = -(Fs-Fc)*delta*1*exp(-1) = -0.2*2*0.3679 = -0.14715
 *   dF/dv = sigma + dg_du/vs = 0.01 + (-0.14715)/0.1 = -1.4615
 */
void test_F2_jacobian_negative_in_stribeck_regime() {
    StribeckParams p(FC, FS, VS, DELTA, SIGMA);
    float jac = stribeck_jacobian(VS, p);
    TEST_ASSERT_FLOAT_WITHIN(TOL_LOOSE, -1.46152f, jac);
    // Must be negative — friction decreases with velocity in this regime
    TEST_ASSERT_LESS_THAN_FLOAT(0.0f, jac);
}

/**
 * At very high velocity, Stribeck contribution vanishes and Jacobian → sigma.
 */
void test_F3_jacobian_approaches_sigma_at_high_velocity() {
    StribeckParams p(FC, FS, VS, DELTA, SIGMA);
    float jac = stribeck_jacobian(100.0f, p);
    TEST_ASSERT_FLOAT_WITHIN(TOL_LOOSE, SIGMA, jac);
}

/**
 * Jacobian must be positive at high velocity (viscous dominates — output
 * increases with velocity once the Stribeck effect is negligible).
 */
void test_F4_jacobian_positive_at_high_velocity() {
    StribeckParams p(FC, FS, VS, DELTA, SIGMA);
    float jac = stribeck_jacobian(10.0f, p);
    TEST_ASSERT_GREATER_THAN_FLOAT(0.0f, jac);
}


// ============================================================================
//  G. Physics checks
// ============================================================================

/**
 * The Stribeck effect: friction magnitude must decrease as velocity rises
 * from just above 0 up to a few multiples of vs.
 * After the Stribeck dip, viscous friction dominates and magnitude increases.
 */
void test_G1_stribeck_dip_then_viscous_rise() {
    // Friction magnitude must decrease from v_low to v_mid (Stribeck dip)
    float f_low  = stribeck(0.01f, FC, FS, VS, DELTA, SIGMA);
    float f_mid  = stribeck(0.20f, FC, FS, VS, DELTA, SIGMA);
    float f_high = stribeck(5.0f,  FC, FS, VS, DELTA, SIGMA);

    TEST_ASSERT_GREATER_THAN_FLOAT(f_mid,  f_low);   // f_low  > f_mid:  Stribeck dip
    TEST_ASSERT_GREATER_THAN_FLOAT(f_mid,  f_high);  // f_high > f_mid:  viscous rise
}

/**
 * Output at any non-zero velocity must be at least Fc in magnitude.
 * Coulomb friction is always present once moving.
 */
void test_G2_output_always_at_least_coulomb_when_moving() {
    float velocities[] = {0.001f, 0.01f, 0.1f, 1.0f, 10.0f};
    for (float v : velocities) {
        float out = stribeck(v, FC, FS, VS, DELTA, 0.0f);  // sigma=0
        TEST_ASSERT_GREATER_THAN_FLOAT(FC * 0.99f, out);  // at least ~Fc
    }
}

/**
 * At very low velocity, output must not exceed Fs in magnitude.
 * Static friction is the upper bound on the directional component.
 */
void test_G3_output_does_not_exceed_fs_at_low_velocity() {
    float out = stribeck(1e-5f, FC, FS, VS, DELTA, 0.0f);  // sigma=0
    TEST_ASSERT_LESS_THAN_FLOAT(FS + TOL_LOOSE, out);
}

/**
 * Increasing sigma must increase output monotonically for positive velocity.
 */
void test_G4_viscous_coefficient_monotone_effect() {
    float v = 1.0f;
    float out_low  = stribeck(v, FC, FS, VS, DELTA, 0.005f);
    float out_mid  = stribeck(v, FC, FS, VS, DELTA, 0.01f);
    float out_high = stribeck(v, FC, FS, VS, DELTA, 0.05f);
    TEST_ASSERT_GREATER_THAN_FLOAT(out_low,  out_mid);   // mid > low
    TEST_ASSERT_GREATER_THAN_FLOAT(out_mid,  out_high);  // high > mid
}


// ============================================================================
//  Unity runner
// ============================================================================

void setUp()    {}
void tearDown() {}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    UNITY_BEGIN();

    // A. Edge cases
    RUN_TEST(test_A1_zero_velocity_returns_zero);
    RUN_TEST(test_A2_high_velocity_coulomb_plus_viscous_only);
    RUN_TEST(test_A3_high_velocity_negative);
    RUN_TEST(test_A4_near_zero_positive_velocity_positive_output);
    RUN_TEST(test_A5_near_zero_negative_velocity_negative_output);

    // B. Symmetry
    RUN_TEST(test_B1_antisymmetry_at_stribeck_velocity);
    RUN_TEST(test_B2_antisymmetry_below_stribeck_velocity);
    RUN_TEST(test_B3_antisymmetry_above_stribeck_velocity);
    RUN_TEST(test_B4_antisymmetry_multiple_velocities);

    // C. Golden values
    RUN_TEST(test_C1_golden_value_v025);
    RUN_TEST(test_C2_golden_value_v025_negative);
    RUN_TEST(test_C3_golden_value_v005);
    RUN_TEST(test_C4_golden_value_high_velocity);
    RUN_TEST(test_C5_golden_value_at_stribeck_velocity_delta1);

    // D. Parameter sweep
    RUN_TEST(test_D1_delta1_output_between_fc_and_fs_at_vs);
    RUN_TEST(test_D2_delta2_steeper_than_delta1_above_vs);
    RUN_TEST(test_D3_pure_coulomb_when_fs_equals_fc);
    RUN_TEST(test_D4_pure_viscous_when_coulomb_zero);
    RUN_TEST(test_D5_zero_viscous_exact_antisymmetry);

    // E. Struct overload
    RUN_TEST(test_E1_struct_overload_matches_explicit);
    RUN_TEST(test_E2_default_params_match_named_defaults);

    // F. Jacobian
    RUN_TEST(test_F1_jacobian_at_zero_returns_sigma);
    RUN_TEST(test_F2_jacobian_negative_in_stribeck_regime);
    RUN_TEST(test_F3_jacobian_approaches_sigma_at_high_velocity);
    RUN_TEST(test_F4_jacobian_positive_at_high_velocity);

    // G. Physics checks
    RUN_TEST(test_G1_stribeck_dip_then_viscous_rise);
    RUN_TEST(test_G2_output_always_at_least_coulomb_when_moving);
    RUN_TEST(test_G3_output_does_not_exceed_fs_at_low_velocity);
    RUN_TEST(test_G4_viscous_coefficient_monotone_effect);

    return UNITY_END();
}
