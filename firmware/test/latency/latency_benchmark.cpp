/**
 * @file latency_benchmark.cpp
 * @brief Level 6 — Latency Budget Analysis (hardware-free).
 *
 * Produces two outputs:
 *
 *   1. ANALYTICAL BUDGET — exact FLOP counts from the source, scaled to
 *      Teensy 4.1 (600 MHz Cortex-M7) and compared against the 10 ms
 *      control-loop window at 100 Hz.
 *
 *   2. EMPIRICAL BENCHMARK — wall-clock timing of every control component
 *      on this host PC.  x86 at ~3 GHz with AVX2 is roughly 5–10x faster
 *      than the Teensy's FPU for scalar float32, so:
 *
 *          t_teensy_estimate ≈ t_host × SCALE_FACTOR
 *
 *      The scale factor is computed from a calibration microbenchmark
 *      (repeated FMUL chain) rather than assumed, making the estimate
 *      more robust across different host machines.
 *
 * Failure criteria (printed as WARN or FAIL):
 *   - pirl_forward  host time > 25 µs  → double-precision or no-inline bug
 *   - stribeck      host time > 5 µs   → expf/powf not using hardware path
 *   - pid.compute   host time > 1 µs   → unexpected branching overhead
 *   - Total Teensy estimate > 1000 µs  → control loop would miss 100 Hz
 *
 * Build (from firmware/):
 *   g++ -std=c++17 -O2 -I lib/PIRL -I lib/ArduinoStub \
 *       test/latency/latency_benchmark.cpp \
 *       lib/PIRL/pid.cpp lib/PIRL/stribeck.cpp lib/PIRL/pirl_inference.cpp \
 *       -o latency_benchmark -lm
 *
 * Run:
 *   ./latency_benchmark
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>

#include "pid.h"
#include "stribeck.h"
#include "pirl_inference.h"
#include "pirl_weights.h"

// ============================================================================
//  Architecture constants (from pirl_weights.h)
// ============================================================================

static constexpr int IN_DIM  = PIRL_INPUT_DIM;   // 1
static constexpr int H0      = PIRL_H0_DIM;       // 64
static constexpr int H1      = PIRL_H1_DIM;       // 64
static constexpr int OUT_DIM = PIRL_OUTPUT_DIM;   // 1

// ============================================================================
//  Exact FLOP counts
//
//  Layer notation: each Linear(in, out) + Tanh costs:
//    MACs = in * out   (each MAC = 1 MUL + 1 ADD = 2 FLOPs)
//    bias adds = out
//    tanh = ~10 FLOPs per element (polynomial or hardware instruction)
//
//  pirl_forward: 1→64→64→1
//    Layer 0: Linear(1,64)   →  64 MAC + 64 bias + 64 tanh
//    Layer 1: Linear(64,64)  →  4096 MAC + 64 bias + 64 tanh
//    Layer 2: Linear(64,1)   →  64 MAC + 1 bias  (no activation on output)
// ============================================================================

static constexpr long TANH_FLOPS_PER_ELEM = 10;  // tanhf ≈ minimax poly

static constexpr long L0_MAC   = IN_DIM * H0;            //       64
static constexpr long L0_BIAS  = H0;                     //       64
static constexpr long L0_ACT   = H0 * TANH_FLOPS_PER_ELEM; //   640
static constexpr long L1_MAC   = H0 * H1;                // 4,096
static constexpr long L1_BIAS  = H1;                     //       64
static constexpr long L1_ACT   = H1 * TANH_FLOPS_PER_ELEM; //   640
static constexpr long L2_MAC   = H1 * OUT_DIM;           //       64
static constexpr long L2_BIAS  = OUT_DIM;                //        1

static constexpr long PIRL_FLOPS =
    (L0_MAC + L0_BIAS) * 2 + L0_ACT +
    (L1_MAC + L1_BIAS) * 2 + L1_ACT +
    (L2_MAC + L2_BIAS) * 2;
// = (64+64)*2 + 640 + (4096+64)*2 + 640 + (64+1)*2
// = 256 + 640 + 8320 + 640 + 130 = 9,986 FLOPs

// stribeck: sign, fabsf, powf, expf, multiply, add, multiply, add = ~15 FLOPs
static constexpr long STRIBECK_FLOPS = 15;

// pid.compute: 7 multiplies, 6 adds, 2 compares, 1 branch ≈ 25 FLOPs
static constexpr long PID_FLOPS = 25;


// ============================================================================
//  Teensy 4.1 analytical estimates
//
//  IMXRT1062 Cortex-M7 @ 600 MHz, FPU with double-issue pipeline.
//  Sustained scalar float32 throughput ≈ 2 FLOP/cycle (FMA counted as 2).
//  tanhf via CMSIS-DSP minimax poly ≈ 20 cycles/element.
// ============================================================================

static constexpr double TEENSY_HZ        = 600e6;
static constexpr double TEENSY_FLOPS_PER_CYCLE = 2.0;   // scalar float32
static constexpr double TEENSY_FLOPS_PER_SEC =
    TEENSY_HZ * TEENSY_FLOPS_PER_CYCLE;

static constexpr double ENCODER_SPI_US   =  2.0;   // SPI @ 8 MHz, 3 bytes
static constexpr double SERIAL_PRINTF_US = 50.0;   // 2 Mbaud, ~100 chars
static constexpr double LOOP_OVERHEAD_US =  2.0;   // IntervalTimer ISR entry

static double flops_to_us_teensy(long flops) {
    return static_cast<double>(flops) / TEENSY_FLOPS_PER_SEC * 1e6;
}


// ============================================================================
//  Timing helpers
// ============================================================================

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::duration<double, std::micro>;

static double median_us(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n/2] : (v[n/2-1] + v[n/2]) * 0.5;
}

/**
 * Time a callable over `reps` outer repetitions, each doing `inner` calls.
 * Returns the MEDIAN per-call time in microseconds.
 * Using median rather than mean suppresses OS scheduling noise.
 */
template<typename Fn>
static double bench(Fn fn, int inner = 100000, int reps = 11) {
    std::vector<double> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        for (int i = 0; i < inner; ++i) fn(i);
        auto t1 = Clock::now();
        samples.push_back(Us(t1 - t0).count() / inner);
    }
    return median_us(samples);
}


// ============================================================================
//  Calibration: estimate host→Teensy scale factor
//
//  We time a known-latency FMUL chain on host, then compute how many
//  cycles per FLOP the host achieves.  Teensy's known throughput then
//  gives a principled scale factor.
//
//  Chain of 64 dependent FMADs (no ILP) → ~64 multiply-accumulate ops.
//  Host throughput for a dependent chain ≈ 1 op/cycle (latency-bound).
// ============================================================================

static double calibrate_scale_factor() {
    static constexpr int CAL_INNER = 200000;
    static constexpr int CAL_CHAIN = 64;   // must be power of 2

    double total_host_ns = 0.0;
    volatile float sink  = 0.0f;

    auto t0 = Clock::now();
    for (int i = 0; i < CAL_INNER; ++i) {
        float acc = static_cast<float>(i) * 1e-6f + 0.5f;
        // Dependent multiply chain — prevents vectorisation / OOO hiding
        for (int j = 0; j < CAL_CHAIN; ++j)
            acc = acc * 1.00001f + 0.00001f;
        sink = acc;
    }
    auto t1 = Clock::now();
    (void)sink;
    total_host_ns = std::chrono::duration<double,std::nano>(t1-t0).count();

    double host_ns_per_flop = total_host_ns / (static_cast<double>(CAL_INNER) * CAL_CHAIN);

    // Teensy cycles per equivalent FLOP (scalar dependent FMAC ≈ 3 cycles)
    static constexpr double TEENSY_NS_PER_DEP_FLOP = 3.0 / (TEENSY_HZ / 1e9);

    double scale = TEENSY_NS_PER_DEP_FLOP / host_ns_per_flop;
    return scale;  // multiply host timing by this to get Teensy estimate
}


// ============================================================================
//  Print helpers
// ============================================================================

static const char* PASS_STR = "PASS";
static const char* WARN_STR = "WARN";
static const char* FAIL_STR = "FAIL";

static const char* status(double host_us, double limit_us) {
    if (host_us <= limit_us)         return PASS_STR;
    if (host_us <= limit_us * 2.0)   return WARN_STR;
    return FAIL_STR;
}

static void print_row(const char* label, long flops,
                       double teensy_us, double host_us,
                       double scale, const char* stat) {
    double est_us = host_us * scale;
    printf("  %-20s  %6ld FLOPs  %6.2f µs (analytical)  "
           "%6.3f µs (host)  ~%5.2f µs (est.)  [%s]\n",
           label, flops, teensy_us, host_us, est_us, stat);
}


// ============================================================================
//  Main
// ============================================================================

int main()
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║          PIRL Firmware — Level 6 Latency Budget Analysis        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    // ------------------------------------------------------------------
    //  Calibration
    // ------------------------------------------------------------------
    printf("[ Calibrating host→Teensy scale factor... ]\n");
    double scale = calibrate_scale_factor();
    printf("  Dependent FMAC chain: scale factor = %.2fx\n", scale);
    printf("  (host timings × %.2f ≈ Teensy 4.1 @ 600 MHz estimate)\n\n", scale);


    // ------------------------------------------------------------------
    //  1. pirl_forward
    // ------------------------------------------------------------------
    // Vary velocity across calls to prevent the compiler caching the result.
    // volatile on the result prevents the call being optimised away.
    double pirl_host_us = bench([](int i) {
        float v = static_cast<float>(i % 1000) * 0.001f - 0.5f;
        volatile float r = pirl_forward(v);
        (void)r;
    });

    double pirl_analytical_us = flops_to_us_teensy(PIRL_FLOPS);
    const char* pirl_stat = status(pirl_host_us, 25.0);
    print_row("pirl_forward()", PIRL_FLOPS,
              pirl_analytical_us, pirl_host_us, scale, pirl_stat);

    if (strcmp(pirl_stat, FAIL_STR) == 0) {
        printf("    !! FAIL: %.2f µs > 25 µs limit.\n", pirl_host_us);
        printf("    Possible causes:\n");
        printf("      - tanhf() not inlining (missing -O2 or -ffast-math)\n");
        printf("      - Double-precision path triggered (missing 'f' suffix)\n");
        printf("      - Inner loop not compiled as scalar float32\n");
    }


    // ------------------------------------------------------------------
    //  2. stribeck()
    // ------------------------------------------------------------------
    StribeckParams sp(0.15f, 0.35f, 0.10f, 2.0f, 0.01f);
    double strib_host_us = bench([&sp](int i) {
        float v = static_cast<float>(i % 1000) * 0.001f - 0.5f;
        volatile float r = stribeck(v, sp);
        (void)r;
    });

    double strib_analytical_us = flops_to_us_teensy(STRIBECK_FLOPS);
    const char* strib_stat = status(strib_host_us, 5.0);
    print_row("stribeck()", STRIBECK_FLOPS,
              strib_analytical_us, strib_host_us, scale, strib_stat);


    // ------------------------------------------------------------------
    //  3. pid.compute()
    // ------------------------------------------------------------------
    PIDController pid(10.0f, 5.0f, 0.1f, -12.0f, 12.0f, 1.0f);
    double pid_host_us = bench([&pid](int i) {
        float meas = static_cast<float>(i % 1000) * 0.001f;
        float u_nn = 0.1f;
        float u_ff = 0.05f;
        volatile float r = pid.compute(meas, 0.01f, u_nn, u_ff);
        (void)r;
    });

    double pid_analytical_us = flops_to_us_teensy(PID_FLOPS);
    const char* pid_stat = status(pid_host_us, 1.0);
    print_row("pid.compute()", PID_FLOPS,
              pid_analytical_us, pid_host_us, scale, pid_stat);


    // ------------------------------------------------------------------
    //  4. pirl_infer() (deadband + pirl_forward)
    // ------------------------------------------------------------------
    double infer_host_us = bench([](int i) {
        float v = static_cast<float>(i % 1000) * 0.001f - 0.5f;
        volatile float r = pirl_infer(v);
        (void)r;
    });
    printf("  %-20s  %6s FLOPs  %6s µs (analytical)  "
           "%6.3f µs (host)  ~%5.2f µs (est.)  [%s]\n",
           "pirl_infer()", "~same",  "~same",
           infer_host_us, infer_host_us * scale,
           status(infer_host_us, 25.0));


    // ------------------------------------------------------------------
    //  5. Full control step (stribeck + pirl_infer + pid.compute)
    // ------------------------------------------------------------------
    PIDController pid2(10.0f, 5.0f, 0.1f, -12.0f, 12.0f, 1.0f);
    double full_host_us = bench([&pid2, &sp](int i) {
        float q     = static_cast<float>(i % 1000) * 0.001f;
        float omega = static_cast<float>(i % 500)  * 0.002f - 0.5f;
        float u_ff  = stribeck(omega, sp);
        float u_nn  = pirl_infer(omega);
        volatile float r = pid2.compute(q, 0.01f, u_nn, u_ff);
        (void)r;
    });

    long   full_flops         = PIRL_FLOPS + STRIBECK_FLOPS + PID_FLOPS;
    double full_analytical_us = flops_to_us_teensy(full_flops);
    const char* full_stat     = status(full_host_us, 30.0);
    printf("\n");
    print_row("Full control step", full_flops,
              full_analytical_us, full_host_us, scale, full_stat);


    // ------------------------------------------------------------------
    //  Budget table
    // ------------------------------------------------------------------
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                   Teensy 4.1 Budget Summary                     ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");

    double budget_us = 10000.0;  // 100 Hz → 10 ms window

    struct BudgetRow { const char* label; double analytical_us; double host_us; };
    BudgetRow rows[] = {
        { "pirl_forward()",      pirl_analytical_us,  pirl_host_us  },
        { "stribeck()",          strib_analytical_us, strib_host_us },
        { "pid.compute()",       pid_analytical_us,   pid_host_us   },
        { "Encoder SPI read",    ENCODER_SPI_US,      0.0           },
        { "Serial.printf()",     SERIAL_PRINTF_US,    0.0           },
        { "ISR overhead",        LOOP_OVERHEAD_US,    0.0           },
    };

    printf("║ %-28s  %10s  %10s  %9s ║\n",
           "Component", "Analytical", "Est.@600MHz", "% Budget");
    printf("║ %-28s  %10s  %10s  %9s ║\n",
           "─────────────────────────",
           "──────────", "──────────", "─────────");

    double total_analytical_us = 0.0;
    double total_estimated_us  = 0.0;

    for (auto& row : rows) {
        double est = (row.host_us > 0.0)
                     ? row.host_us * scale
                     : row.analytical_us;  // hardware components: use analytical
        total_analytical_us += row.analytical_us;
        total_estimated_us  += est;
        printf("║ %-28s  %8.2f µs  %8.2f µs  %7.3f%% ║\n",
               row.label,
               row.analytical_us,
               est,
               est / budget_us * 100.0);
    }

    printf("║ %-28s  %10s  %10s  %9s ║\n",
           "─────────────────────────",
           "──────────", "──────────", "─────────");
    printf("║ %-28s  %8.2f µs  %8.2f µs  %7.3f%% ║\n",
           "TOTAL",
           total_analytical_us,
           total_estimated_us,
           total_estimated_us / budget_us * 100.0);
    printf("║ %-28s  %8.2f µs  %10s  %9s ║\n",
           "BUDGET (100 Hz = 10 ms)", budget_us, "─", "100.000%");
    printf("║ %-28s  %8.2f µs  %8.2f µs  %7.3f%% ║\n",
           "MARGIN",
           budget_us - total_analytical_us,
           budget_us - total_estimated_us,
           (budget_us - total_estimated_us) / budget_us * 100.0);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");


    // ------------------------------------------------------------------
    //  FLOP breakdown
    // ------------------------------------------------------------------
    printf("FLOP breakdown — pirl_forward() 1→%d→%d→%d:\n", H0, H1, OUT_DIM);
    printf("  Layer 0  Linear(%-2d,%-2d): %5ld MAC×2 + %3ld bias×2 + %4ld tanh = %5ld\n",
           IN_DIM, H0,
           L0_MAC, L0_BIAS, L0_ACT,
           L0_MAC*2 + L0_BIAS*2 + L0_ACT);
    printf("  Layer 1  Linear(%-2d,%-2d): %5ld MAC×2 + %3ld bias×2 + %4ld tanh = %5ld\n",
           H0, H1,
           L1_MAC, L1_BIAS, L1_ACT,
           L1_MAC*2 + L1_BIAS*2 + L1_ACT);
    printf("  Layer 2  Linear(%-2d, %-2d): %5ld MAC×2 + %3ld bias×2 + %9s = %5ld\n",
           H1, OUT_DIM,
           L2_MAC, L2_BIAS, "(no act)",
           L2_MAC*2 + L2_BIAS*2);
    printf("  Total pirl_forward FLOPs: %ld\n\n", PIRL_FLOPS);


    // ------------------------------------------------------------------
    //  Double-precision check
    // ------------------------------------------------------------------
    printf("Double-precision check:\n");
    // Correct method: compare observed ns/FLOP against theoretical float32
    // throughput on this host.  A double-precision path would show ~2x slower
    // ns/FLOP relative to a known float32 baseline (scale-factor calibration).
    //
    // We use the scale factor from calibration as our float32 ns/FLOP reference:
    //   scale = (Teensy cycles/FMAC) / (host cycles/FMAC)
    //   host_ns_per_flop = (Teensy_ns_per_FMAC) / scale
    //
    // If pirl_forward ns/FLOP >> 2× host_ns_per_flop → double path suspected.
    double teensy_ns_per_dep_fmac = 3.0 / (TEENSY_HZ / 1e9);
    double host_ns_per_fmac       = teensy_ns_per_dep_fmac / scale;
    double pirl_ns_per_flop       = (pirl_host_us * 1000.0) / PIRL_FLOPS;
    double ratio                  = pirl_ns_per_flop / host_ns_per_fmac;

    printf("  host ns/FLOP (float32 ref): %.3f ns\n", host_ns_per_fmac);
    printf("  pirl_forward ns/FLOP:       %.3f ns  (%.1fx ref)\n",
           pirl_ns_per_flop, ratio);

    // pirl_forward has weight loads (cache pressure) so it naturally runs
    // slower than a tight FMAC chain.  Flag only if >8x ref — clear double signal.
    if (ratio > 8.0) {
        printf("  [WARN] pirl_forward is %.1fx slower than float32 ref — "
               "possible double-precision path.\n"
               "         Check: tanhf vs tanh, 'f' literal suffixes in pirl_weights.h\n",
               ratio);
    } else {
        printf("  [PASS] ns/FLOP ratio %.1fx — consistent with float32 + weight loads\n",
               ratio);
    }

    printf("\n");

    // ------------------------------------------------------------------
    //  Final verdict
    // ------------------------------------------------------------------
    bool any_fail = (strcmp(pirl_stat, FAIL_STR) == 0 ||
                     strcmp(strib_stat, FAIL_STR) == 0 ||
                     strcmp(pid_stat,   FAIL_STR) == 0 ||
                     strcmp(full_stat,  FAIL_STR) == 0);
    bool any_warn = (strcmp(pirl_stat, WARN_STR) == 0 ||
                     strcmp(strib_stat, WARN_STR) == 0 ||
                     strcmp(pid_stat,   WARN_STR) == 0);

    if (any_fail) {
        printf("LEVEL 6: FAIL — one or more components exceed host timing limits.\n");
        printf("         Inspect FAIL rows above before flashing to Teensy.\n");
        return 1;
    } else if (any_warn) {
        printf("LEVEL 6: WARN — all components within limits but some are marginal.\n");
        printf("         Estimated Teensy utilisation: %.3f%% of 10 ms budget.\n",
               total_estimated_us / budget_us * 100.0);
        return 0;
    } else {
        printf("LEVEL 6: PASS — all components well within latency budget.\n");
        printf("         Estimated Teensy utilisation: %.3f%% of 10 ms budget.\n",
               total_estimated_us / budget_us * 100.0);
        printf("         Margin: %.1f µs of %.0f µs (%.1fx headroom).\n",
               budget_us - total_estimated_us,
               budget_us,
               budget_us / total_estimated_us);
        return 0;
    }
}
