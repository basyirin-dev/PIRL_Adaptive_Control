"""
validate_firmware.py
Cross-validation: Python simulation vs C++ firmware binary.

Runs the PIRL control loop in both Python and C++ over an identical
deterministic input sequence, then asserts every control term matches
within float32 tolerance (1e-4).

Usage (run from firmware/ directory):
    python test/cross_validation/validate_firmware.py

    Optional flags:
        --binary    PATH  path to cross_val_binary   (default: ./cross_val_binary)
        --sim-root  PATH  path to Python sim root    (default: ../../)
        --steps     N     number of timesteps         (default: 500)
        --seed      S     numpy random seed           (default: 42)
        --disable-nn      force u_nn=0 on both sides (auto-detected if needed)
        --plot            show matplotlib plots
        --verbose         print per-step diff table for failing columns

Prerequisites:
    - cross_val_binary built (see build command in cross_val_driver.cpp)
    - numpy, pandas  (pip install numpy pandas)
    - matplotlib     (optional, for --plot)
"""

import argparse
import io
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# ============================================================================
#  Configuration — must match cross_val_driver.cpp exactly
# ============================================================================

KP = 10.0
KI = 5.0
KD = 0.1
V_MAX = 12.0
V_MIN = -12.0
SETPOINT = 1.0
DT = 0.01

FC, FS, VS, DELTA, SIGMA = 0.15, 0.35, 0.10, 2.0, 0.01
DEADBAND = 0.005


# ============================================================================
#  NN availability detection
# ============================================================================


def _probe_nn(sim_root: str) -> bool:
    """Return True if pirl_model.pirl_forward is importable."""
    try:
        sys.path.insert(0, sim_root)
        from pirl_model import pirl_forward  # type: ignore[import]

        _ = float(pirl_forward(0.1))
        return True
    except Exception:
        return False


def _pirl_forward_real(velocity: float, sim_root: str) -> float:
    sys.path.insert(0, sim_root)
    from pirl_model import pirl_forward  # type: ignore[import]

    return float(pirl_forward(velocity))


def make_pirl_infer(nn_available: bool, sim_root: str):
    """Return a pirl_infer callable that mirrors pirl_infer() in C++."""

    def _infer(velocity: float) -> float:
        if not nn_available:
            return 0.0
        if abs(velocity) > DEADBAND:
            return _pirl_forward_real(velocity, sim_root)
        return 0.0

    return _infer


# ============================================================================
#  Python reference implementations
# ============================================================================


class PIDController:
    """Direct Python mirror of pid.cpp — NN-aware anti-windup included."""

    def __init__(self, kp, ki, kd, v_min, v_max, setpoint, alpha=0.1):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.v_min, self.v_max = v_min, v_max
        self.setpoint = setpoint
        self.alpha = alpha
        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filt = 0.0
        self._first_call = True
        self._u_p = self._u_i = self._u_d = 0.0

    def compute(self, measurement, dt, u_nn=0.0, u_ff=0.0):
        if dt <= 0.0:
            return self._u_p + self._u_i + self._u_d
        error = self.setpoint - measurement
        u_p = self.kp * error
        if self._first_call:
            deriv_filt = 0.0
            self._first_call = False
        else:
            deriv_raw = (error - self._prev_error) / dt
            deriv_filt = self.alpha * deriv_raw + (1.0 - self.alpha) * self._deriv_filt
        self._deriv_filt = deriv_filt
        u_d = self.kd * deriv_filt
        u_i = self.ki * self._integral
        u_total_est = u_p + u_i + u_d + u_nn + u_ff
        saturated = u_total_est > self.v_max or u_total_est < self.v_min
        same_sign = (error > 0 and self._integral > 0) or (
            error < 0 and self._integral < 0
        )
        if not (saturated and same_sign):
            self._integral += error * dt
            u_i = self.ki * self._integral
        self._u_p, self._u_i, self._u_d = u_p, u_i, u_d
        self._prev_error = error
        return u_p + u_i + u_d

    @property
    def integral(self):
        return self._integral

    @property
    def deriv_filtered(self):
        return self._deriv_filt

    @property
    def u_p(self):
        return self._u_p

    @property
    def u_i(self):
        return self._u_i

    @property
    def u_d(self):
        return self._u_d


def stribeck_friction(v):
    sign_v = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
    abs_v = abs(v)
    exp_term = math.exp(-pow(abs_v / VS, DELTA))
    return sign_v * (FC + (FS - FC) * exp_term) + SIGMA * v


# ============================================================================
#  Python reference run
# ============================================================================


def run_python(
    q_traj: np.ndarray, dq_traj: np.ndarray, dt: float, pirl_infer
) -> pd.DataFrame:
    pid = PIDController(KP, KI, KD, V_MIN, V_MAX, SETPOINT)
    rows = []
    for step, (q, dq) in enumerate(zip(q_traj, dq_traj)):
        pid.setpoint = SETPOINT
        u_stribeck = stribeck_friction(float(dq))
        u_ff = u_stribeck
        u_nn = pirl_infer(float(dq))
        u_pid = pid.compute(float(q), dt, u_nn=u_nn, u_ff=u_ff)
        u_total = float(np.clip(u_pid + u_nn + u_ff, V_MIN, V_MAX))
        rows.append(
            {
                "step": step,
                "q": float(q),
                "dq": float(dq),
                "dt": dt,
                "setpoint": SETPOINT,
                "u_pid": u_pid,
                "u_p": pid.u_p,
                "u_i": pid.u_i,
                "u_d": pid.u_d,
                "u_stribeck": u_stribeck,
                "u_nn": u_nn,
                "u_ff": u_ff,
                "u_total": u_total,
                "integral": pid.integral,
                "deriv_filtered": pid.deriv_filtered,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
#  C++ binary run
# ============================================================================


def run_cpp(
    binary: str, q_traj: np.ndarray, dq_traj: np.ndarray, dt: float, disable_nn: bool
) -> pd.DataFrame:
    if not os.path.isfile(binary):
        raise FileNotFoundError(
            f"cross_val_binary not found at '{binary}'.\n"
            f"Build with:\n"
            f"  g++ -std=c++17 -O2 -I lib/PIRL -I lib/ArduinoStub \\\n"
            f"      test/cross_validation/cross_val_driver.cpp \\\n"
            f"      lib/PIRL/pid.cpp lib/PIRL/stribeck.cpp "
            f"lib/PIRL/pirl_inference.cpp \\\n"
            f"      -o cross_val_binary -lm"
        )
    cmd = [binary]
    if disable_nn:
        cmd.append("--disable-nn")

    input_str = "\n".join(
        f"{q:.8f} {dq:.8f} {dt:.8f}" for q, dq in zip(q_traj, dq_traj)
    )
    result = subprocess.run(cmd, input=input_str, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cross_val_binary exited {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )
    return pd.read_csv(io.StringIO(result.stdout))


# ============================================================================
#  Diff and report
# ============================================================================

COLUMNS_TO_DIFF = [
    "u_pid",
    "u_p",
    "u_i",
    "u_d",
    "u_stribeck",
    "u_nn",
    "u_ff",
    "u_total",
    "integral",
    "deriv_filtered",
]
TOLERANCE = 1e-4


def diff_report(
    py_df: pd.DataFrame, cpp_df: pd.DataFrame, verbose: bool = False
) -> bool:
    if len(py_df) != len(cpp_df):
        print(f"[FAIL] Row count mismatch: Python={len(py_df)}, C++={len(cpp_df)}")
        return False

    passed = True
    results = []
    for col in COLUMNS_TO_DIFF:
        if col not in py_df.columns or col not in cpp_df.columns:
            continue
        diff = np.abs(
            np.asarray(py_df[col], dtype=np.float64)
            - np.asarray(cpp_df[col], dtype=np.float64)
        )
        max_diff = diff.max()
        mean_diff = diff.mean()
        ok = max_diff < TOLERANCE
        if not ok:
            passed = False
        results.append((col, max_diff, mean_diff, "PASS" if ok else "FAIL"))

    col_w = max(len(r[0]) for r in results) + 2
    print(f"\n{'Column':<{col_w}}  {'Max |diff|':>12}  {'Mean |diff|':>12}  Status")
    print("-" * (col_w + 34))
    for col, max_d, mean_d, status in results:
        flag = "  <- FAIL" if status == "FAIL" else ""
        print(f"{col:<{col_w}}  {max_d:>12.2e}  {mean_d:>12.2e}  {status}{flag}")

    if verbose:
        for col, _, _, status in results:
            if status == "FAIL":
                diff = np.abs(
                    np.asarray(py_df[col], dtype=np.float64)
                    - np.asarray(cpp_df[col], dtype=np.float64)
                )
                first_bad = int(np.argmax(diff >= TOLERANCE))
                print(
                    f"  [{col}] first divergence at step {first_bad}: "
                    f"Python={py_df[col].iloc[first_bad]:.8f}  "
                    f"C++={cpp_df[col].iloc[first_bad]:.8f}  "
                    f"diff={diff[first_bad]:.2e}"
                )
    return passed


# ============================================================================
#  Optional plots
# ============================================================================


def plot_comparison(py_df: pd.DataFrame, cpp_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not available -- skipping plots")
        return

    steps = np.asarray(py_df["step"], dtype=np.int64)
    cols = ["u_pid", "u_total", "integral"]
    fig, axes = plt.subplots(len(cols), 2, figsize=(14, 3 * len(cols)))

    for row, col in enumerate(cols):
        if col not in py_df.columns or col not in cpp_df.columns:
            continue
        ax = axes[row][0]
        ax.plot(
            steps,
            np.asarray(py_df[col], dtype=np.float64),
            label="Python",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            np.asarray(cpp_df[col], dtype=np.float64),
            label="C++",
            linewidth=1.0,
            linestyle="--",
        )
        ax.set_title(col)
        ax.legend(fontsize=8)
        ax.set_xlabel("step")

        ax2 = axes[row][1]
        diff = np.abs(
            np.asarray(py_df[col], dtype=np.float64)
            - np.asarray(cpp_df[col], dtype=np.float64)
        )
        ax2.semilogy(steps, diff + 1e-12)
        ax2.axhline(
            TOLERANCE, color="red", linestyle="--", label=f"tol={TOLERANCE:.0e}"
        )
        ax2.set_title(f"|diff|  {col}")
        ax2.legend(fontsize=8)
        ax2.set_xlabel("step")

    plt.tight_layout()
    plt.savefig("cross_validation_result.png", dpi=120)
    print("[INFO] Plot saved to cross_validation_result.png")
    plt.show()


# ============================================================================
#  Entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate C++ firmware against Python simulation"
    )
    parser.add_argument("--binary", default="./cross_val_binary")
    parser.add_argument("--sim-root", default="../../")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-nn", action="store_true", help="Force u_nn=0 on both sides"
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    #  NN availability — auto-detect if not forced off
    # ------------------------------------------------------------------
    if args.disable_nn:
        nn_available = False
        print("[INFO] NN disabled via --disable-nn flag")
    else:
        nn_available = _probe_nn(args.sim_root)
        if not nn_available:
            nn_available = False
            print(
                "[WARN] pirl_model not importable from "
                f"'{args.sim_root}' — forcing --disable-nn on both sides."
            )
            print(
                "[WARN] Both Python and C++ will use u_nn=0. "
                "Cross-validation tests PID + Stribeck paths only."
            )
        else:
            print(f"[INFO] NN available — pirl_model loaded from '{args.sim_root}'")

    disable_nn = not nn_available
    pirl_infer = make_pirl_infer(nn_available, args.sim_root)

    # ------------------------------------------------------------------
    #  Generate deterministic input trajectory
    # ------------------------------------------------------------------
    np.random.seed(args.seed)
    N = args.steps
    q_traj = np.cumsum(np.random.randn(N) * 0.005).astype(np.float32)
    dq_traj = (np.diff(q_traj, prepend=q_traj[0]) / DT).astype(np.float32)

    print(f"[INFO] Running cross-validation: {N} steps, seed={args.seed}, " f"dt={DT}s")
    print(f"[INFO] Binary: {args.binary}" + (" + --disable-nn" if disable_nn else ""))

    # ------------------------------------------------------------------
    #  Python reference
    # ------------------------------------------------------------------
    print("[INFO] Running Python reference controller...")
    py_df = run_python(q_traj, dq_traj, DT, pirl_infer)
    print(
        f"[INFO] Python: {len(py_df)} rows, "
        f"u_total [{py_df['u_total'].min():.3f}, "
        f"{py_df['u_total'].max():.3f}] V, "
        f"u_nn sum={py_df['u_nn'].abs().sum():.4f}"
    )

    # ------------------------------------------------------------------
    #  C++ binary
    # ------------------------------------------------------------------
    print("[INFO] Running C++ firmware binary...")
    try:
        cpp_df = run_cpp(args.binary, q_traj, dq_traj, DT, disable_nn)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(
        f"[INFO] C++:    {len(cpp_df)} rows, "
        f"u_total [{cpp_df['u_total'].min():.3f}, "
        f"{cpp_df['u_total'].max():.3f}] V, "
        f"u_nn sum={cpp_df['u_nn'].abs().sum():.4f}"
    )

    # ------------------------------------------------------------------
    #  Diff
    # ------------------------------------------------------------------
    print(f"\n[INFO] Tolerance: {TOLERANCE:.0e}  (float32 cross-precision)\n")
    passed = diff_report(py_df, cpp_df, verbose=args.verbose)

    if args.plot:
        plot_comparison(py_df, cpp_df)

    # ------------------------------------------------------------------
    #  Final verdict
    # ------------------------------------------------------------------
    nn_note = "  [NN disabled — testing PID+Stribeck paths only]" if disable_nn else ""
    if passed:
        print("=" * 55)
        print("CROSS-VALIDATION PASSED — firmware matches Python simulation")
        if nn_note:
            print(nn_note)
        print("=" * 55)
        sys.exit(0)
    else:
        print("=" * 55)
        print("CROSS-VALIDATION FAILED — see table above")
        print("Common causes:")
        print("  1. Gains in cross_val_driver.cpp don't match this script")
        print("  2. Anti-windup logic differs between Python and C++")
        print("  3. Derivative filter alpha mismatch (default: 0.1)")
        if not disable_nn:
            print("  4. NN mismatch — try --disable-nn to isolate")
        print("=" * 55)
        sys.exit(1)


if __name__ == "__main__":
    main()
