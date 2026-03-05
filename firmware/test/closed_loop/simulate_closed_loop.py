"""
simulate_closed_loop.py
Level 5 — Integrated Closed-Loop Simulation.

Runs three simulation variants over the same DC motor plant:

  A. Python-only       — plant + Python controller (reference)
  B. Hybrid (C++ ctrl) — plant in Python, controller in C++ binary
  C. Baseline PID      — same Python plant, no u_ff / u_nn

Compares A vs B to validate firmware correctness.
Compares A vs C to quantify the PIRL architecture benefit.

Usage (run from firmware/ directory):
    python test/closed_loop/simulate_closed_loop.py

    Optional flags:
        --binary    PATH   path to closed_loop_binary (default: ./closed_loop_binary)
        --steps     N      simulation steps            (default: 1000)
        --dt        DT     timestep [s]                (default: 0.01)
        --setpoint  SP     position setpoint [rad]     (default: 1.0)
        --disable-nn       force u_nn=0 on both sides (auto-detected)
        --plot             show matplotlib plots
        --save-csv  PATH   save all trajectories to CSV

Prerequisites:
    - closed_loop_binary built:
        g++ -std=c++17 -O2 -I lib/PIRL -I lib/ArduinoStub \\
            test/closed_loop/closed_loop_driver.cpp \\
            lib/PIRL/pid.cpp lib/PIRL/stribeck.cpp lib/PIRL/pirl_inference.cpp \\
            -o closed_loop_binary -lm
    - numpy, pandas  (pip install numpy pandas)
    - matplotlib     (optional, for --plot)
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================================
#  Constants — must match closed_loop_driver.cpp
# ============================================================================

KP, KI, KD = 10.0, 5.0, 0.1
V_MAX, V_MIN = 12.0, -12.0
DERIV_ALPHA = 0.1

FC, FS, VS, STRIBECK_DELTA, SIGMA = 0.15, 0.35, 0.10, 2.0, 0.01
DEADBAND = 0.005

J_MOTOR = 0.001
B_MOTOR = 0.002

TRAJ_TOLERANCE = 1e-3


# ============================================================================
#  NN availability detection
# ============================================================================


def _probe_nn(sim_root: str) -> bool:
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
    def _infer(v: float) -> float:
        if not nn_available:
            return 0.0
        if abs(v) > DEADBAND:
            return _pirl_forward_real(v, sim_root)
        return 0.0

    return _infer


# ============================================================================
#  Python physics models
# ============================================================================


def stribeck_friction(v: float) -> float:
    sign_v = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
    abs_v = abs(v)
    exp_term = math.exp(-pow(abs_v / VS, STRIBECK_DELTA))
    return sign_v * (FC + (FS - FC) * exp_term) + SIGMA * v


def plant_step(omega: float, u: float, dt: float) -> float:
    """Euler integration: J*dω/dt = u - F_friction(ω) - B*ω"""
    domega = (u - stribeck_friction(omega) - B_MOTOR * omega) / J_MOTOR
    return omega + domega * dt


# ============================================================================
#  Python PID (mirror of pid.cpp — NN-aware anti-windup)
# ============================================================================


class PIDController:
    def __init__(self, kp, ki, kd, v_min, v_max, setpoint, alpha=DERIV_ALPHA):
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
        sat = (u_p + u_i + u_d + u_nn + u_ff) > self.v_max or (
            u_p + u_i + u_d + u_nn + u_ff
        ) < self.v_min
        ss = (error > 0 and self._integral > 0) or (error < 0 and self._integral < 0)
        if not (sat and ss):
            self._integral += error * dt
            u_i = self.ki * self._integral
        self._u_p, self._u_i, self._u_d = u_p, u_i, u_d
        self._prev_error = error
        return u_p + u_i + u_d

    def reset(self):
        self._integral = self._prev_error = self._deriv_filt = 0.0
        self._first_call = True
        self._u_p = self._u_i = self._u_d = 0.0


# ============================================================================
#  Trajectory container
# ============================================================================


@dataclass
class Trajectory:
    label: str
    steps: list[int] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    q: list[float] = field(default_factory=list)
    omega: list[float] = field(default_factory=list)
    u_total: list[float] = field(default_factory=list)
    u_pid: list[float] = field(default_factory=list)
    u_stribeck: list[float] = field(default_factory=list)
    u_nn: list[float] = field(default_factory=list)
    integral: list[float] = field(default_factory=list)
    error: list[float] = field(default_factory=list)

    def append(
        self, step, t, q, omega, u_total, u_pid, u_stribeck, u_nn, integral, setpoint
    ):
        self.steps.append(step)
        self.t.append(t)
        self.q.append(q)
        self.omega.append(omega)
        self.u_total.append(u_total)
        self.u_pid.append(u_pid)
        self.u_stribeck.append(u_stribeck)
        self.u_nn.append(u_nn)
        self.integral.append(integral)
        self.error.append(setpoint - q)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                k: getattr(self, k)
                for k in [
                    "steps",
                    "t",
                    "q",
                    "omega",
                    "u_total",
                    "u_pid",
                    "u_stribeck",
                    "u_nn",
                    "integral",
                    "error",
                ]
            }
        )

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(np.array(self.error) ** 2)))

    @property
    def settling_step(self) -> Optional[int]:
        errors = np.abs(self.error)
        for i in range(len(errors) - 50):
            if np.all(errors[i : i + 50] < 0.02):
                return self.steps[i]
        return None


# ============================================================================
#  Simulation A: Python-only reference
# ============================================================================


def simulate_python(steps, dt, setpoint, pirl_infer) -> Trajectory:
    traj = Trajectory(label="Python (reference)")
    pid = PIDController(KP, KI, KD, V_MIN, V_MAX, setpoint)
    omega, q = 0.0, 0.0
    for step in range(steps):
        pid.setpoint = setpoint
        u_stribeck = stribeck_friction(omega)
        u_ff = u_stribeck
        u_nn = pirl_infer(omega)
        u_pid = pid.compute(q, dt, u_nn=u_nn, u_ff=u_ff)
        u_total = float(np.clip(u_pid + u_nn + u_ff, V_MIN, V_MAX))
        traj.append(
            step,
            step * dt,
            q,
            omega,
            u_total,
            u_pid,
            u_stribeck,
            u_nn,
            pid._integral,
            setpoint,
        )
        omega = plant_step(omega, u_total, dt)
        q += omega * dt
    return traj


# ============================================================================
#  Simulation B: Hybrid — Python plant + C++ controller
# ============================================================================


def simulate_hybrid(binary, steps, dt, setpoint, disable_nn) -> Trajectory:
    if not os.path.isfile(binary):
        raise FileNotFoundError(
            f"closed_loop_binary not found at '{binary}'.\n"
            f"Build with:\n"
            f"  g++ -std=c++17 -O2 -I lib/PIRL -I lib/ArduinoStub \\\n"
            f"      test/closed_loop/closed_loop_driver.cpp \\\n"
            f"      lib/PIRL/pid.cpp lib/PIRL/stribeck.cpp "
            f"lib/PIRL/pirl_inference.cpp \\\n"
            f"      -o closed_loop_binary -lm"
        )

    proc = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Consume header
    header = proc.stdout.readline().strip()  # type: ignore[union-attr]
    if not header.startswith("u_total"):
        raise RuntimeError(f"Unexpected header: '{header}'")

    # Disable NN in C++ if needed — send command, wait for ACK
    if disable_nn:
        proc.stdin.write("DISABLE_NN\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
        ack = proc.stdout.readline().strip()  # type: ignore[union-attr]
        if ack != "DISABLE_NN_OK":
            raise RuntimeError(
                f"Expected DISABLE_NN_OK, got '{ack}'. "
                f"Rebuild closed_loop_binary from updated source."
            )

    traj = Trajectory(label="Hybrid (C++ controller)")
    omega, q = 0.0, 0.0
    cols = ["u_total", "u_pid", "u_p", "u_i", "u_d", "u_stribeck", "u_nn", "integral"]

    try:
        for step in range(steps):
            proc.stdin.write(  # type: ignore[union-attr]
                f"{q:.8f} {omega:.8f} {dt:.8f} {setpoint:.8f}\n"
            )
            proc.stdin.flush()  # type: ignore[union-attr]

            response = proc.stdout.readline().strip()  # type: ignore[union-attr]
            if response in ("ERROR", ""):
                raise RuntimeError(
                    f"C++ binary error at step {step}: q={q:.4f} omega={omega:.4f}"
                )

            row = dict(zip(cols, [float(x) for x in response.split(",")]))
            traj.append(
                step,
                step * dt,
                q,
                omega,
                row["u_total"],
                row["u_pid"],
                row["u_stribeck"],
                row["u_nn"],
                row["integral"],
                setpoint,
            )

            omega = plant_step(omega, row["u_total"], dt)
            q += omega * dt
    finally:
        proc.stdin.write("QUIT\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
        proc.wait(timeout=5)

    return traj


# ============================================================================
#  Simulation C: Baseline PID (no ff, no NN)
# ============================================================================


def simulate_baseline_pid(steps, dt, setpoint) -> Trajectory:
    traj = Trajectory(label="Baseline PID (no ff, no NN)")
    pid = PIDController(KP, KI, KD, V_MIN, V_MAX, setpoint)
    omega, q = 0.0, 0.0
    for step in range(steps):
        pid.setpoint = setpoint
        u_pid = pid.compute(q, dt)
        u_total = float(np.clip(u_pid, V_MIN, V_MAX))
        traj.append(
            step, step * dt, q, omega, u_total, u_pid, 0.0, 0.0, pid._integral, setpoint
        )
        omega = plant_step(omega, u_total, dt)
        q += omega * dt
    return traj


# ============================================================================
#  Comparison report
# ============================================================================


def compare_trajectories(ref: Trajectory, hyb: Trajectory) -> bool:
    columns = ["q", "omega", "u_total", "u_pid", "u_stribeck", "u_nn"]
    passed = True
    results = []
    for col in columns:
        py_arr = np.array(getattr(ref, col))
        cpp_arr = np.array(getattr(hyb, col))
        diff = np.abs(py_arr - cpp_arr)
        ok = diff.max() < TRAJ_TOLERANCE
        if not ok:
            passed = False
        results.append((col, diff.max(), diff.mean(), "PASS" if ok else "FAIL"))

    col_w = max(len(r[0]) for r in results) + 2
    print(f"\n{'Column':<{col_w}}  {'Max |diff|':>12}  {'Mean |diff|':>12}  Status")
    print("-" * (col_w + 34))
    for col, max_d, mean_d, status in results:
        flag = "  <- FAIL" if status == "FAIL" else ""
        print(f"{col:<{col_w}}  {max_d:>12.2e}  {mean_d:>12.2e}  {status}{flag}")
    return passed


def print_metrics(trajectories: list[Trajectory]) -> None:
    print(f"\n{'Simulation':<38}  {'RMSE [rad]':>12}  {'Settling step':>14}")
    print("-" * 69)
    for traj in trajectories:
        s = str(traj.settling_step) if traj.settling_step is not None else "not settled"
        print(f"{traj.label:<38}  {traj.rmse:>12.6f}  {s:>14}")


# ============================================================================
#  Plots
# ============================================================================


def plot_results(ref, hyb, baseline, setpoint) -> None:
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not available -- skipping plots")
        return

    t = np.array(ref.t)
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("PIRL Closed-Loop Simulation — Level 5 Validation", fontsize=13)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, :])
    ax.axhline(
        setpoint,
        color="k",
        linestyle="--",
        linewidth=0.8,
        label=f"Setpoint ({setpoint} rad)",
    )
    ax.plot(t, ref.q, label=ref.label, linewidth=2.0)
    ax.plot(t, hyb.q, label=hyb.label, linewidth=1.5, linestyle="--")
    ax.plot(
        t, baseline.q, label=baseline.label, linewidth=1.2, linestyle=":", alpha=0.8
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [rad]")
    ax.set_title("Position Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, np.abs(ref.error), label=ref.label, linewidth=1.5)
    ax2.plot(
        t,
        np.abs(baseline.error),
        label=baseline.label,
        linewidth=1.2,
        linestyle=":",
        alpha=0.8,
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("|Error| [rad]")
    ax2.set_title("Tracking Error")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, ref.u_pid, label="u_pid", linewidth=1.2)
    ax3.plot(t, ref.u_stribeck, label="u_stribeck", linewidth=1.2, linestyle="--")
    ax3.plot(t, ref.u_nn, label="u_nn", linewidth=1.2, linestyle=":")
    ax3.plot(t, ref.u_total, label="u_total", linewidth=1.5, color="k", alpha=0.7)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Voltage [V]")
    ax3.set_title("Control Terms (Python ref)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    diff_q = np.abs(np.array(ref.q) - np.array(hyb.q))
    ax4.semilogy(t, diff_q + 1e-12)
    ax4.axhline(
        TRAJ_TOLERANCE, color="red", linestyle="--", label=f"tol={TRAJ_TOLERANCE:.0e}"
    )
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("|Δq| [rad]")
    ax4.set_title("Python vs C++ Position Divergence")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(t, ref.integral, label=ref.label, linewidth=1.5)
    ax5.plot(t, hyb.integral, label=hyb.label, linewidth=1.2, linestyle="--")
    ax5.plot(
        t,
        baseline.integral,
        label=baseline.label,
        linewidth=1.2,
        linestyle=":",
        alpha=0.8,
    )
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Integral state")
    ax5.set_title("Integrator State")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.savefig("closed_loop_simulation.png", dpi=140, bbox_inches="tight")
    print("[INFO] Plot saved to closed_loop_simulation.png")
    plt.show()


# ============================================================================
#  Entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Level 5 -- Integrated closed-loop simulation"
    )
    parser.add_argument("--binary", default="./closed_loop_binary")
    parser.add_argument("--sim-root", default="../../")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--setpoint", type=float, default=1.0)
    parser.add_argument(
        "--disable-nn", action="store_true", help="Force u_nn=0 on both sides"
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-csv", metavar="PATH", default=None)
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
            print(
                f"[WARN] pirl_model not importable from '{args.sim_root}' "
                f"-- forcing u_nn=0 on both sides."
            )
            print("[WARN] Cross-validation tests PID + Stribeck paths only.")
        else:
            print(f"[INFO] NN available -- loaded from '{args.sim_root}'")

    disable_nn = not nn_available
    pirl_infer = make_pirl_infer(nn_available, args.sim_root)

    print(
        f"[INFO] Steps={args.steps}, dt={args.dt}s, "
        f"setpoint={args.setpoint} rad, T={args.steps * args.dt:.1f}s"
        + (" [NN disabled]" if disable_nn else "")
    )

    # ------------------------------------------------------------------
    #  Run simulations
    # ------------------------------------------------------------------
    print("\n[1/3] Python reference simulation...")
    ref = simulate_python(args.steps, args.dt, args.setpoint, pirl_infer)
    print(
        f"      RMSE={ref.rmse:.6f} rad  "
        f"settling={ref.settling_step or 'not settled'}"
    )

    print("[2/3] Hybrid simulation (Python plant + C++ controller)...")
    try:
        hyb = simulate_hybrid(
            args.binary, args.steps, args.dt, args.setpoint, disable_nn
        )
        print(
            f"      RMSE={hyb.rmse:.6f} rad  "
            f"settling={hyb.settling_step or 'not settled'}"
        )
        hybrid_ok = True
    except FileNotFoundError as e:
        print(f"      [SKIP] {e}")
        hyb = None
        hybrid_ok = False

    print("[3/3] Baseline PID simulation (no ff, no NN)...")
    baseline = simulate_baseline_pid(args.steps, args.dt, args.setpoint)
    print(
        f"      RMSE={baseline.rmse:.6f} rad  "
        f"settling={baseline.settling_step or 'not settled'}"
    )

    # ------------------------------------------------------------------
    #  Metrics + comparison
    # ------------------------------------------------------------------
    trajs = [ref] + ([hyb] if hyb else []) + [baseline]
    print_metrics(trajs)  # type: ignore[arg-type]

    if hyb is not None:
        print(f"\n[INFO] Cross-checking trajectories (tol={TRAJ_TOLERANCE:.0e}):")
        passed = compare_trajectories(ref, hyb)
    else:
        passed = None

    # ------------------------------------------------------------------
    #  PIRL vs baseline benefit
    # ------------------------------------------------------------------
    if baseline.rmse > 0:
        improvement = (baseline.rmse - ref.rmse) / baseline.rmse * 100
        print(f"\n[PIRL benefit] RMSE vs baseline: {improvement:+.1f}%")
        sp_r, sp_b = ref.settling_step, baseline.settling_step
        if sp_r and sp_b:
            print(
                f"[PIRL benefit] Settling speedup: {sp_b/sp_r:.1f}x "
                f"({sp_r} vs {sp_b} steps)"
            )

    # ------------------------------------------------------------------
    #  Optional CSV
    # ------------------------------------------------------------------
    if args.save_csv:
        dfs = [traj.to_dataframe().assign(simulation=traj.label) for traj in trajs]
        pd.concat(dfs, ignore_index=True).to_csv(args.save_csv, index=False)
        print(f"[INFO] Trajectories saved to {args.save_csv}")

    # ------------------------------------------------------------------
    #  Plots
    # ------------------------------------------------------------------
    if args.plot and hyb is not None:
        plot_results(ref, hyb, baseline, args.setpoint)

    # ------------------------------------------------------------------
    #  Final verdict
    # ------------------------------------------------------------------
    nn_note = "  [NN disabled -- PID+Stribeck paths only]" if disable_nn else ""
    print()
    if passed is True:
        print("=" * 55)
        print("LEVEL 5 PASSED -- firmware matches Python simulation")
        if nn_note:
            print(nn_note)
        print("=" * 55)
        sys.exit(0)
    elif passed is False:
        print("=" * 55)
        print("LEVEL 5 FAILED -- trajectory divergence exceeds tolerance")
        print("Check: integration order, anti-windup, gains, plant params")
        if not disable_nn:
            print("Try --disable-nn to isolate NN contribution")
        print("=" * 55)
        sys.exit(1)
    else:
        print("=" * 55)
        print("LEVEL 5 PARTIAL -- Python ran, C++ binary skipped")
        print("=" * 55)
        sys.exit(0)


if __name__ == "__main__":
    main()
