import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sim.hybrid_controller import HybridPIRLController

# Since the script is in the root directory, standard package imports will work.
from sim.sim_env import SimpleArmEnv


def set_seed(seed=42):
    """Enforce strict determinism for rigorous reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    print("FastTrack Protocol: Executing Task 2.1 (Phase 1 Gate Check)...")
    set_seed(42)

    # 1. Initialize Environment & Controller
    # SimpleArmEnv handles dt internally (default 0.001)
    env = SimpleArmEnv()

    # Load the trained residual network robustly
    REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(REPO_ROOT, "notebooks", "pirl_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Train the model first."
        )

    # Initialize HybridPIRLController (it handles model loading internally)
    # Gain rationale: Stribeck system has F_c=0.5 Nm, F_s=1.0 Nm.
    # With Kp=1.5, overcoming F_c alone requires ~0.33 rad of error — too slow.
    # Kp=80 keeps error < 0.01 rad while fighting friction. Kd=3 damps the
    # ~200 rad/s bandwidth. Ki=10 eliminates viscous steady-state offset.
    controller = HybridPIRLController(kp=80, ki=10, kd=3, model_path=model_path)

    # 2. Define Fixed Test Trajectory (Sine wave)
    # Using 5000 steps since SimpleArmEnv default dt is 0.001 (5 seconds total)
    steps = 5000
    t = np.linspace(0, 5, steps)

    # Kinematic profiles for the controller's FF and PID inputs
    target_pos = np.sin(np.pi * t)
    target_vel = np.pi * np.cos(np.pi * t)
    target_accel = -(np.pi**2) * np.sin(np.pi * t)

    actual_pos = np.zeros(steps)

    # 3. Execution Loop
    reset_out = env.reset()
    # Handle both old Gym (obs) and new Gymnasium (obs, info) reset signatures
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    # Initial state sanitization
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state).flatten()

    for i in range(steps):
        # Calculate action using the full Hybrid controller API
        # compute() returns (u_total, u_pid, u_nn, u_ff) — unpack to get scalar torque
        action, u_pid, u_nn, u_ff = controller.compute(
            q=state[0],
            dq=state[1],
            target_q=target_pos[i],
            target_dq=target_vel[i],
            target_ddq=target_accel[i],
            dt=env.dt,
        )

        # Step environment
        step_out = env.step(action)

        # Handle Gym step signature: (obs, reward, done, info) or (obs, reward, term, trunc, info)
        state = step_out[0] if isinstance(step_out, tuple) else step_out

        # Enforcement: Sanitize leaked PyTorch tensors or nested arrays into flat 1D NumPy arrays
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.asarray(state).flatten()

        actual_pos[i] = state[0]

    # 4. Compute Rigor Metrics
    rmse = np.sqrt(np.mean((target_pos - actual_pos) ** 2))
    print(f"Validation RMSE: {rmse:.5f} rad")

    # 5. The Gate Assertion (Phase 1 Constraint)
    try:
        assert rmse < 0.05, f"GATE FAILED: RMSE {rmse:.5f} exceeds 0.05 rad threshold."
        print("GATE PASSED: Simulation error is within the Phase 1 boundary.")
    except AssertionError as e:
        print(f"\n[!] STRATEGIC HALT: {e}")
        print(
            "Do not proceed to Phase 2 (Hardware). Fix the Stribeck residual learning first."
        )
        raise

    # 6. Optional: Save Reproducibility Figure
    fig_dir = os.path.join(REPO_ROOT, "docs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(t, target_pos, label="Target Trajectory", linestyle="--")
    plt.plot(t, actual_pos, label="PIRL Hybrid Control", alpha=0.8)
    plt.title(f"Phase 1 Validation (RMSE: {rmse:.5f} rad)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.legend()
    plt.grid(True)
    fig_path = os.path.join(fig_dir, "Figure_1_Convergence_Reproduced.png")
    plt.savefig(fig_path)
    print(f"Reproduced figure saved to {fig_path}")


if __name__ == "__main__":
    main()
