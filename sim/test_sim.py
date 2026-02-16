import torch
import matplotlib.pyplot as plt
from sim_env import StribeckSystem


def run_stiction_test():
    print(">>> Initializing Simulation Test Bench...")

    # 1. Setup
    sim = StribeckSystem(J=0.01, dt=0.001)
    sim.reset()

    # 2. Define Inputs
    # We ramp up torque slowly to find the "Breakaway Point" (Fs)
    # Fs is set to 1.0 Nm in sim_env.py
    torques = torch.linspace(0, 1.5, 1000)

    velocities = []
    applied_torques = []

    print(">>> Running Torque Ramp Test...")
    for u in torques:
        state = sim.step(u)
        velocities.append(state[1].item())
        applied_torques.append(u.item())

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(
        applied_torques,
        velocities,
        label="System Response",
        linewidth=2,
        color="orange",
    )

    # Draw the theoretical stiction line
    plt.axvline(
        x=1.0, color="red", linestyle="--", label="True Stiction Limit (1.0 Nm)"
    )

    plt.title("Stiction Test: Does the 'Rust' Hold?", fontsize=14)
    plt.xlabel("Applied Torque (Nm)", fontsize=12)
    plt.ylabel("Resulting Velocity (rad/s)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save
    save_path = "docs/figures/sim_test_week3.png"
    plt.savefig(save_path)
    print(f">>> Test Complete. Plot saved to: {save_path}")
    print(">>> INTERPRETATION: Velocity should stay at 0 until Torque > 1.0 Nm.")


if __name__ == "__main__":
    run_stiction_test()
