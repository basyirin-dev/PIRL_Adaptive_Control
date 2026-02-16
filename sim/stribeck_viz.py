import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# Output directory for the figure
OUTPUT_DIR = "docs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def stribeck_friction(velocity, F_c, F_s, v_s, delta, sigma):
    """
    Computes Stribeck friction based on velocity.

    Args:
        velocity (torch.Tensor): Joint velocity (rad/s).
        F_c (float): Coulomb friction (dynamic).
        F_s (float): Static friction (stiction).
        v_s (float): Stribeck velocity threshold.
        delta (float): Exponent (shape factor).
        sigma (float): Viscous friction coefficient.

    Returns:
        torch.Tensor: Friction torque.
    """
    # Equation: F_f(v) = F_c + (F_s - F_c) * exp(-|v/v_s|^delta) + sigma * v
    # Note: We create a sign mask to handle direction, but the Stribeck magnitude
    # is usually applied opposing motion.

    speed = torch.abs(velocity)
    direction = torch.sign(velocity)

    # The Stribeck Effect (Exponential decay from Static to Coulomb)
    stribeck_term = F_c + (F_s - F_c) * torch.exp(-torch.pow(speed / v_s, delta))

    # Viscous Term
    viscous_term = sigma * speed

    # Total Friction Magnitude
    friction_magnitude = stribeck_term + viscous_term

    # Friction opposes motion
    return friction_magnitude * direction


def main():
    print(">>> Generating Stribeck Curve Visualization...")

    # 1. Define Parameters (Teensy Motor Estimates)
    # These are "guess" values we will try to learn later.
    F_c = 0.5  # Coulomb Friction (Nm)
    F_s = 1.2  # Static Friction (Nm) - High stiction!
    v_s = 0.1  # Stribeck Velocity (rad/s)
    delta = 2.0  # Exponent
    sigma = 0.1  # Viscous Damping

    # 2. Generate Velocity Vector (-2.0 to 2.0 rad/s)
    velocity = torch.linspace(-2.0, 2.0, 1000)

    # 3. Compute Friction
    friction = stribeck_friction(velocity, F_c, F_s, v_s, delta, sigma)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        velocity.numpy(),
        friction.numpy(),
        label="Total Friction Model",
        color="blue",
        linewidth=2,
    )

    # Annotate critical points
    plt.axhline(
        y=F_c, color="green", linestyle="--", alpha=0.5, label="Coulomb Level (Fc)"
    )
    plt.axhline(
        y=F_s, color="red", linestyle="--", alpha=0.5, label="Static Limit (Fs)"
    )
    plt.axhline(y=-F_c, color="green", linestyle="--", alpha=0.5)
    plt.axhline(y=-F_s, color="red", linestyle="--", alpha=0.5)

    plt.title(f"Stribeck Friction Model\n(Fs={F_s}, Fc={F_c}, vs={v_s})", fontsize=14)
    plt.xlabel("Velocity (rad/s)", fontsize=12)
    plt.ylabel("Friction Torque (Nm)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # 5. Save
    filepath = os.path.join(OUTPUT_DIR, "stribeck_curve_week2.png")
    plt.savefig(filepath, dpi=300)
    print(f">>> Figure saved to: {filepath}")
    print(">>> Week 2 Visualization Task Complete.")


if __name__ == "__main__":
    main()
