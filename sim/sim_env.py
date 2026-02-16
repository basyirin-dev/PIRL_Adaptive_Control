import torch
import torch.nn as nn


class StribeckSystem(nn.Module):
    """
    A Differentiable Physics Simulator for a 1-DoF Joint with Stribeck Friction.

    Equations:
        q_dot = v
        v_dot = (1/J) * (tau - F_friction(v, tau) - tau_ext)
    """

    # Explicit Type Hints
    F_c: torch.Tensor
    F_s: torch.Tensor
    v_s: torch.Tensor
    delta: torch.Tensor
    sigma: torch.Tensor
    J: torch.Tensor

    def __init__(
        self,
        J=0.002,  # Inertia (kg*m^2)
        dt=0.001,  # Time step (1ms)
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.dt = dt

        # System Parameters
        self.J = torch.tensor(J, device=device)

        # Friction Parameters (The "Rust")
        self.register_buffer("F_c", torch.tensor(0.5, device=device))  # Coulomb
        self.register_buffer(
            "F_s", torch.tensor(1.0, device=device)
        )  # Stiction (1.0 Nm)
        self.register_buffer(
            "v_s", torch.tensor(0.1, device=device)
        )  # Stribeck Velocity
        self.register_buffer("delta", torch.tensor(2.0, device=device))  # Shape
        self.register_buffer("sigma", torch.tensor(0.01, device=device))  # Viscous

        # State: [position, velocity]
        self.state = torch.zeros(2, device=device)

    def reset(self, init_state=None):
        """Resets the system to zero or a specific state."""
        if init_state is None:
            self.state = torch.zeros(2, device=self.device)
        else:
            self.state = init_state.to(self.device)
        return self.state

    def _compute_friction(self, velocity, torque):
        """
        Hybrid Friction Model:
        1. Static Regime (|v| ~ 0): Friction opposes torque exactly up to F_s.
        2. Kinetic Regime (|v| > 0): Stribeck Curve.
        """
        # --- Kinetic Friction (Stribeck) ---
        speed = torch.abs(velocity)

        # We use a tanh approximation for direction to keep gradients smooth-ish during sliding,
        # but the Static Regime logic below handles the "stuck" case.
        direction = torch.tanh(100 * velocity)

        stribeck = self.F_c + (self.F_s - self.F_c) * torch.exp(
            -torch.pow(speed / self.v_s, self.delta)
        )
        f_kinetic = (stribeck * direction) + (self.sigma * velocity)

        # --- Static Friction (Constraint) ---
        # If not moving, friction cancels the applied torque up to the static limit.
        f_static = torch.clamp(torque, -self.F_s, self.F_s)

        # --- The Switch ---
        # If velocity is effectively zero, we use the static model.
        # Threshold 1e-4 rad/s is small enough to be "stopped" but large enough for numerical stability.
        is_static = torch.abs(velocity) < 1e-4

        f_total = torch.where(is_static, f_static, f_kinetic)
        return f_total

    def dynamics(self, state, u):
        """
        Computes the derivative of the state (q_dot, v_dot).
        """
        _, v = state[0], state[1]

        # Compute Friction (Now depends on u for the static constraint)
        friction = self._compute_friction(v, u)

        # Equation of Motion: J*v_dot = u - friction
        v_dot = (u - friction) / self.J
        q_dot = v

        return torch.stack([q_dot, v_dot])

    def step(self, u):
        """
        Steps the simulation forward by dt using Semi-Implicit Euler.
        u: Control input (Torque, Nm)
        """
        if not isinstance(u, torch.Tensor):
            u = torch.tensor([u], device=self.device)

        # 1. Compute Dynamics
        _, v_dot = self.dynamics(self.state, u)

        # 2. Integration (Semi-Implicit Euler)
        new_v = self.state[1] + v_dot * self.dt
        new_q = self.state[0] + new_v * self.dt

        # 3. Update State
        self.state = torch.stack([new_q, new_v])

        return self.state
