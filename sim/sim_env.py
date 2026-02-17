import torch
import torch.nn as nn
import numpy as np


class StribeckSystem(nn.Module):
    """
    A Differentiable Physics Simulator for a 1-DoF Joint with Stribeck Friction.

    Physics Model:
        J * v_dot = u - F_friction(v, u)
    """

    def __init__(self, J=0.002, dt=0.001, device="cpu"):
        super().__init__()
        self.device = device
        self.dt = dt

        # Physics Constants
        self.J = torch.tensor(J, device=device)
        self.F_c = torch.tensor(0.5, device=device)  # Coulomb
        self.F_s = torch.tensor(1.0, device=device)  # Stiction (Breakaway)
        self.v_s = torch.tensor(0.1, device=device)  # Stribeck Velocity Scale
        self.delta = torch.tensor(2.0, device=device)  # Shape factor
        self.sigma = torch.tensor(0.01, device=device)  # Viscous Damping

        # State: [position (rad), velocity (rad/s)]
        self.state = torch.zeros(2, device=device)

    def _compute_friction(self, v, torque):
        """
        Calculates friction torque based on velocity and applied torque.
        """
        # 1. Kinetic Regime (Stribeck Curve)
        stribeck_term = self.F_c + (self.F_s - self.F_c) * torch.exp(
            -torch.pow(torch.abs(v) / self.v_s, self.delta)
        )
        f_kinetic = torch.sign(v) * stribeck_term + (self.sigma * v)

        # 2. Static Regime (Stiction)
        f_static = torch.clamp(torque, -self.F_s, self.F_s)

        # Soft Switch
        is_static = torch.abs(v) < 1e-4

        f_total = torch.where(is_static, f_static, f_kinetic)
        return f_total

    def forward(self, u):
        """
        Steps the physics forward by one dt.
        """
        q, v = self.state

        # Ensure u is a tensor
        if not isinstance(u, torch.Tensor):
            # FIX: Create a 0-D tensor (scalar) if input is float
            # This prevents broadcasting the state to (N, 1)
            u = torch.tensor(u, device=self.device)

        # Safety: If u came in as [0.5], squeeze it to scalar 0.5
        if u.ndim > 0:
            u = u.squeeze()

        # Calculate Friction
        f_fric = self._compute_friction(v, u)

        # Dynamics: J*a = u - f_fric
        acc = (u - f_fric) / self.J

        # Integration (Semi-Implicit Euler)
        v_new = v + acc * self.dt
        q_new = q + v_new * self.dt

        self.state = torch.stack([q_new, v_new])
        return self.state


class SimpleArmEnv:
    """
    The Gym-Like Wrapper.
    Handles Numpy <-> Torch conversion.
    """

    def __init__(self):
        self.sys = StribeckSystem()
        self.dt = self.sys.dt

    def reset(self):
        """Resets the arm to zero state."""
        self.sys.state = torch.zeros(2)
        return self.sys.state.cpu().numpy().flatten()

    def step(self, u):
        """
        Apply torque u (Nm) and step simulation.
        """
        # 1. Input Sanitization
        if isinstance(u, (float, np.floating)):
            u_val = float(u)
            u_tensor = torch.tensor(u_val)  # Scalar tensor
        elif isinstance(u, np.ndarray):
            u_tensor = torch.tensor(float(u.flatten()[0]))
        else:
            u_tensor = u

        # 2. Physics Step
        state_tensor = self.sys.forward(u_tensor)

        # 3. Output Conversion
        # FIX: Explicitly flatten to guarantee shape (2,)
        state_np = state_tensor.detach().cpu().numpy().flatten()

        return state_np, 0.0, False, {}
