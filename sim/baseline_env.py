import numpy as np
import torch
from sim.sim_env import StribeckSystem


class SimpleJointSim:
    """
    Gym-style wrapper for the StribeckSystem.
    (Preserved from Week 5 for backward compatibility)
    """

    def __init__(self, dt=0.01, friction_model="stribeck", device="cpu"):
        self.dt = dt
        self.device = device

        # Initialize the Physics Core
        self.system = StribeckSystem(dt=dt, device=device)

        # Internal state buffer [position, velocity]
        self.state = np.array([0.0, 0.0])

        # Force a reset on init
        self.reset()

    def reset(self):
        """Resets the system to zero state."""
        self.state = np.array([0.0, 0.0])

        # Directly manipulate the internal tensor state of the physics engine
        if hasattr(self.system, "state"):
            self.system.state = torch.tensor([0.0, 0.0], device=self.device)

        return self.state

    def step(self, u):
        """
        Steps the simulation forward.
        """
        # 1. INPUT: Numpy -> Torch
        if isinstance(u, np.ndarray):
            u_val = float(u.flatten()[0])
        elif isinstance(u, torch.Tensor):
            u_val = float(u.item())
        else:
            u_val = float(u)

        u_tensor = torch.tensor(u_val, dtype=torch.float32, device=self.device)

        # 2. PHYSICS: Step the differentiable engine
        next_state_tensor = self.system.forward(u_tensor)

        # 3. OUTPUT: Torch -> Numpy
        self.state = next_state_tensor.cpu().detach().numpy().flatten()
        return self.state, 0.0, False, {}


class PIDController:
    """
    Standard PID Controller.
    Used as the Baseline and the Linear component of the Hybrid Controller.
    """

    def __init__(self, kp, ki, kd, output_limits=None):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.output_limits = output_limits

        self.integral_error = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0

    def compute(self, q, dq, target_q, target_dq, dt):
        """
        Compute control signal u.
        Args:
            q: Current position
            dq: Current velocity
            target_q: Target position
            target_dq: Target velocity
            dt: Time step
        """
        # Error terms
        error = target_q - q
        d_error = target_dq - dq

        # Integration
        self.integral_error += error * dt

        # PID Law
        u = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * d_error)

        # Saturation (Optional)
        if self.output_limits is not None:
            u = np.clip(u, self.output_limits[0], self.output_limits[1])

        return u
