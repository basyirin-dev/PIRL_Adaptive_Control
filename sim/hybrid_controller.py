import numpy as np
import torch


class HybridPIRLController:
    """
    Hybrid Controller: u_total = u_PID + u_NN(v) + u_FF(a)

    Key Features:
    1. PID: Handles linear errors and disturbances.
    2. NN: Predicts and cancels nonlinear friction (The "Rust").
    3. FF: Feedforward Inertia cancellation (J * target_accel).
    4. Deadband: Silences NN at zero-crossing to prevent chattering.
    """

    def __init__(self, kp, ki, kd, model_path, inertia=0.002, deadband=0.05):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.J = inertia  # System Inertia for FF
        self.deadband = deadband  # Velocity threshold (rad/s)

        # Integrator State
        self.integral_error = 0.0
        self.prev_error = 0.0

        # Load the Neural Network
        try:
            from sim.pirl_network import PIRLNetwork

            self.model = PIRLNetwork()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set to inference mode
            print(f"[HybridPIRLController] Model loaded from {model_path}")
        except Exception as e:
            print(f"[CRITICAL] Failed to load NN model: {e}")
            raise e

    def compute(self, q, dq, target_q, target_dq, target_ddq, dt):
        """
        Compute control output u (Nm).
        Args:
            q, dq: Current state
            target_q, target_dq: Reference state
            target_ddq: Reference acceleration (Required for FF)
            dt: Time step
        """
        # 1. Error Calculation
        error = target_q - q
        d_error = target_dq - dq
        self.integral_error += error * dt

        # 2. PID Term
        u_pid = (
            (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * d_error)
        )

        # 3. Feedforward Inertia Term (u_ff = J * a_ref)
        # Without this, the PID has to "drag" the mass, causing lag.
        u_ff = self.J * target_ddq

        # 4. Neural Residual Term (Friction Cancellation)
        # Constraint: Apply Deadband to prevent chattering at zero velocity
        if abs(dq) < self.deadband:
            u_nn = 0.0
        else:
            # Prepare tensor input
            v_tensor = torch.FloatTensor([[dq]])
            with torch.no_grad():
                u_nn = self.model(v_tensor).item()

        # 5. Total Control Law
        # Note: We ADD u_nn to help the motor overcome friction.
        # If the model predicts +1Nm friction, we push +1Nm to cancel it.
        u_total = u_pid + u_nn + u_ff

        return u_total, u_pid, u_nn, u_ff

    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
