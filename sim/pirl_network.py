import torch
import torch.nn as nn


class PIRLNetwork(nn.Module):
    """
    A lightweight MLP to approximate the Stribeck Friction Curve.
    Architecture: 1 (Velocity) -> 32 -> 32 -> 1 (Friction Torque)

    Constraints:
    - Small size for <10ms inference on embedded hardware.
    - Tanh activation for smooth derivatives (essential for physics compliance).
    """

    def __init__(self):
        super(PIRLNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
        )

        # Initialize weights using Xavier for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, velocity):
        """
        Input: velocity (rad/s) [Batch, 1]
        Output: Friction Torque Estimate (Nm) [Batch, 1]
        """
        return self.net(velocity)
