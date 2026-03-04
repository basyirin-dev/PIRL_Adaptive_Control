import numpy as np
import pytest
import torch
from sim_env import SimpleArmEnv


def test_stiction_breakaway():
    """
    Phase 1 Gate Check:
    Verify that the simulated friction ('rust') strictly prevents movement
    below the breakaway torque threshold (F_s = 1.0 Nm).
    """
    env = SimpleArmEnv()
    env.reset()

    # 1. Apply torque below the stiction threshold
    # The system must NOT move.
    u_stick = torch.tensor([0.8], dtype=torch.float32)
    state_stick = env.step(u_stick)

    # Extract state if env.step returns a tuple (obs, reward, done, info)
    state_stick = state_stick[0] if isinstance(state_stick, tuple) else state_stick

    if isinstance(state_stick, torch.Tensor):
        state_stick = state_stick.detach().cpu().numpy()
    vel_stick = np.asarray(state_stick).flatten()[1]

    assert (
        abs(vel_stick) < 1e-3
    ), f"GATE FAILED: System moved with {u_stick.item()} Nm applied (velocity={vel_stick:.5f}). Stiction prior is broken."

    # 2. Apply torque above the stiction threshold
    # The system MUST break away and accelerate.
    u_slip = torch.tensor([1.5], dtype=torch.float32)
    state_slip = env.step(u_slip)

    # Extract state if env.step returns a tuple
    state_slip = state_slip[0] if isinstance(state_slip, tuple) else state_slip

    if isinstance(state_slip, torch.Tensor):
        state_slip = state_slip.detach().cpu().numpy()
    vel_slip = np.asarray(state_slip).flatten()[1]

    assert (
        abs(vel_slip) > 1e-3
    ), "GATE FAILED: System failed to break away under 1.5 Nm high torque."


def test_physics_determinism():
    """
    Phase 1 Gate Check:
    Ensures identical inputs yield identical physics states for reproducibility.
    """
    env1 = SimpleArmEnv()
    env2 = SimpleArmEnv()

    env1.reset()
    env2.reset()

    u = torch.tensor([1.2], dtype=torch.float32)
    state1 = env1.step(u)
    state2 = env2.step(u)

    # Extract states if env.step returns a tuple
    state1 = state1[0] if isinstance(state1, tuple) else state1
    state2 = state2[0] if isinstance(state2, tuple) else state2

    if isinstance(state1, torch.Tensor):
        state1 = state1.detach().cpu().numpy()
    if isinstance(state2, torch.Tensor):
        state2 = state2.detach().cpu().numpy()

    np.testing.assert_allclose(
        np.asarray(state1).flatten(),
        np.asarray(state2).flatten(),
        err_msg="Physics simulation is not deterministic.",
    )
