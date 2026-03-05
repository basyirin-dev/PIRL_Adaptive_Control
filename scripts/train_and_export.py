"""
deploy/train_and_export.py
==========================
Trains PIRL_NN on the symmetric-harvest dataset and saves the best
checkpoint to 'model_best.pt', then immediately calls export_onnx.py.

Run this ONCE before export_onnx.py:

    python deploy/train_and_export.py           # train + export
    python deploy/train_and_export.py --no-export   # train only
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.ablation_runner import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    EST_DAMPING,
    EST_MASS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    PIRL_NN,
    load_data,
)

DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "model_best.pt")


def train_and_save(weights_path: str) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_data()

    # Residual target: strip out the physics prior
    u_physics = (EST_MASS * df["ddq_ref"]) + (EST_DAMPING * df["dq"])
    u_res = df["u_measured"] - u_physics

    X = torch.tensor(df[["dq"]].values, dtype=torch.float32)
    y = torch.tensor(u_res.values, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    model = PIRL_NN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    print(f"Training PIRL_NN for {EPOCHS} epochs …")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # Save whenever we beat the running best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), weights_path)

        if epoch % 20 == 0:
            marker = " ← saved" if avg_loss == best_loss else ""
            print(f"  Epoch {epoch:>4d}  loss={avg_loss:.6f}{marker}")

    print(f"\n[✓] Training complete. Best loss: {best_loss:.6f}")
    print(f"[✓] Weights saved → {weights_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument(
        "--no-export", action="store_true", help="Skip ONNX export after training."
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run numeric parity check during export.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_and_save(args.weights)

    if not args.no_export:
        # Re-use the export script directly
        export_script = os.path.join(PROJECT_ROOT, "deploy", "export_onnx.py")
        validate_flag = "--validate" if args.validate else ""
        cmd = (
            f"{sys.executable} {export_script} --weights {args.weights} {validate_flag}"
        )
        print(f"\n[→] Launching: {cmd}\n")
        os.system(cmd)
