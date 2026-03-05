"""
deploy/export_onnx.py
=====================
Exports the trained PIRL_NN residual-friction model from PyTorch (.pt)
to ONNX format for embedded / cross-platform deployment.

Usage
-----
    # From the project root:
    python deploy/export_onnx.py                          # uses defaults
    python deploy/export_onnx.py --weights model_best.pt  # custom weight path
    python deploy/export_onnx.py --validate               # run numeric check

Architecture recap
------------------
    PIRL_NN: velocity (1,) -> [Linear(1,H) -> Tanh -> Linear(H,H) -> Tanh -> Linear(H,1)]
    Input  : 'velocity'              -- shape [batch, 1]  (rad/s)
    Output : 'friction_compensation' -- shape [batch, 1]  (Nm)

Opset 11 is chosen for maximum compatibility with
onnxruntime >= 1.8 and TensorRT >= 8.0.

NOTE: dynamo=False forces the legacy TorchScript exporter.
PyTorch 2.x defaults to the dynamo exporter which requires the
optional 'onnxscript' package. The TorchScript path is stable,
has no extra dependencies, and fully supports this simple MLP.
"""

import argparse
import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path hygiene -- allow running from project root OR from deploy/
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sim.ablation_runner import PIRL_NN  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "model_best.pt")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "deploy", "pirl_model.onnx")
OPSET_VERSION = 11


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model(weights_path: str) -> PIRL_NN:
    """Load PIRL_NN weights with safe map_location."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weight file not found: '{weights_path}'\n"
            "Run the training script first, or pass --weights <path>.\n"
            "  python deploy/train_and_export.py"
        )

    model = PIRL_NN()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[OK] Loaded weights from: {weights_path}")
    return model


def export(model: PIRL_NN, output_path: str) -> None:
    """Trace and export the model to ONNX via the legacy TorchScript exporter."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Representative input: single velocity sample.
    # Shape [batch=1, features=1] matching PIRL_NN.forward(x).
    dummy_input = torch.FloatTensor([[0.5]])  # 0.5 rad/s

    torch.onnx.export(
        model,
        (dummy_input,),  # tuple required by torch.onnx.export signature
        output_path,
        input_names=["velocity"],
        output_names=["friction_compensation"],
        dynamic_axes={
            "velocity": {0: "batch_size"},
            "friction_compensation": {0: "batch_size"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # legacy TorchScript path -- no onnxscript needed
    )
    print(f"[OK] ONNX model exported -> {output_path}")


def validate(model: PIRL_NN, output_path: str, n_samples: int = 256) -> None:
    """
    Numeric parity check: compare PyTorch vs. ONNX Runtime outputs
    over a velocity sweep covering the full Stribeck operating range.
    Raises RuntimeError if max absolute deviation exceeds 1e-5 Nm.
    """
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError:
        print(
            "[!] onnxruntime not installed -- skipping numeric validation.\n"
            "    Install with:  pip install onnxruntime"
        )
        return

    # Sweep: stiction zone, Stribeck knee, viscous regime
    velocities = np.linspace(-3.0, 3.0, n_samples, dtype=np.float32).reshape(-1, 1)

    # PyTorch reference
    with torch.no_grad():
        pt_out = model(torch.from_numpy(velocities)).numpy()

    # ONNX Runtime
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # suppress INFO logs
    session = ort.InferenceSession(output_path, sess_options)
    onnx_out = session.run(
        output_names=["friction_compensation"],
        input_feed={"velocity": velocities},
    )[0]

    max_err = float(np.abs(pt_out - onnx_out).max())
    mean_err = float(np.abs(pt_out - onnx_out).mean())

    print(
        f"[OK] Numeric validation  |  max delta = {max_err:.2e} Nm  |  mean delta = {mean_err:.2e} Nm"
    )

    if max_err > 1e-5:
        raise RuntimeError(
            f"ONNX parity check FAILED: max deviation {max_err:.2e} Nm exceeds 1e-5 Nm.\n"
            "Check for unsupported ops or opset mismatch."
        )
    print(
        "[OK] Parity check PASSED -- ONNX graph matches PyTorch to numerical precision."
    )


def print_model_summary(model: PIRL_NN) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[i]  PIRL_NN  |  total params: {total_params}  |  trainable: {trainable}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PIRL_NN (PyTorch) -> ONNX for embedded deployment."
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Path to trained .pt weights file (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Destination .onnx file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run onnxruntime numeric parity check after export.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  PIRL_NN  .  PyTorch -> ONNX Export")
    print("=" * 60)

    model = load_model(args.weights)
    print_model_summary(model)

    export(model, args.output)

    if args.validate:
        validate(model, args.output)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[i]  Output size: {size_kb:.1f} KB")
    print("=" * 60)
    print("Done. Deploy 'pirl_model.onnx' to your target runtime.")
    print("=" * 60)
