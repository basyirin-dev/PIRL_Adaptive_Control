# Physics-Informed Residual Learning for Adaptive Control

- **Principal Investigator:** Basyirin Amsyar bin Basri
- **Status:** Phase 1 (Simulation & Theory)
- **License:** MIT
- **Target Hardware:** Teensy 4.1 / ESP32
- **Control Frequency Goal:** >100Hz (<10ms latency)

## 📌 Project Overview
This research develops a Sim-to-Real adaptive control framework. It utilizes Physics-Informed Neural Networks (PINNs) to learn and compensate for nonlinear friction dynamics (specifically Stribeck and LuGre models) that analytical PID controllers fail to capture.

The core objective is to bridge the "Reality Gap" in low-cost actuator control by learning the residual dynamics (the "rust") rather than the full system dynamics.

## ⚡ Technical Constraints (The "No" List)
Strict adherence to the following engineering constraints:

1.  **Latency:** Control loop must execute in < 10ms on embedded hardware.
2.  **Compute:** Inference must run entirely on edge (Teensy 4.1). No PC-in-the-loop control.
3.  **Dynamics:** Single-joint focus; no complex multi-link dynamics initially.
4.  **Learning:** No "Learning from Scratch" on hardware. Pre-training via Sim-to-Real is mandatory.

## 📂 Repository Structure
*   `sim/`: Differentiable physics simulators (JAX/PyTorch) and training loops.
*   `firmware/`: C++ embedded control code (Teensy 4.1).
*   `notebooks/`: Exploratory analysis and figure generation.
*   `docs/`: Theory derivations and manuscript drafts.

## 🛠️ Installation (Phase 1)

```bash
# Clone repository
git clone [https://github.com/basyirin-dev/PIRL_Adaptive_Control.git](https://github.com/basyirin-dev/PIRL_Adaptive_Control.git)
cd PIRL_Adaptive_Control

# Install dependencies (Phase 1: Sim only)
pip install -r requirements.txt
```

## 📚 Citation
See `CITATION.cff` for academic referencing.
