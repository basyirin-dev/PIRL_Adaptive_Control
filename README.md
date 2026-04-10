# Physics-Informed Residual Learning for Adaptive Friction Compensation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)

A research project developing Sim-to-Real adaptive control frameworks using Physics-Informed Neural Networks (PINNs) to compensate for nonlinear friction dynamics in low-cost actuators.

## 📋 Project Overview

This research addresses the challenge of precise motion control in cost-constrained robotic systems where traditional PID controllers fail to capture complex friction phenomena (Stribeck effect, stiction, Coulomb friction). The core innovation is learning the **residual dynamics** — the "rust" between ideal models and real hardware — rather than full system identification.

### Key Contributions
- **Physics-Informed Learning**: PINN architecture constrained by Stribeck and LuGre friction models
- **Sim-to-Real Transfer**: Pre-training in differentiable physics simulators before hardware deployment
- **Edge Deployment**: Sub-10ms inference latency on embedded targets (Teensy 4.1)
- **Hybrid Control**: Combines classical PID with learned residual compensation

## 🎯 Technical Constraints

| Constraint | Requirement | Rationale |
|------------|-------------|-----------|
| **Latency** | < 10ms control loop | Real-time performance for stable control |
| **Compute** | Edge-only inference | No PC-in-the-loop; standalone operation |
| **Dynamics** | Single-joint focus | Proof-of-concept before multi-DOF extension |
| **Learning** | Sim-to-Real pre-training | Safety; no unsafe exploration on hardware |

## 📂 Repository Structure

```
PIRL_Adaptive_Control/
├── sim/                          # Differentiable physics simulators & training
│   ├── controllers.py           # Baseline controllers (PID, PPO, etc.)
│   ├── pirl_network.py          # Physics-Informed Neural Network
│   ├── sim_env.py               # Gym-like simulation environment
│   └── train_*                  # Training scripts
├── firmware/                     # Embedded C++ code for Teensy 4.1
│   ├── src/main.cpp             # Main control loop
│   ├── lib/                     # Hardware abstractions
│   └── test/                    # Unit tests and validation
├── notebooks/                    # Exploratory analysis & visualization
│   ├── 01_Deriving_Stribeck.ipynb
│   ├── 02_Differentiable_Physics.ipynb
│   └── 03_Stribeck_PINN.ipynb
├── models/                       # Trained model checkpoints
├── scripts/                      # Export and deployment scripts
├── docs/                         # Theory derivations and manuscripts
└── data/                         # Training and validation datasets
```

## 🛠️ Installation

### Quick Start (Simulation Only)

```bash
# Clone repository
git clone https://github.com/basyirin-dev/PIRL_Adaptive_Control.git
cd PIRL_Adaptive_Control

# Install Python dependencies
pip install -r requirements.txt

# Run basic simulation test
python -m sim.test_sim
```

### Full Installation (Including Firmware)

```bash
# Install PlatformIO for embedded development
pip install platformio

# Build and test firmware
cd firmware
pio run -e native_test
pio test -e native_test
```

## 🚀 Usage

### Training Physics-Informed Models

```bash
# Train baseline PID controller
python scripts/train_and_export.py --controller pid --epochs 100

# Train PIRL hybrid controller
python scripts/train_and_export.py --controller hybrid --epochs 200

# Export to ONNX for deployment
python scripts/export_onnx.py --model models/pirl_model.pth
```

### Running Simulations

```bash
# Basic simulation test
python -m sim.test_sim

# Generate convergence figures
python sim/generate_figure_1.py

# Run ablation studies
python sim/run_asymmetric_ablation.py
```

### Firmware Deployment

```bash
# Build for Teensy 4.1
pio run -e teensy41

# Upload to hardware
pio run -e teensy41 --target upload

# Monitor serial output
pio device monitor -b 2000000
```

## 📊 Results

### Simulation Performance
- **Stribeck Curve Fitting**: RMSE < 0.05 Nm
- **Velocity Tracking**: < 2% error at 10 Hz
- **Convergence**: Stable within 500 training episodes

### Hardware Performance (Teensy 4.1)
- **Control Frequency**: 100 Hz (10 ms period)
- **Inference Latency**: < 100 μs
- **Total Loop Time**: < 1 ms (including sensor I/O)

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{pirl_adaptive_control,
  author = {Basyirin Amsyar bin Basri},
  title = {Physics-Informed Residual Learning for Adaptive Friction Compensation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/basyirin-dev/PIRL_Adaptive_Control}}
}
```

See `CITATION.cff` for additional citation formats.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest sim/test_sim.py -v

# Check code formatting
black --check sim/ firmware/
```

## 📅 Project Status

- ✅ **Phase 1**: Simulation & Theory (Complete)
- 🚧 **Phase 2**: Hardware Validation (In Progress)
- 📋 **Phase 3**: Multi-DOF Extension (Planned)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [GitHub Repository](https://github.com/basyirin-dev/PIRL_Adaptive_Control)
- [Issue Tracker](https://github.com/basyirin-dev/PIRL_Adaptive_Control/issues)
- [Changelog](CHANGELOG.md)

## 📧 Contact

For questions or collaboration opportunities, please contact:
- **Principal Investigator**: Basyirin Amsyar bin Basri
- **Repository**: https://github.com/basyirin-dev/PIRL_Adaptive_Control

---

**Note**: This is an active research project. APIs and implementations may change as the project evolves.
