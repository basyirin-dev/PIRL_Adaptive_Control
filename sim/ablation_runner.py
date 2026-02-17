import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURATION ---
DATA_DIR = "data"
DATA_FILE = "symmetric_harvest_week6.csv"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
RESULTS_PATH = "ablation_results.csv"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
HIDDEN_SIZE = 64

# --- PHYSICS CONSTANTS (The "Ground Truth" for Data Generation) ---
# These match the Phase 1 Stribeck specifications
REAL_MASS = 1.0  # kg
REAL_VISCOUS = 0.1  # Ns/m
REAL_COULOMB = 0.5  # N (Static friction base)
REAL_STATIC = 0.8  # N (Breakaway friction)
REAL_VS = 0.1  # m/s (Stribeck velocity)
REAL_DELTA = 2.0  # Stribeck exponent

# --- ESTIMATED PHYSICS (What the Controller "Thinks" it knows) ---
# We intentionally use slightly imperfect estimates for the "Physics Prior"
EST_MASS = 1.0  # Assume mass is known well
EST_DAMPING = 0.0  # We let the NN handle ALL friction (Viscous + Stribeck) as a test

# --- UTILS ---


def stribeck_friction(v):
    """
    Calculates the ground truth friction force using the exponential Stribeck model.
    F_f(v) = F_c + (F_s - F_c) * exp(-|v/v_s|^delta) + sigma * v
    """
    # Sign of velocity determines direction
    sign_v = np.sign(v)
    abs_v = np.abs(v)

    # Stribeck term
    friction = REAL_COULOMB + (REAL_STATIC - REAL_COULOMB) * np.exp(
        -np.power(abs_v / REAL_VS, REAL_DELTA)
    )

    # Add Viscous term and apply sign
    total_friction = (friction * sign_v) + (REAL_VISCOUS * v)
    return total_friction


def generate_symmetric_chirp(n_samples=2000):
    """
    Generates a frequency-swept sine wave (Chirp) to excite all dynamic modes.
    Ensures symmetry (equal positive and negative velocities).
    """
    print(">>> Generating Synthetic Stribeck Data (Symmetric Chirp)...")
    t = np.linspace(0, 10, n_samples)

    # Chirp signal: sin(2 * pi * f(t) * t) where f(t) increases linearly
    # q = sin(t^2) covers low to high frequencies
    q = np.sin(2.0 * t + 0.5 * t**2)

    # Analytical derivatives
    dq = (2.0 + t) * np.cos(2.0 * t + 0.5 * t**2)
    ddq = np.cos(2.0 * t + 0.5 * t**2) - (2.0 + t) ** 2 * np.sin(2.0 * t + 0.5 * t**2)

    # Calculate Ground Truth Torque required (Inverse Dynamics)
    # u = M*ddq + F_stribeck(dq)
    f_fric = stribeck_friction(dq)
    u_measured = (REAL_MASS * ddq) + f_fric

    # Add sensor noise
    u_measured += np.random.normal(0, 0.02, n_samples)

    return pd.DataFrame({"q": q, "dq": dq, "ddq_ref": ddq, "u_measured": u_measured})


def load_data():
    """
    Loads data if exists, otherwise generates it using Stribeck math.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"Data Loaded from {DATA_PATH}: {len(df)} samples.")
    else:
        print(f"File {DATA_PATH} not found.")
        df = generate_symmetric_chirp()
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved generated data to {DATA_PATH}")

    return df


# --- MODELS ---


class PureNN(nn.Module):
    """
    The Black Box: Learns f(q, dq, ddq) -> u
    Inputs: 3 (q, dq, ddq_ref)
    Outputs: 1 (u_total)
    """

    def __init__(self):
        super(PureNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, HIDDEN_SIZE),
            nn.Tanh(),  # Tanh often works better for control than ReLU
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.net(x)


class PIRL_NN(nn.Module):
    """
    The Residual Learner: Learns f(dq) -> u_friction_residual
    Inputs: 1 (dq) - assuming friction depends mostly on velocity
    Outputs: 1 (u_residual)
    """

    def __init__(self):
        super(PIRL_NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.net(x)


# --- TRAINING LOOPS ---


def train_pure_nn(df):
    print(">>> Training Pure NN (Black Box)...")

    # Prep Data: Inputs [q, dq, ddq], Target [u_measured]
    X = torch.tensor(df[["q", "dq", "ddq_ref"]].values, dtype=torch.float32)
    y = torch.tensor(df[["u_measured"]].values, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PureNN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        # Logging less frequently to keep console clean
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {avg_loss:.5f}")

    return loss_history


def train_pirl(df):
    print(">>> Training PIRL (Residual)...")

    # 1. CALCULATE PHYSICS PRIOR (Nominal Model)
    # u_physics = M*ddq + D*dq (Simple linear model)
    # Note: We use EST_MASS and EST_DAMPING here, representing our incomplete knowledge
    u_physics = (EST_MASS * df["ddq_ref"]) + (EST_DAMPING * df["dq"])

    # 2. CALCULATE RESIDUAL TARGET
    # u_res = u_measured - u_physics
    # This forces the NN to learn ONLY the Stribeck friction curve
    u_res = df["u_measured"] - u_physics

    # Prep Data: Inputs [dq], Target [u_res]
    X = torch.tensor(df[["dq"]].values, dtype=torch.float32)
    y = torch.tensor(u_res.values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PIRL_NN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {avg_loss:.5f}")

    return loss_history


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load or Generate Data
    df = load_data()

    # 2. Run Ablation Study
    loss_pure = train_pure_nn(df)
    loss_pirl = train_pirl(df)

    # 3. Save Metrics
    results_df = pd.DataFrame(
        {"epoch": range(EPOCHS), "loss_pure_nn": loss_pure, "loss_pirl": loss_pirl}
    )

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Ablation Complete. Results saved to '{RESULTS_PATH}'.")
    print("Run 'plot_ablation.py' to visualize Figure 1.")
