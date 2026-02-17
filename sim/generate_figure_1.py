import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONFIGURATION: ACADEMIC AESTHETICS ---
# We use a style that mimics LaTeX/IEEE papers
plt.rcParams.update(
    {
        "text.usetex": False,  # Set to True if you have local LaTeX installed
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),  # Standard column width ratio
        "figure.dpi": 300,
    }
)


def generate_mock_data(epochs=100):
    """
    Generates synthetic data matching the Week 7 Supervisor Report findings.
    Replace this with your actual 'history' lists from the training loop if available.
    """
    x = np.arange(1, epochs + 1)

    # 1. PID Baseline (Constant Error)
    # PID doesn't 'learn', so it's a flat line of error based on tuning quality.
    pid_error = np.full_like(x, 0.15, dtype=float)

    # 2. Pure Neural Network (The "Black Box")
    # Starts high, learns slowly, noisy descent.
    # Model: Exponential decay + significant noise
    nn_error = 0.5 * np.exp(-0.05 * x) + 0.05 + np.random.normal(0, 0.005, size=len(x))

    # 3. PIRL (Physics-Informed Residual Learning) - OUR METHOD
    # Starts much lower (physics prior), converges instantly to friction floor.
    # Model: Fast decay + low noise floor
    pirl_error = 0.2 * np.exp(-0.8 * x) + 0.01 + np.random.normal(0, 0.001, size=len(x))

    return x, pid_error, nn_error, pirl_error


def plot_convergence():
    # 1. Get Data
    epochs, pid_rmse, nn_rmse, pirl_rmse = generate_mock_data()

    # 2. Setup Plot
    fig, ax = plt.subplots()

    # 3. Plot Lines (Colorblind Friendly Palette)
    # PID: Grey dashed (Reference)
    ax.plot(
        epochs,
        pid_rmse,
        label="PID Baseline (Fixed)",
        color="#7f7f7f",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )

    # Pure NN: Red/Orange (The Problem)
    ax.plot(
        epochs,
        nn_rmse,
        label="Pure Neural Network (Black Box)",
        color="#d62728",
        linestyle="-",
        marker="^",
        markevery=10,
    )  # Markers to distinguish in B&W print

    # PIRL: Blue/Teal (The Solution)
    ax.plot(
        epochs,
        pirl_rmse,
        label="PIRL (Ours)",
        color="#1f77b4",
        linestyle="-",
        marker="o",
        markevery=10,
        linewidth=3,
    )  # Thicker line for emphasis

    # 4. Axes & Scales
    ax.set_xlabel(r"Training Epochs")
    ax.set_ylabel(r"RMSE (rad/s)")
    ax.set_yscale("log")  # CRITICAL: Log scale shows order-of-magnitude dominance

    # 5. Grid & Formatting
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlim(0, 60)  # Focus on early convergence
    ax.set_ylim(0.005, 1.0)

    # Custom formatting for log ticks to look clean
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # 6. Annotations (The "So What?")
    # Point out the Physics Start
    ax.annotate(
        "Physics Prior\nAdvantage",
        xy=(1, pirl_rmse[0]),
        xytext=(10, 0.3),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
    )

    # Point out the Convergence Gap
    gap_x = 50
    gap_y_top = nn_rmse[49]
    gap_y_bot = pirl_rmse[49]
    ax.annotate(
        "",
        xy=(gap_x, gap_y_bot),
        xytext=(gap_x, gap_y_top),
        arrowprops=dict(arrowstyle="<->", color="black"),
    )
    ax.text(
        gap_x + 2,
        (gap_y_top * gap_y_bot) ** 0.5,
        r"10$\times$ Accuracy",
        verticalalignment="center",
    )

    # 7. Legend
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="black")

    # 8. Title (Optional for paper, good for slides)
    ax.set_title(
        "Convergence Rate: Physics-Informed vs. Pure Learning", fontweight="bold"
    )

    # 9. Save
    plt.tight_layout()
    plt.savefig("Figure_1_Convergence.png", dpi=300)
    plt.savefig("Figure_1_Convergence.pdf")  # Vector format for LaTeX
    print("Generate: Figure_1_Convergence.png saved successfully.")
    print("Generate: Figure_1_Convergence.pdf saved successfully.")


if __name__ == "__main__":
    plot_convergence()
