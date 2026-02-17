import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_convergence():
    # Load Data
    try:
        df = pd.read_csv("ablation_results.csv")
    except FileNotFoundError:
        print("Error: 'ablation_results.csv' not found. Run ablation_runner.py first.")
        return

    # Setup Style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot
    plt.plot(
        df["epoch"],
        df["loss_pure_nn"],
        label="Pure NN (Black Box)",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        df["epoch"],
        df["loss_pirl"],
        label="PIRL (Physics-Informed)",
        color="blue",
        linewidth=2.5,
    )

    # Annotations
    plt.yscale("log")  # Log scale is crucial to show convergence orders of magnitude
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("MSE Loss (Log Scale)", fontsize=12)
    plt.title(
        "Ablation Study: Convergence Speed\n(Pure Data-Driven vs. Physics-Informed)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # Save
    plt.tight_layout()
    plt.savefig("Figure_1_Convergence.png", dpi=300)
    print("Generated 'Figure_1_Convergence.png'")


if __name__ == "__main__":
    plot_convergence()
