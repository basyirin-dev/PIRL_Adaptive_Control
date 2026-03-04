# -------------------------------------------------------------------
# FastTrack Phase 1: Simulation Reproducibility Container
# Target: PIRL Stribeck Friction Simulation (Sim-RMSE < 0.05 rad)
# -------------------------------------------------------------------

# Match the Python version strictly to your main.yml CI specification
FROM python:3.10-slim

# Enforce deterministic behavior and suppress buffer delays
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Force PyTorch to use deterministic algorithms for reproducible RMSE
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONHASHSEED=42

# Define execution directory
WORKDIR /app

# Install minimal system dependencies (e.g., for matplotlib or compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST to leverage Docker layer caching
# Do NOT invalidate this layer when changing Python scripts
COPY requirements.txt .

# Install PyTorch and simulation dependencies
# Note: Ensure PyTorch is strictly specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the core logic (Sim, Checkpoints, and Scripts)
COPY sim/ ./sim/
COPY notebooks/pirl_model.pth ./notebooks/pirl_model.pth
COPY reproduce_results.py .

# Create output directory for artifact generation (Figure 1)
RUN mkdir -p docs/figures

# The container acts as an executable that strictly runs the Gate 1 check
ENTRYPOINT ["python", "reproduce_results.py"]
