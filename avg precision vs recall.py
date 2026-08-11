import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ==========================================================
# LOAD SUMMARY CSV
# ==========================================================
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
csv_path = os.path.join(RESULT_DIR, "summary.csv")

df = pd.read_csv(csv_path)

# Extract precision and recall values
precision = df["Precision"].values
recall = df["Recall"].values

# Compute averages
avg_precision = np.mean(precision)
avg_recall = np.mean(recall)

# ==========================================================
# PRECISION vs RECALL SCATTER PLOT
# ==========================================================
plt.figure(figsize=(8, 6))

# Scatter plot for each image
plt.scatter(
    recall,
    precision,
    alpha=0.7,
    label="Per-image Precision–Recall"
)

# Plot average point
plt.scatter(
    avg_recall,
    avg_precision,
    color="red",
    s=120,
    marker="*",
    label=f"Average (P={avg_precision:.2f}, R={avg_recall:.2f})"
)

# Axis labels
plt.xlabel("Recall")
plt.ylabel("Precision")

# Plot limits
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)

# Title
plt.title("Precision vs Recall per Image (YOLOv8 on KITTI)")

# Grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()
