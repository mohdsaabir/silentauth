import pandas as pd
import numpy as np

# ==========================
# Load dataset
# ==========================

df = pd.read_csv("AN_multimodal_logs_new.csv")

# ==========================
# Clean data
# ==========================

df = df.dropna(subset=["face_username", "voice_username", "voice_conf_raw"])

# Ignore rows where face is Unknown
df = df[df["face_username"] != "Unknown"]

# ==========================
# Split into groups
# ==========================

genuine = df[df["face_username"] == df["voice_username"]]
contradictions = df[df["face_username"] != df["voice_username"]]

genuine_scores = genuine["voice_conf_raw"]
contradiction_scores = contradictions["voice_conf_raw"]

print("Genuine samples:", len(genuine_scores))
print("Contradiction samples:", len(contradiction_scores))

print("\nGenuine score stats")
print(genuine_scores.describe())

print("\nContradiction score stats")
print(contradiction_scores.describe())

# ==========================
# Threshold sweep
# ==========================

thresholds = np.arange(0.4, 0.9, 0.001)

best_t = None
min_diff = float("inf")

print("\nSweeping thresholds...")

for t in thresholds:

    # Genuine voices that would be penalized
    genuine_penalized = np.mean(genuine_scores < t)

    # Contradictions that exceed threshold
    contradiction_danger = np.mean(contradiction_scores > t)

    diff = abs(genuine_penalized - contradiction_danger)

    if diff < min_diff:
        min_diff = diff
        best_t = t
        best_g = genuine_penalized
        best_c = contradiction_danger

# ==========================
# Result
# ==========================

print("\n==============================")
print("Optimal Contradiction Threshold")
print("==============================")

print(f"Threshold ≈ {best_t:.3f}")
print(f"Genuine penalized  : {best_g*100:.2f}%")
print(f"Contradictions > T : {best_c*100:.2f}%")
print(f"Difference         : {min_diff:.5f}")