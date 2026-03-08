import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================

CSV_PATH = "AN_multimodal_logs_new.csv"
THRESHOLD = 0.63

FACE_START = 0.50
FACE_END = 0.90
STEP = 0.01

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(CSV_PATH)

# remove attacks that rely on voice modality
df = df[df["attack_type"].str.lower() != "spoof_voice"]

# ===============================
# GENUINE / IMPOSTOR SPLIT
# ===============================

genuine_mask = df["ground_truth"].str.lower() == "genuine"
impostor_mask = df["ground_truth"].str.lower() == "impostor"

# modality scores
face_scores = df["face_conf_raw"]
gesture_scores = df["gesture_conf_effective"]

# ===============================
# WEIGHT SWEEP
# ===============================

results = []

for face_w in np.arange(FACE_START, FACE_END + STEP, STEP):

    gesture_w = 1 - face_w

    fusion_score = (
        face_w * face_scores +
        gesture_w * gesture_scores
    )

    decision = fusion_score >= THRESHOLD

    FAR = decision[impostor_mask].mean()
    FRR = (~decision[genuine_mask]).mean()

    results.append({
        "face_weight": round(face_w, 2),
        "gesture_weight": round(gesture_w, 2),
        "FAR": FAR,
        "FRR": FRR,
        "error_sum": FAR + FRR
    })

results_df = pd.DataFrame(results)

# ===============================
# SHOW ALL RESULTS
# ===============================

print("\n===== ALL WEIGHT RESULTS =====")
print(results_df)

# ===============================
# SHOW BEST WEIGHTS
# ===============================

best = results_df.loc[results_df["error_sum"].idxmin()]

print("\n===== BEST WEIGHTS =====")
print(best)

# ===============================
# SHOW ZERO ERROR REGION
# ===============================

perfect = results_df[(results_df["FAR"] == 0) & (results_df["FRR"] == 0)]

print("\n===== WEIGHTS WITH FAR=0 AND FRR=0 =====")
print(perfect)

# ===============================
# GRAPH
# ===============================

plt.figure()

plt.plot(results_df["face_weight"], results_df["FAR"], label="FAR", color="green")
plt.plot(results_df["face_weight"], results_df["FRR"], label="FRR", color="red")

plt.xlabel("Face Weight")
plt.ylabel("Error Rate")
plt.title("Voice-Impaired Preset Weight Sweep")

plt.legend()
plt.savefig("results/voice_impared_weight.png", dpi=300, bbox_inches="tight")
plt.show()