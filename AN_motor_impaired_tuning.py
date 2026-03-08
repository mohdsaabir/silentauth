import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================

CSV_PATH = "AN_multimodal_logs_new.csv"
THRESHOLD = 0.63
CONTRADICTION_THRESHOLD = 0.65

FACE_START = 0.50
FACE_END = 0.90
STEP = 0.01

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(CSV_PATH)

# ===============================
# GENUINE / IMPOSTOR SPLIT
# ===============================

genuine_mask = df["ground_truth"].str.lower() == "genuine"
impostor_mask = df["ground_truth"].str.lower() == "impostor"

# ===============================
# WEIGHT SWEEP
# ===============================

results = []

for face_w in np.arange(FACE_START, FACE_END + STEP, STEP):

    voice_w = 1 - face_w

    fusion_scores = []

    for _, row in df.iterrows():

        face_user = row["face_username"]
        voice_user = row["voice_username"]

        face_conf = row["face_conf_raw"]
        voice_conf_raw = row["voice_conf_raw"]

        voice_conf = 0
        voice_penalty = 0

        # ============================
        # FACE ANCHOR LOGIC
        # ============================

        if face_user == "Unknown":

            voice_conf = 0

        else:

            if voice_user == face_user:

                voice_conf = voice_conf_raw

            else:

                voice_conf = 0

                if voice_conf_raw >= CONTRADICTION_THRESHOLD:
                    voice_penalty = voice_w * voice_conf_raw

        # ============================
        # FUSION SCORE
        # ============================

        fusion = (
            face_w * face_conf
            + voice_w * voice_conf
            - voice_penalty
        )

        fusion_scores.append(fusion)

    df["fusion"] = fusion_scores

    decision = df["fusion"] >= THRESHOLD

    FAR = decision[impostor_mask].mean()
    FRR = (~decision[genuine_mask]).mean()

    results.append({
        "face_weight": round(face_w, 2),
        "voice_weight": round(voice_w, 2),
        "FAR": FAR,
        "FRR": FRR,
        "error_sum": FAR + FRR
    })

results_df = pd.DataFrame(results)

# ===============================
# SHOW RESULTS
# ===============================

print("\n===== ALL WEIGHT RESULTS =====")
print(results_df)

best = results_df.loc[results_df["error_sum"].idxmin()]

print("\n===== BEST WEIGHTS =====")
print(best)

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
plt.title("Motor-Impaired Preset Weight Sweep")

plt.legend()

plt.savefig("results/motor_impaired_weight.png", dpi=300, bbox_inches="tight")

plt.show()