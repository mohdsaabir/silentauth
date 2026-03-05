import pandas as pd
import numpy as np

CONTRADICTION_THRESHOLD = 0.65

df = pd.read_csv("AN_multimodal_logs_new.csv")

# weight search space
steps = np.arange(0.2, 0.7, 0.05)

best_gap = -1
best_weights = None

results = []

for fw in steps:
    for vw in steps:

        gw = 1 - fw - vw

        # basic validity
        if gw <= 0 or gw > 0.6:
            continue

        # -----------------------------
        # ARCHITECTURAL CONSTRAINTS
        # -----------------------------
        if fw < vw:
            continue

        if fw < gw:
            continue

        fusion_scores = []

        for _, row in df.iterrows():

            face_user = row["face_username"]
            voice_user = row["voice_username"]
            gesture_users = eval(row["gesture_userset"])

            face_conf = row["face_conf_raw"]
            voice_conf_raw = row["voice_conf_raw"]
            gesture_conf_raw = row["gesture_conf_raw"]

            voice_penalty = 0
            voice_conf = 0
            gesture_conf = 0

            # -----------------------------
            # FACE ANCHOR RULE
            # -----------------------------
            if face_user == "Unknown":

                voice_conf = 0
                gesture_conf = 0

            else:

                # gesture anchoring
                if face_user in gesture_users:
                    gesture_conf = gesture_conf_raw
                else:
                    gesture_conf = 0

                # voice logic
                if voice_user == face_user:
                    voice_conf = voice_conf_raw
                else:
                    voice_conf = 0

                    if voice_conf_raw >= CONTRADICTION_THRESHOLD:
                        voice_penalty = vw * voice_conf_raw

            fusion = (
                fw * face_conf
                + vw * voice_conf
                + gw * gesture_conf
                - voice_penalty
            )

            fusion_scores.append(fusion)

        df["fusion"] = fusion_scores

        genuine = df[df["ground_truth"] == "genuine"]["fusion"]
        impostor = df[df["ground_truth"] == "impostor"]["fusion"]

        gap = genuine.mean() - impostor.mean()

        results.append((fw, vw, gw, gap))

        if gap > best_gap:
            best_gap = gap
            best_weights = (fw, vw, gw)

print("\nBest Weights Found:")
print("Face:", best_weights[0])
print("Voice:", best_weights[1])
print("Gesture:", best_weights[2])
print("Separation Gap:", best_gap)