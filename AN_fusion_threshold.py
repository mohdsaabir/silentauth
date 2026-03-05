import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ================================
# Load data
# ================================

df = pd.read_csv("AN_fusion_final.csv")

genuine = df[df["label"] == "genuine"]["fusion_score"]
impostor = df[df["label"] == "impostor"]["fusion_score"]

print("\n==============================")
print("Basic Score Statistics")
print("==============================")

print("\nGenuine Scores")
print("Count :", len(genuine))
print("Min   :", genuine.min())
print("Max   :", genuine.max())
print("Mean  :", genuine.mean())
print("Std   :", genuine.std())

print("\nImpostor Scores")
print("Count :", len(impostor))
print("Min   :", impostor.min())
print("Max   :", impostor.max())
print("Mean  :", impostor.mean())
print("Std   :", impostor.std())

# ================================
# Midpoint Threshold
# ================================

midpoint_threshold = (genuine.min() + impostor.max()) / 2

print("\n==============================")
print("Midpoint Threshold Estimate")
print("==============================")

print("Midpoint threshold =", midpoint_threshold)

# ================================
# Histogram
# ================================

plt.figure(figsize=(8,5))

plt.hist(genuine, bins=30, alpha=0.6,color="green", label="Genuine")
plt.hist(impostor, bins=30, alpha=0.6,color="red", label="Impostor")

plt.axvline(midpoint_threshold, color='red', linestyle='--', label='Midpoint Threshold')

plt.title("Fusion Score Distribution")
plt.xlabel("Fusion Score")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("results/fusion_histogram.png", dpi=300, bbox_inches="tight")
plt.show()

# ================================
# ROC + AUC
# ================================

labels = np.array([1 if x == "genuine" else 0 for x in df["label"]])
scores = df["fusion_score"].values

fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

print("\n==============================")
print("ROC / AUC")
print("==============================")

print("AUC =", roc_auc)

# Plot ROC
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label="ROC curve (AUC = %0.4f)" % roc_auc)
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("results/roc_curve.png", dpi=300, bbox_inches="tight")
plt.legend()
plt.show()

# ================================
# EER Calculation
# ================================

fnr = 1 - tpr
eer_index = np.nanargmin(np.absolute((fnr - fpr)))

eer_threshold = thresholds[eer_index]
eer = fpr[eer_index]

print("\n==============================")
print("Equal Error Rate (EER)")
print("==============================")

print("EER =", eer)
print("EER Threshold =", eer_threshold)

# ================================
# Youden's Index
# ================================

youden_index = tpr - fpr
best_index = np.argmax(youden_index)

youden_threshold = thresholds[best_index]

print("\n==============================")
print("Youden Optimal Threshold")
print("==============================")

print("Youden Threshold =", youden_threshold)
print("Youden Index =", youden_index[best_index])



# ================================
# Classification Metrics using Optimal Threshold
# ================================

threshold = 0.63

print("\n==============================")
print("Classification Metrics")
print("==============================")
print("Using Threshold =", threshold)

# Convert labels to numeric
true_labels = np.array([1 if x == "genuine" else 0 for x in df["label"]])

# Predict using threshold
predictions = (scores >= threshold).astype(int)

# Confusion matrix components
TP = np.sum((predictions == 1) & (true_labels == 1))
TN = np.sum((predictions == 0) & (true_labels == 0))
FP = np.sum((predictions == 1) & (true_labels == 0))
FN = np.sum((predictions == 0) & (true_labels == 1))

# Metrics
accuracy = (TP + TN) / len(df)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

FAR = FP / (FP + TN) if (FP + TN) != 0 else 0
FRR = FN / (FN + TP) if (FN + TP) != 0 else 0

print("\nConfusion Matrix")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print("\nPerformance Metrics")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nBiometric Metrics")
print("FAR:", FAR)
print("FRR:", FRR)