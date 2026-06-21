"""
Compute ROC-AUC, average precision, and roc.dat/pr.dat from a run's predictions.
Usage: python analyze_predictions.py [path/to/test_predictions.csv]   (default: ./test_predictions.csv)
Defect = class 0 = positive; score = -y_score (higher => more defect-like).
"""

import csv
import sys

predictions_path = sys.argv[1] if len(sys.argv) > 1 else "test_predictions.csv"
true_labels, raw_scores = [], []
with open(predictions_path) as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        if len(row) < 2:
            continue
        true_labels.append(int(float(row[0])))
        raw_scores.append(float(row[1]))

n_total = len(true_labels)
is_defect = [1 if label == 0 else 0 for label in true_labels]   # defect (class 0) = positive
defect_scores = [-s for s in raw_scores]                        # flip sign: higher = more defect-like
n_defect, n_no_defect = sum(is_defect), n_total - sum(is_defect)
print(f"{n_total} tiles: {n_defect} defects (positive class), {n_no_defect} non-defects")

# Confusion counts at the natural threshold (predict Defect when the raw score < 0).
TP = FP = TN = FN = 0
for label, raw in zip(true_labels, raw_scores):
    predicted_defect = raw < 0
    if label == 0:
        TP += predicted_defect
        FN += (not predicted_defect)
    else:
        FP += predicted_defect
        TN += (not predicted_defect)
print(f"At the natural threshold (predict defect when score < 0):  TP={TP} FP={FP} TN={TN} FN={FN}")
print(f"  recall={TP/n_defect:.4f}  precision={TP/(TP+FP):.4f}  false-positive rate={FP/n_no_defect:.4f}")

# Sweep the threshold from most to least defect-like, building the ROC and PR
# curves (and their areas) with the trapezoid rule. Tied scores advance together.
order_by_defect_score = sorted(range(n_total), key=lambda i: defect_scores[i], reverse=True)
tp = fp = 0
roc, pr = [(0.0, 0.0)], []
auc = ap = 0.0
prev_fpr = prev_tpr = prev_recall = 0.0
i = 0
while i < n_total:
    j = i
    while j + 1 < n_total and defect_scores[order_by_defect_score[j + 1]] == defect_scores[order_by_defect_score[i]]:
        j += 1
    for k in range(i, j + 1):
        if is_defect[order_by_defect_score[k]]:
            tp += 1
        else:
            fp += 1
    tpr, fpr = tp / n_defect, fp / n_no_defect
    recall = tp / n_defect
    precision = tp / (tp + fp) if tp + fp else 1.0
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
    ap += (recall - prev_recall) * precision
    prev_fpr, prev_tpr, prev_recall = fpr, tpr, recall
    roc.append((fpr, tpr))
    pr.append((recall, precision))
    i = j + 1
print(f"ROC-AUC={auc:.4f}   average precision={ap:.4f}")


def downsample(points, max_points=200):
    """Thin a long list of points down to about max_points for a lighter plot file."""
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    thinned = [points[int(k * step)] for k in range(max_points)]
    thinned.append(points[-1])
    return thinned


for filename, points in [("roc.dat", roc), ("pr.dat", pr)]:
    with open(filename, "w") as file:
        file.write("x y\n")
        for x, y in downsample(points):
            file.write(f"{x:.5f} {y:.5f}\n")
print("Wrote roc.dat and pr.dat")
