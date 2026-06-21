"""Report the confusion matrix and metrics at the model's DEFAULT decision
threshold (SVM score 0; Defect = class 0 = positive) from a run's predictions.
This is the operating point reported in the paper (Fig. 10 / Sec. 3.12). For
reference it also prints the threshold-sweep points (max accuracy, recall~0.95).

Usage: python check_point.py [path/to/test_predictions.csv]
"""
import csv
import sys

predictions_path = sys.argv[1] if len(sys.argv) > 1 else "test_predictions.csv"
true_labels, scores = [], []
with open(predictions_path) as file:
    reader = csv.reader(file)
    next(reader)                                 # skip the header row
    for row in reader:
        if len(row) < 2:
            continue
        true_labels.append(int(float(row[0])))   # 0 = Defect (positive), 1 = No_Defect
        scores.append(float(row[1]))             # raw SVM decision score

n_total = len(true_labels)
n_defect = sum(1 for label in true_labels if label == 0)   # number of Defect tiles
n_no_defect = n_total - n_defect

def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else 0.0

# ---- default decision threshold: score > 0 means predicted No_Defect ----
TP = sum(1 for label, score in zip(true_labels, scores) if label == 0 and score <= 0)  # Defect kept as Defect
FN = sum(1 for label, score in zip(true_labels, scores) if label == 0 and score > 0)   # Defect missed
FP = sum(1 for label, score in zip(true_labels, scores) if label == 1 and score <= 0)  # false alarm
TN = sum(1 for label, score in zip(true_labels, scores) if label == 1 and score > 0)
print("Default threshold (score 0), as reported in the paper:")
print(f"  TP={TP} FP={FP} TN={TN} FN={FN}  "
      f"recall={ratio(TP, TP + FN)}  precision={ratio(TP, TP + FP)}  "
      f"false-positive rate={ratio(FP, FP + TN)}  accuracy={ratio(TP + TN, n_total)}")

# ---- reference: sweep the threshold to find other operating points ----
# Walk through the tiles from most defect-like (most negative score) to least,
# keeping running true/false-positive counts as the threshold moves.
is_defect = [1 if label == 0 else 0 for label in true_labels]
order_by_score = sorted(range(n_total), key=lambda i: scores[i])   # most defect-like first
tp_running = fp_running = 0
best_accuracy, best_point, recall95_point = 0.0, None, None
for i in order_by_score:
    if is_defect[i]:
        tp_running += 1
    else:
        fp_running += 1
    tn_running = n_no_defect - fp_running
    fn_running = n_defect - tp_running
    accuracy = (tp_running + tn_running) / n_total
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_point = (tp_running, fp_running, tn_running, fn_running,
                      round(tp_running / n_defect, 4),
                      round(tp_running / (tp_running + fp_running), 4))
    if recall95_point is None and tp_running >= round(0.95 * n_defect):
        recall95_point = (tp_running, fp_running, tn_running, fn_running,
                          round(tp_running / n_defect, 4),
                          round(tp_running / (tp_running + fp_running), 4),
                          round(fp_running / n_no_defect, 4))
print("For reference, two other operating points from a threshold sweep:")
print(f"  highest accuracy: TP={best_point[0]} FP={best_point[1]} TN={best_point[2]} FN={best_point[3]}  "
      f"recall={best_point[4]} precision={best_point[5]} accuracy={round(best_accuracy, 4)}")
print(f"  recall near 0.95: TP={recall95_point[0]} FP={recall95_point[1]} TN={recall95_point[2]} FN={recall95_point[3]}  "
      f"recall={recall95_point[4]} precision={recall95_point[5]} false-positive rate={recall95_point[6]}")
