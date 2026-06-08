"""Compute ROC-AUC, average precision, and roc.dat/pr.dat from a run's predictions.
Usage: python analyze_predictions.py [path/to/test_predictions.csv]   (default: ./test_predictions.csv)
Defect = class 0 = positive; score = -y_score (higher => more defect-like)."""
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "test_predictions.csv"
yt, ys = [], []
with open(path) as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) < 2:
            continue
        yt.append(int(float(row[0])))
        ys.append(float(row[1]))

n = len(yt)
pos = [1 if t == 0 else 0 for t in yt]   # defect (class 0) = positive
score = [-s for s in ys]                 # higher => more defect-like
npos, nneg = sum(pos), n - sum(pos)
print(f"n={n}, defects(pos)={npos}, nondefects={nneg}")

# verify operating point: predict defect if logit < 0
TP = FP = TN = FN = 0
for t, s in zip(yt, ys):
    pd = s < 0
    if t == 0:
        TP += pd; FN += (not pd)
    else:
        FP += pd; TN += (not pd)
print(f"@score<0 (natural threshold):  TP={TP} FP={FP} TN={TN} FN={FN}")
print(f"recall={TP/npos:.4f}  precision={TP/(TP+FP):.4f}  FPR={FP/nneg:.4f}")

order = sorted(range(n), key=lambda i: score[i], reverse=True)
TPc = FPc = 0
roc, pr = [(0.0, 0.0)], []
auc = ap = 0.0
pf = pt = prec_r = 0.0
i = 0
while i < n:
    j = i
    while j + 1 < n and score[order[j + 1]] == score[order[i]]:
        j += 1
    for k in range(i, j + 1):
        if pos[order[k]]:
            TPc += 1
        else:
            FPc += 1
    tpr, fpr = TPc / npos, FPc / nneg
    recall = TPc / npos
    precision = TPc / (TPc + FPc) if TPc + FPc else 1.0
    auc += (fpr - pf) * (tpr + pt) / 2
    ap += (recall - prec_r) * precision
    pf, pt, prec_r = fpr, tpr, recall
    roc.append((fpr, tpr))
    pr.append((recall, precision))
    i = j + 1
print(f"ROC-AUC={auc:.4f}   AP(PR)={ap:.4f}")


def ds(pts, m=200):
    if len(pts) <= m:
        return pts
    step = len(pts) / m
    out = [pts[int(k * step)] for k in range(m)]
    out.append(pts[-1])
    return out


for fn, pts in [("roc.dat", roc), ("pr.dat", pr)]:
    with open(fn, "w") as f:
        f.write("x y\n")
        for x, y in ds(pts):
            f.write(f"{x:.5f} {y:.5f}\n")
print("wrote roc.dat, pr.dat")
