"""Find the recall~0.95 operating point and the max-accuracy point from a run's
predictions. Usage: python check_point.py [path/to/test_predictions.csv]."""
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
pos = [1 if t == 0 else 0 for t in yt]
score = [-s for s in ys]
npos, nneg = sum(pos), n - sum(pos)
order = sorted(range(n), key=lambda i: score[i], reverse=True)

TPc = FPc = 0
best_acc = 0.0
best = None
rec95 = None
for idx in order:
    if pos[idx]:
        TPc += 1
    else:
        FPc += 1
    TN = nneg - FPc
    FN = npos - TPc
    acc = (TPc + TN) / n
    if acc > best_acc:
        best_acc = acc
        best = (TPc, FPc, TN, FN, round(TPc / npos, 4), round(TPc / (TPc + FPc), 4))
    if rec95 is None and TPc >= round(0.95 * npos):
        rec95 = (TPc, FPc, TN, FN, round(TPc / npos, 4), round(TPc / (TPc + FPc), 4), round(FPc / nneg, 4))

print(f"at recall~0.95:  TP={rec95[0]} FP={rec95[1]} TN={rec95[2]} FN={rec95[3]}  recall={rec95[4]} prec={rec95[5]} FPR={rec95[6]}")
print(f"max-accuracy:    TP={best[0]} FP={best[1]} TN={best[2]} FN={best[3]}  recall={best[4]} prec={best[5]} acc={round(best_acc,4)}")
