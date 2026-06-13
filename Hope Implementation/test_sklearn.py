import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
y_true = [0, 1, 1, 1, 0, 0, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1, 0, 1]
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred, average='weighted')
prec = precision_score(y_true, y_pred, average='weighted')
print(f"Acc: {acc}, Rec: {rec}, Prec: {prec}")
