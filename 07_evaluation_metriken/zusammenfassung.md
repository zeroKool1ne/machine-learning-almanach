# Zusammenfassung: Evaluation Metriken

## Warum nicht nur Accuracy?

```
Betrug-Datensatz: 99.8% normal, 0.2% Betrug
Modell sagt immer "normal" → Accuracy 99.8% ← WERTLOS!
```

Bei ungleichen Klassen → Accuracy lügt!

---

## Alle Metriken auf einen Blick

```
Confusion Matrix:
                VORHERGESAGT
              Neg      Pos
TATS.    Neg |  TN  |  FP  |   FP = falscher Alarm
         Pos |  FN  |  TP  |   FN = übersehen!
```

| Metrik | Formel | Bedeutung |
|--------|--------|-----------|
| **Accuracy** | (TP+TN) / Alle | Anteil richtiger Vorhersagen |
| **Precision** | TP / (TP+FP) | Wie zuverlässig sind positive Vorhersagen? |
| **Recall** | TP / (TP+FN) | Wie viele echte Positive gefunden? |
| **F1-Score** | 2·P·R/(P+R) | Kompromiss Precision+Recall |
| **Balanced Acc.** | Ø Recall pro Klasse | Accuracy bei ungleichen Klassen |
| **AUC-ROC** | Fläche unter ROC | Modellgüte unabhängig vom Threshold |

---

## Code

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score,
                              roc_auc_score, balanced_accuracy_score)

# Alles auf einmal
print(classification_report(y_test, y_pred))

# Einzeln
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))

# AUC (braucht Wahrscheinlichkeiten)
wahrsch = modell.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, wahrsch))
```

---

## Fbeta-Score

```python
from sklearn.metrics import fbeta_score

# β=2: Recall doppelt so wichtig (Medizin, Betrug)
fbeta_score(y_test, y_pred, beta=2)

# β=0.5: Precision doppelt so wichtig (Spam)
fbeta_score(y_test, y_pred, beta=0.5)
```

---

## Threshold anpassen

```python
# Standard: P(positiv) >= 0.5 → Klasse 1
# Senken → mehr Recall, weniger Precision
# Erhöhen → mehr Precision, weniger Recall

wahrsch = modell.predict_proba(X_test)[:, 1]
y_pred_custom = (wahrsch >= 0.3).astype(int)  # Threshold = 0.3
```

---

## Wann welche Metrik?

| Situation | Metrik |
|-----------|--------|
| Ausgewogene Klassen | Accuracy |
| Unausgewogene Klassen | Balanced Accuracy / F1 |
| Übersehen gefährlich (Krebs, Betrug) | **Recall** ↑ |
| Falscher Alarm teuer (Spam) | **Precision** ↑ |
| Modelle vergleichen | **AUC-ROC** |
