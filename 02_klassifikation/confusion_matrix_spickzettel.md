# Spickzettel: Confusion Matrix, Precision & Recall

## Die 4 Felder — Merkhilfe

```
                    VORHERGESAGT
                  Negativ   Positiv
TATSÄCHLICH  Neg |   TN   |   FP  |
             Pos |   FN   |   TP  |
```

| Kürzel | Name | Vorhersage | Realität | Bedeutung |
|--------|------|-----------|---------|-----------|
| **TP** | True Positive  | Positiv | Positiv | ✅ Richtig erkannt |
| **TN** | True Negative  | Negativ | Negativ | ✅ Richtig abgelehnt |
| **FP** | False Positive | Positiv | Negativ | ❌ Falscher Alarm |
| **FN** | False Negative | Negativ | Positiv | ❌ Übersehen! |

---

## Merkhilfe

```
True/False  →  war die Vorhersage richtig oder falsch?
Positive/Negative  →  was hat das Modell vorhergesagt?
```

---

## Formeln

```
Precision  =  TP / (TP + FP)   →  von meinen positiven Vorhersagen, wie viele stimmen?
Recall     =  TP / (TP + FN)   →  von allen echten Positiven, wie viele habe ich gefunden?
F1-Score   =  2 × (P × R) / (P + R)   →  Kompromiss aus beiden
Accuracy   =  (TP + TN) / Alle
```

---

## Wann was wichtig ist

| Situation | Schlimmster Fehler | Metrik |
|-----------|-------------------|--------|
| Krebs-Diagnose | FN — kranken Patienten übersehen | **Recall** |
| Spam-Filter | FP — wichtige Mail löschen | **Precision** |
| Beides gleich wichtig | — | **F1-Score** |
| Ausgewogene Klassen | — | **Accuracy** |

---

## Beispiele zum Nachrechnen

### Beispiel 1
> 100 kranke Patienten. Modell erkennt 80 korrekt.
```
Recall = 80 / (80 + 20) = 80%
```

### Beispiel 2
> Modell sagt bei 50 Patienten "krank". Nur 40 sind wirklich krank.
```
Precision = 40 / (40 + 10) = 80%
```

---

## Code

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Alles auf einmal
print(classification_report(y_test, y_pred))

# Confusion Matrix visualisieren
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negativ', 'Positiv'],
            yticklabels=['Negativ', 'Positiv'])
```
