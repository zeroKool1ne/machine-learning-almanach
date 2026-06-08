# Zusammenfassung: Klassifikation

## Die wichtigsten Algorithmen

### KNN — K-Nächste-Nachbarn
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
```
- **Idee:** Neue Datenpunkte bekommen die Klasse ihrer k nächsten Nachbarn
- **Wichtig:** Daten normalisieren! (KNN ist abstandsbasiert)
- **Hyperparameter k:** Klein = Overfitting, Groß = Underfitting

### Logistische Regression
```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=10000)
```
- **Idee:** Berechnet Wahrscheinlichkeit jeder Klasse via Sigmoid-Funktion
- **Gut als Baseline** — einfach, schnell, interpretierbar
- Gibt `.predict_proba()` für Wahrscheinlichkeiten

### Decision Tree (Entscheidungsbaum)
```python
from sklearn.tree import DecisionTreeClassifier
baum = DecisionTreeClassifier(max_depth=5)
```
- **Idee:** Ja/Nein-Fragen bis zur Klassenzuweisung
- **Achtung:** Ohne `max_depth` → starkes Overfitting!
- **Vorteil:** Feature Importance und gute Visualisierbarkeit

### SVM (Support Vector Machine)
```python
from sklearn.svm import LinearSVC
svm = LinearSVC(max_iter=10000)
```
- **Idee:** Maximiere Abstand zwischen Klassen (größte Trennlinie)
- **Gut bei:** vielen Merkmalen, klarer Klassenseparation

---

## Metriken für Klassifikation

### Confusion Matrix
```
                Vorhergesagt
               Neg    Pos
Tatsächlich  Neg  TN   FP   ← Falschalarm (False Positive)
             Pos  FN   TP   ← Übersehen   (False Negative)
```

### Die wichtigsten Metriken
| Metrik | Formel | Wann wichtig? |
|--------|--------|---------------|
| **Accuracy** | (TP+TN) / Gesamt | Ausgewogene Klassen |
| **Precision** | TP / (TP+FP) | Falscher Alarm teuer (Spam-Filter) |
| **Recall** | TP / (TP+FN) | Übersehen gefährlich (Krebs-Diagnose) |
| **F1-Score** | 2 · (P·R)/(P+R) | Beides gleich wichtig |

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```

---

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(modell, X, y, cv=5)
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Overfitting erkennen

| Situation | Training | Test | Problem |
|-----------|----------|------|---------|
| Gut | ~95% | ~93% | Keins |
| Overfitting | ~99% | ~82% | Zu komplex |
| Underfitting | ~72% | ~71% | Zu simpel |

---

## Schnell-Vergleich aller Algorithmen

```python
from sklearn.model_selection import cross_val_score

modelle = {
    'KNN': KNeighborsClassifier(n_neighbors=9),
    'LogReg': LogisticRegression(max_iter=10000),
    'Tree': DecisionTreeClassifier(max_depth=5),
    'SVM': LinearSVC(max_iter=10000)
}
for name, m in modelle.items():
    s = cross_val_score(m, X, y, cv=5)
    print(f"{name}: {s.mean():.3f} ± {s.std():.3f}")
```
