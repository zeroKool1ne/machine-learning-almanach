# Zusammenfassung: Bias-Varianz-Tradeoff & Cross-Validation

## Das Kernproblem

**Fehler = Bias² + Varianz + irreduzibler Fehler**

| | Bias | Varianz | Symptom |
|--|------|---------|---------|
| **Underfitting** | Hoch | Niedrig | Train ≈ Test, beide schlecht |
| **Overfitting** | Niedrig | Hoch | Train gut, Test viel schlechter |
| **Optimal** | Niedrig | Niedrig | Train ≈ Test, beide gut |

---

## Diagnose

### Lernkurven
```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    modell, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
```
- **Overfitting:** Großer Abstand zwischen Train- und Val-Kurve
- **Underfitting:** Beide Kurven niedrig, kleiner Abstand

---

## Datenaufteilung

### Einfache Validierung (Train / Val / Test)
```python
# 60% Training, 20% Validierung, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
```
- Validierung = für Hyperparameter-Tuning & Modellauswahl
- Test = EINMAL am Ende verwenden!

### Cross-Validation (robuster!)
```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modell, X, y, cv=kf, scoring='accuracy')
print(f"{scores.mean():.4f} ± {scores.std():.4f}")
```

---

## Lösungen

### Overfitting bekämpfen
- `max_depth` verringern (Tree)
- `n_neighbors` erhöhen (KNN)
- Regularisierung hinzufügen (Ridge/Lasso)
- Mehr Daten sammeln
- Weniger Features verwenden

### Underfitting bekämpfen
- Komplexeres Modell wählen
- Mehr Features hinzufügen / Feature Engineering
- Regularisierung verringern (kleineres α)

---

## Hyperparameter-Tuning mit Cross-Validation

```python
# Richtig: Hyperparameter mit CV optimieren
for alpha in [0.01, 0.1, 1, 10, 100]:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
    print(f"α={alpha}: {scores.mean():.4f}")

# Oder automatisch mit GridSearchCV:
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]}, cv=5)
grid.fit(X_train, y_train)
print(f"Bestes α: {grid.best_params_}")
```

---

## Das Heilige Prinzip

```
Trainingsdaten  → Modell trainieren
Validierungsdaten → Hyperparameter wählen, Modelle vergleichen
Testdaten        → NUR EINMAL am Ende zur finalen Bewertung

Testdaten nie für Entscheidungen verwenden!
```
