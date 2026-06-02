# Zusammenfassung: Hyperparameter Tuning

## Was sind Hyperparameter?

```
Parameter       = vom Modell GELERNT (z.B. Gewichte in linearer Regression)
Hyperparameter  = von DIR gesetzt VOR dem Training (z.B. max_depth, n_estimators)
```

| Modell | Hyperparameter |
|--------|---------------|
| KNN | `n_neighbors` |
| Decision Tree | `max_depth`, `min_samples_split` |
| Random Forest | `n_estimators`, `max_depth`, `max_leaf_nodes` |
| Logistische Regression | `C`, `max_iter` |

---

## Warum Hyperparameter tunen?

```
Random Forest ohne Tuning:  97.2% Accuracy
Random Forest mit Tuning:   97.9% Accuracy  ← besser!
```

Falsche Hyperparameter → Overfitting oder Underfitting

---

## Die 3 Methoden

### 1. Manual (von Hand)
```python
# Einfach verschiedene Werte ausprobieren
model = RandomForestClassifier(n_estimators=100, max_depth=5)
```
✅ Einfach | ❌ Langsam, nicht systematisch

---

### 2. Grid Search — Alle Kombinationen testen
```python
from sklearn.model_selection import GridSearchCV

# Grid = alle Werte die du testen willst
grid = {
    'n_estimators': [10, 100, 500],
    'max_depth': [5, 10],
    'max_leaf_nodes': [15, 30, 40]
}
# 3 x 2 x 3 = 18 Kombinationen werden ALLE getestet

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=grid,
    cv=3,        # 3-Fold Cross-Validation für jede Kombination!
    verbose=1
)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)   # beste Kombination
print(grid_search.best_score_)    # bester CV-Score
grid_search.score(X_test, y_test) # Test-Score
```
✅ Systematisch, findet optimale Kombination | ❌ Langsam bei vielen Parametern

---

### 3. Random Search — Zufällige Kombinationen testen
```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Großer Parameterraum
random_grid = {
    'n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [int(x) for x in np.linspace(10, 110, 11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
# Gesamt: 10x2x12x3x3x2 = 4320 mögliche Kombinationen!

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=random_grid,
    n_iter=15,    # nur 15 zufällige Kombinationen testen
    cv=3,
    n_jobs=-1     # alle CPU-Kerne nutzen
)
random_search.fit(X_train, y_train)

print(random_search.best_params_)
random_search.score(X_test, y_test)
```
✅ Schnell bei großen Parameterräumen | ❌ Findet nicht garantiert das Optimum

---

## Grid Search vs. Random Search

```
Grid Search:
  Parameterraum: 18 Kombinationen
  Testet: ALLE 18 → sicher optimal

Random Search:
  Parameterraum: 4320 Kombinationen
  Testet: nur 15 zufällige → schnell, aber nicht garantiert optimal
```

| | Grid Search | Random Search |
|--|-------------|---------------|
| **Wann?** | Kleiner Parameterraum | Großer Parameterraum |
| **Kombinationen** | Alle | Zufällige n_iter |
| **Geschwindigkeit** | Langsam | Schnell |
| **Garantiert optimal?** | Ja (aus Grid) | Nein |

**Empfehlung:** Random Search zuerst (grob suchen), dann Grid Search in dem besten Bereich (fein suchen).

---

## Wichtige Regel: CV in Grid/Random Search

```
⚠️ GridSearchCV macht automatisch Cross-Validation!
   → Kein separates cross_val_score nötig
   → cv=3 bedeutet: jede Kombination wird 3x trainiert/getestet
   → Bei 18 Kombinationen + cv=3 = 54 Trainingsläufe!
```

---

## Bestes Modell wiederverwenden

```python
# Bestes Modell direkt nutzen (schon trainiert!)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```
