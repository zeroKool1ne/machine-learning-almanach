# Zusammenfassung: Ensemble-Methoden

## Grundidee
Viele schwache Modelle → zusammen ein starkes Modell  
**"Die Weisheit der Masse"**

---

## Bagging vs. Boosting

| | Bagging | Boosting |
|--|---------|---------|
| **Reihenfolge** | Parallel (unabhängig) | Sequentiell |
| **Fokus** | Varianz reduzieren | Bias reduzieren |
| **Beispiele** | Random Forest | AdaBoost, GradBoost, XGBoost |
| **Robustheit** | Robuster | Empfindlicher für Ausreißer |

---

## Algorithmen im Überblick

### BaggingRegressor / BaggingClassifier
```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

modell = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=100,
    max_samples=0.8,   # 80% der Daten pro Baum
    bootstrap=True,    # Mit Zurücklegen
    random_state=42
)
```

### Random Forest ⭐ (oft beste Wahl)
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,    # Mehr = besser (bis zum Plateau)
    max_features='sqrt', # Features pro Split (Standard: gut)
    max_depth=None,      # Bäume wachsen voll
    random_state=42
)
# Feature Importance:
print(pd.Series(rf.feature_importances_, index=X.columns).sort_values())
```

### AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier
modell = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # "Stump"
    n_estimators=100,
    learning_rate=0.1
)
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingRegressor
modell = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,  # Kleiner = robuster, aber mehr Bäume nötig
    random_state=42
)
```

### XGBoost ⭐ (oft beste Performance)
```python
import xgboost as xgb
modell = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
```

---

## Wichtige Hyperparameter

| Parameter | Effekt | Tipp |
|-----------|--------|------|
| `n_estimators` | Mehr = besser | 100-500, dann CV |
| `max_depth` | Tiefer = komplexer | 3-8 für Boosting |
| `learning_rate` | Kleiner = robuster | 0.01-0.1 mit mehr Bäumen |
| `max_features` | Mehr Diversität | `sqrt` für Klassifikation, `log2` oder 0.5 für Regression |

---

## Wann welche Methode?

```
Schnell & gut?              → Random Forest
Maximale Genauigkeit?       → XGBoost / Gradient Boosting  
Interpretierbarkeit wichtig → Einzelner Entscheidungsbaum
Baseline?                   → Lineare/Logistische Regression
```

---

## Typische Leistungssteigerung (Boston-Datensatz)

```
Lineare Regression:    R² ≈ 0.74
Decision Tree:         R² ≈ 0.76
Bagging:              R² ≈ 0.84
Random Forest:        R² ≈ 0.86
Gradient Boosting:    R² ≈ 0.89
XGBoost:              R² ≈ 0.91
```
