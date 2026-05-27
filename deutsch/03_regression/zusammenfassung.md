# Zusammenfassung: Regression

## Algorithmen

### Lineare Regression
```python
from sklearn.linear_model import LinearRegression
modell = LinearRegression()
# ŷ = w₀ + w₁·x₁ + ... + wₙ·xₙ
```
- Findet die "beste Gerade" durch die Datenpunkte (minimiert MSE)
- `modell.coef_` = Gewichte (Einfluss jedes Features)
- Keine Regularisierung → kann bei vielen Features overfitten

### Ridge-Regression (L2)
```python
from sklearn.linear_model import Ridge
modell = Ridge(alpha=1.0)
```
- Wie lineare Regression, aber bestraft große Koeffizienten
- α klein = kaum Regularisierung; α groß = starke Regularisierung
- Koeffizienten → klein, aber nie exakt 0
- **Muss Daten skalieren!**

### Lasso-Regression (L1)
```python
from sklearn.linear_model import Lasso
modell = Lasso(alpha=1.0)
```
- Setzt unwichtige Koeffizienten auf **exakt 0** → Feature Selection!
- Gut wenn du glaubst, dass viele Features irrelevant sind
- **Muss Daten skalieren!**

### Decision Tree Regression
```python
from sklearn.tree import DecisionTreeRegressor
modell = DecisionTreeRegressor(max_depth=5)
```
- Gibt Mittelwert der Datenpunkte im Blatt zurück
- Kann nicht-lineare Beziehungen modellieren
- Ohne `max_depth`: starkes Overfitting!

### KNN Regression
```python
from sklearn.neighbors import KNeighborsRegressor
modell = KNeighborsRegressor(n_neighbors=5, weights='distance')
```
- Gibt Mittelwert der k nächsten Nachbarn zurück
- **Muss Daten normalisieren!**

---

## Bewertungsmetriken

| Metrik | Formel | Einheit | Interpretation |
|--------|--------|---------|----------------|
| **MAE** | ∅ \|y - ŷ\| | Wie y | "Durchschnittlich X daneben" |
| **MSE** | ∅ (y - ŷ)² | y² | Bestraft Ausreißer stark |
| **RMSE** | √MSE | Wie y | MAE + Ausreißer-Sensitivität |
| **R²** | 1 - MSE/Var(y) | Einheitenlos (0-1) | "Modell erklärt X% der Varianz" |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
```

---

## Ridge vs. Lasso — wann was?

```
Viele irrelevante Features? → Lasso  (macht automatisch Feature Selection)
Alle Features relevant?      → Ridge  (schrumpft alle, entfernt aber keinen)
Unsicher?                    → ElasticNet (Kombination beider)
```

---

## Regularisierung: Skalierung ist Pflicht!

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # fit NUR auf Training!
X_test_s  = scaler.transform(X_test)

ridge = Ridge(alpha=10)
ridge.fit(X_train_s, y_train)
```

---

## R² interpretieren

- **R² = 1.0**: Perfekte Vorhersage
- **R² = 0.8**: Modell erklärt 80% der Varianz — gut!
- **R² = 0.0**: Nicht besser als "immer den Mittelwert vorhersagen"
- **R² < 0**: Schlechter als der Mittelwert → etwas stimmt nicht
