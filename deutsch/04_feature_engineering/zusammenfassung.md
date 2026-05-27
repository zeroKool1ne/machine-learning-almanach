# Zusammenfassung: Feature Engineering

## Die goldene Regel
**Alle Transformationen NUR auf Trainingsdaten "lernen" (`.fit()`), dann auf alle Daten "anwenden" (`.transform()`)!**

---

## Kategorische Variablen kodieren

### One-Hot Encoding — für ungeordnete Kategorien
```python
# Farbe: Rot, Blau, Grün → Farbe_Blau: 0/1, Farbe_Grün: 0/1
pd.get_dummies(df['Farbe'], prefix='Farbe', drop_first=True, dtype=int)
```
- `drop_first=True` vermeidet Multikollinearität
- Wenn viele Kategorien → Kardinalität kann ein Problem sein

### Label Encoding — für geordnete Kategorien
```python
mapping = {'Junior': 0, 'Senior': 1}
df['Erfahrung_code'] = df['Erfahrung'].map(mapping)
```
- Nur wenn echte Reihenfolge existiert (Junior < Senior)!

---

## Skalierung

### MinMaxScaler — Werte → [0, 1]
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```
- Gut für: KNN, SVM, neuronale Netze
- Empfindlich für Ausreißer

### StandardScaler — Mittelwert=0, Std=1
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```
- **Pflicht bei:** Ridge, Lasso, SVM
- Robuster gegen Ausreißer als MinMaxScaler

---

## Feature Selection

### Korrelationsbasiert
```python
import seaborn as sns
corr = X_train.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Features mit Korrelation > 0.9 entfernen
hoch_korr = corr[corr.abs() > 0.9].stack().index
```

### Feature Importance (nach Tree-Modell)
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X_train, y_train)
wichtigkeit = pd.Series(rf.feature_importances_, index=X.columns)
# Features mit Wichtigkeit < 0.01 entfernen
```

---

## Neue Features erstellen

```python
# Verhältnis
df['preis_pro_qm'] = df['preis'] / df['flaeche']

# Differenz
df['auto_alter'] = 2025 - df['baujahr']

# Polynomiale Features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

## Binning
```python
# Numerisch → Kategorien
df['altersgruppe'] = pd.cut(df['alter'], 
                             bins=[0, 25, 40, 60, 100],
                             labels=['jung', 'erwachsen', 'mittelalt', 'senior'])
```

---

## Fehlende Werte
```python
# Einfache Imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')  # oder 'mean', 'most_frequent'
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)
```

---

## Checkliste vor dem Modelltraining

- [ ] Kategorische Variablen enkodiert?
- [ ] Skalierung gemacht (wenn nötig)?
- [ ] Fehlende Werte behandelt?
- [ ] Hoch-korrelierte Features geprüft?
- [ ] Ausreißer geprüft?
- [ ] Scaler NUR auf Trainingsdaten gefittet?
