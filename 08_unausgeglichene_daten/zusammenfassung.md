# Zusammenfassung: Unausgeglichene Daten

## Das Problem

```
Klasse 0 (normal): 99%  ← Mehrheitsklasse
Klasse 1 (krank):   1%  ← Minderheitsklasse

Modell sagt immer 0 → Accuracy 99% ← wertlos!
Recall für Klasse 1 = 0% ← alle Kranken übersehen!
```

---

## Die 3 Lösungen

### 1. Oversampling
```python
from sklearn.utils import resample

# Minderheitsklasse hochsampeln
minderheit_over = resample(minderheit,
                            replace=True,              # MIT Zurücklegen
                            n_samples=len(mehrheit),   # gleiche Größe
                            random_state=42)

train_balanced = pd.concat([mehrheit, minderheit_over])
```
✅ Kein Datenverlust | ❌ Nur Kopien

---

### 2. Undersampling
```python
# Mehrheitsklasse reduzieren
mehrheit_under = resample(mehrheit,
                           replace=False,                # OHNE Zurücklegen
                           n_samples=len(minderheit),
                           random_state=42)

train_balanced = pd.concat([mehrheit_under, minderheit])
```
✅ Keine Duplikate | ❌ Datenverlust

---

### 3. SMOTE (empfohlen)
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, sampling_strategy=1.0)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
```
✅ Neue synthetische Punkte | ❌ Kann unrealistisch sein

---

## Die wichtigste Regel

```
⚠️ NUR Trainingsdaten ausbalancieren!
   Testdaten bleiben wie sie sind!

# RICHTIG:
train_balanced = ausbalancieren(X_train, y_train)
modell.fit(train_balanced)
modell.score(X_test, y_test)  ← Original Testdaten!

# FALSCH:
alle_daten_balanced = ausbalancieren(X, y)   ← NIE SO!
```

---

## Wann welche Methode?

| Methode | Wann? |
|---------|-------|
| **Oversampling** | Kleiner Datensatz, kein großes Imbalance |
| **Undersampling** | Sehr großer Datensatz, starkes Imbalance |
| **SMOTE** | Standard-Empfehlung in den meisten Fällen |

---

## Metriken bei unausgeglichenen Daten

```python
# NIE nur Accuracy!
from sklearn.metrics import classification_report, balanced_accuracy_score

print(classification_report(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
```
