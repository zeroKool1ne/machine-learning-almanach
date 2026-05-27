# Zusammenfassung: Einführung in Scikit-Learn

## Die wichtigsten Konzepte auf einen Blick

### Was ist Machine Learning?
Ein Algorithmus **lernt aus Daten**, statt explizit programmiert zu werden.  
Beispiel: Anstatt Regeln zu schreiben „wenn Betreff SPAM enthält → Spam", zeigen wir dem Algorithmus 10.000 Spam-Mails und er lernt die Muster selbst.

---

### Die 3 Arten von ML

| Typ | Beschriftete Daten? | Typische Aufgaben |
|-----|--------------------|--------------------|
| **Supervised Learning** | Ja (X + y) | Spam-Erkennung, Preisvorhersage |
| **Unsupervised Learning** | Nein (nur X) | Kundensegmentierung, Clustering |
| **Reinforcement Learning** | Nein (Belohnungen) | Spielen, Robotik |

---

### Klassifikation vs. Regression

```
Antwort ist eine KATEGORIE  →  Klassifikation
  Beispiel: Tumor = gutartig oder bösartig?

Antwort ist eine ZAHL        →  Regression
  Beispiel: Wie teuer ist das Haus?
```

---

### Datenformat in Sklearn

```python
X.shape == (n_samples, n_features)  # 2D! Zeilen=Datenpunkte, Spalten=Merkmale
y.shape == (n_samples,)              # 1D! Ein Label pro Datenpunkt
```

---

### Der Standard-Workflow (immer gleich!)

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # oder anderer Algorithmus

# 1. Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Modell erstellen
modell = KNeighborsClassifier(n_neighbors=5)

# 3. Trainieren
modell.fit(X_train, y_train)

# 4. Vorhersagen
y_pred = modell.predict(X_test)

# 5. Bewerten
score = modell.score(X_test, y_test)
print(f"Genauigkeit: {score:.2%}")
```

---

### Wichtige Begriffe (Deutsch ↔ Englisch)

| Deutsch | Englisch | Bedeutung |
|---------|----------|-----------|
| Merkmal | Feature | Eine Eingangsvariable (z.B. Alter, Größe) |
| Zielvariable | Target / Label | Was wir vorhersagen wollen |
| Trainingsdaten | Training set | Daten zum Lernen |
| Testdaten | Test set | Daten zum Bewerten (ungesehen!) |
| Vorhersage | Prediction | Die Ausgabe des Modells |
| Genauigkeit | Accuracy | Anteil richtiger Vorhersagen |
| Hyperparameter | Hyperparameter | Einstellungen des Modells (vor dem Training) |

---

### Wichtige Methoden der Sklearn-API

| Methode | Wann | Was |
|---------|------|-----|
| `.fit(X_train, y_train)` | Einmal zum Trainieren | Modell lernt aus Daten |
| `.predict(X_test)` | Vorhersagen machen | Gibt vorhergesagte Labels zurück |
| `.score(X_test, y_test)` | Nach dem Training | Gibt Genauigkeit zurück |
| `.transform(X)` | Preprocessing | Daten umformen (z.B. normalisieren) |
| `.fit_transform(X)` | Preprocessing | fit + transform in einem Schritt |

---

### Häufige Fehler vermeiden

❌ **FALSCH:** Modell auf ALLEN Daten trainieren und dann auch auf denselben testen  
✅ **RICHTIG:** Immer Train/Test-Split machen

❌ **FALSCH:** `modell.score(X_train, y_train)` als einzige Metrik  
✅ **RICHTIG:** `modell.score(X_test, y_test)` — nur Testdaten zeigen die wahre Leistung

❌ **FALSCH:** Hyperparameter anhand der Testdaten optimieren  
✅ **RICHTIG:** Dafür Cross-Validation oder einen separaten Validierungssatz verwenden
