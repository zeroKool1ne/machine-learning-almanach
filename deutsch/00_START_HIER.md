# Deutsche Machine Learning Materialien

Alle Materialien wurden aus den englischen Kursunterlagen erstellt und auf Deutsch übersetzt und erklärt.

---

## Wie du diese Materialien nutzt

Für jedes Thema gibt es **3 Dateien**:

| Datei | Wann nutzen |
|-------|------------|
| `notebook.ipynb` | **Zuerst** — Konzepte lernen mit Erklärungen und Code |
| `zusammenfassung.md` | **Danach** — Nachschlagen, kurz vor Prüfungen |
| `uebungen.ipynb` | **Zum Üben** — Aufgaben mit Musterlösungen |

---

## Lernpfad (empfohlene Reihenfolge)

### Schritt 1: Fundament
📁 `01_scikit_learn/`
- Was ist Sklearn?
- Klassifikation vs. Regression
- Der universelle Workflow: fit → predict → score

### Schritt 2: Klassifikation (Kategorien vorhersagen)
📁 `02_klassifikation/`
- KNN, Logistische Regression, Decision Tree, SVM
- Metriken: Accuracy, Precision, Recall, F1, Confusion Matrix
- Cross-Validation

### Schritt 3: Regression (Zahlen vorhersagen)
📁 `03_regression/`
- Lineare Regression, Ridge, Lasso
- Decision Tree Regression, KNN Regression
- Metriken: MAE, RMSE, R²

### Schritt 4: Bessere Daten = bessere Modelle
📁 `04_feature_engineering/`
- One-Hot Encoding, Label Encoding
- Normalisierung, Standardisierung
- Korrelationsanalyse, Feature Selection

### Schritt 5: Warum scheitern Modelle?
📁 `05_bias_varianz/`
- Overfitting vs. Underfitting
- Lernkurven als Diagnose-Werkzeug
- Cross-Validation richtig einsetzen
- Regularisierung

### Schritt 6: Die mächtigsten Algorithmen
📁 `06_ensemble_methoden/`
- Bagging, Random Forest
- AdaBoost, Gradient Boosting, XGBoost
- Wann welche Methode?

---

## Wichtige deutsche Fachbegriffe

| Englisch | Deutsch |
|---------|---------|
| Feature | Merkmal / Attribut |
| Label / Target | Zielvariable / Klasse |
| Training set | Trainingsdaten |
| Test set | Testdaten |
| Overfitting | Überanpassung |
| Underfitting | Unteranpassung |
| Cross-validation | Kreuzvalidierung |
| Accuracy | Genauigkeit / Trefferquote |
| Prediction | Vorhersage |
| Hyperparameter | Hyperparameter (gleich) |
| Regularization | Regularisierung |
| Normalization | Normalisierung |
| Confusion Matrix | Wahrheitsmatrix / Konfusionsmatrix |
| Feature importance | Merkmalswichtigkeit |
| Ensemble | Ensemble (gleich) |

---

## Hilfreiche Erinnerungen

### Das passiert bei JEDEM ML-Projekt:
```python
# 1. Daten laden
X, y = ...

# 2. Aufteilen (immer!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. (Optional) Skalieren — immer fit nur auf Training!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 4. Modell wählen, trainieren
modell = WelcherAlgorithmAuchImmer(hyperparameter=wert)
modell.fit(X_train, y_train)

# 5. Bewerten
print(f"Test-Score: {modell.score(X_test, y_test):.4f}")
```

### Goldene Regeln:
1. **Testdaten sind heilig** — nie für Entscheidungen verwenden
2. **Scaler fit() nur auf Training** — sonst Datenleck
3. **Immer Train vs. Test vergleichen** — um Overfitting zu erkennen
4. **Cross-Validation > einfacher Split** — zuverlässigere Schätzung
