# Machine Learning Almanach

Persönliche Sammlung von ML-Materialien für den Ironhack-Kurs — auf Deutsch und Englisch.

---

## Struktur

```
machine-learning-almanach/
│
├── deutsch/                    # Vollständige deutsche Lernmaterialien (6 Themen)
│   ├── 00_START_HIER.md        # Lernpfad & Begriffstabelle
│   ├── 01_scikit_learn/
│   ├── 02_klassifikation/
│   ├── 03_regression/
│   ├── 04_feature_engineering/
│   ├── 05_bias_varianz/
│   └── 06_ensemble_methoden/
│
├── notebooks/                  # Standalone Jupyter Notebooks
│   ├── ML_Cheatsheet.ipynb     # Datensatz-unabhängiger ML-Workflow
│   └── example_salaries.ipynb  # Praxis-Beispiel: Gehaltsvorhersage
│
├── docs/                       # Markdown-Referenzdokumente
│   ├── ML_Bible_DE.md          # Vollständige ML-Referenz (Deutsch)
│   └── ML_Bible_EN.md          # Vollständige ML-Referenz (Englisch)
│
├── scripts/                    # Wiederverwendbare Python-Hilfsfunktionen
│   └── ml_utils.py             # EDA, Datenreinigung, Profiling
│
├── quizzes/                    # Interaktive HTML-Quizze
│   └── quiz_ml.html            # 50 ML-Fragen (im Browser öffnen)
│
└── data/                       # Datensätze
    └── salaries.csv
```

---

## Deutsche Lernmaterialien (`deutsch/`)

Jedes Thema enthält **3 Dateien**:

| Datei | Zweck |
|-------|-------|
| `notebook.ipynb` | Konzepte lernen — Erklärungen + ausführbarer Code |
| `zusammenfassung.md` | Kompakte Referenz zum Nachschlagen |
| `uebungen.ipynb` | Aufgaben mit Musterlösungen |

### Themenreihenfolge (empfohlen)

1. **Scikit-Learn Grundlagen** — Workflow, API, Klassifikation vs. Regression
2. **Klassifikation** — KNN, Logistische Regression, Decision Tree, SVM, Metriken
3. **Regression** — Lineare, Ridge, Lasso, Decision Tree Regression, MAE/RMSE/R²
4. **Feature Engineering** — Encoding, Skalierung, Korrelationsanalyse
5. **Bias-Varianz-Tradeoff** — Overfitting, Cross-Validation, Regularisierung
6. **Ensemble-Methoden** — Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost

→ Mit [`deutsch/00_START_HIER.md`](deutsch/00_START_HIER.md) beginnen.

---

## Schnellstart

```bash
# Repo klonen
git clone https://github.com/zeroKool1ne/machine-learning-almanach.git
cd machine-learning-almanach

# Abhängigkeiten installieren
pip install scikit-learn pandas numpy matplotlib seaborn jupyter xgboost

# Erstes Notebook öffnen
jupyter notebook deutsch/01_scikit_learn/notebook.ipynb
```

---

## Verwendete Bibliotheken

- `scikit-learn` — ML-Algorithmen
- `pandas` + `numpy` — Datenverarbeitung
- `matplotlib` + `seaborn` — Visualisierung
- `xgboost` — Extreme Gradient Boosting
- `ydata-profiling` — Automatisches EDA (für `scripts/ml_utils.py`)
