# Machine Learning Bibel 🧠
> Eine vollständige Referenz — von Rohdaten bis zum trainierten Modell.  
> Zum Offenhalten während der Arbeit. Wird kontinuierlich erweitert.

---

## Inhaltsverzeichnis

1. [Arten von Machine Learning](#1-arten-von-machine-learning)
2. [Klassifikation vs. Regression](#2-klassifikation-vs-regression)
3. [Datenstruktur](#3-datenstruktur)
4. [Vollständiger ML-Workflow](#4-vollständiger-ml-workflow)
5. [Datenbereinigung & Preprocessing](#5-datenbereinigung--preprocessing)
6. [Arbeiten mit SQL-Daten](#6-arbeiten-mit-sql-daten)
7. [Scikit-Learn — Alle Methoden erklärt](#7-scikit-learn--alle-methoden-erklärt)
8. [Modell-Evaluation](#8-modell-evaluation)
9. [Algorithmen-Referenz](#9-algorithmen-referenz)
10. [Schnell-Import-Cheatsheet](#10-schnell-import-cheatsheet)

---

## 1. Arten von Machine Learning

Machine Learning wird in drei Hauptparadigmen unterteilt, je nachdem wie der Algorithmus aus Daten lernt.

### 1.1 Supervised Learning (Überwachtes Lernen)
Der Algorithmus lernt aus **beschrifteten Daten** — d.h. jedes Trainingsbeispiel hat eine Eingabe (Features) und eine bekannte korrekte Ausgabe (Label/Zielgröße).

**Wie es funktioniert:** Das Modell findet eine Abbildungsfunktion `f(X) → y`, die auf ungesehene Daten verallgemeinert.

**Wann verwenden:** Wenn du historische Daten mit bekannten Ergebnissen hast und zukünftige Ergebnisse vorhersagen möchtest.

| Algorithmus | Aufgabe | Beispielanwendung |
|-------------|---------|------------------|
| Lineare Regression | Regression | Hauspreise vorhersagen |
| Logistische Regression | Klassifikation | Spam-Erkennung |
| KNN | Beides | Kundensegmentierung |
| Entscheidungsbaum | Beides | Kreditgenehmigung |
| Random Forest | Beides | Medizinische Diagnose |
| SVM | Beides | Bildklassifikation |

### 1.2 Unsupervised Learning (Unüberwachtes Lernen)
Der Algorithmus lernt aus **unbeschrifteten Daten** — es gibt keine vorgegebenen richtigen Antworten. Das Modell findet eigenständig versteckte Muster oder Strukturen.

**Wann verwenden:** Wenn du Strukturen in Daten ohne vordefinierten Zielwert entdecken möchtest.

| Algorithmus | Aufgabe | Beispielanwendung |
|-------------|---------|------------------|
| K-Means | Clustering | Kundengruppen bilden |
| DBSCAN | Clustering | Anomalieerkennung |
| PCA | Dimensionsreduktion | Feature-Komprimierung |
| Autoencoder | Feature-Lernen | Bildkomprimierung |

### 1.3 Reinforcement Learning (Bestärkendes Lernen)
Ein Agent lernt durch **Interaktion mit einer Umgebung**. Er erhält Belohnungen für gute Aktionen und Strafen für schlechte, und lernt, die Gesamtbelohnung zu maximieren.

**Wann verwenden:** Bei sequenziellen Entscheidungsproblemen, bei denen der Agent erkunden und Feedback erhalten kann.

Beispiele: Spielende KI (Schach, Go), Robotersteuerung, Empfehlungssysteme.

### 1.4 Semi-Supervised & Self-Supervised Learning
- **Semi-supervised:** Mischung aus beschrifteten und unbeschrifteten Daten. Häufig verwendet, wenn das Beschriften teuer ist.
- **Self-supervised:** Das Modell generiert eigene Labels aus den Daten (z.B. das nächste Wort in einem Satz vorhersagen — so werden LLMs trainiert).

---

## 2. Klassifikation vs. Regression

Die wichtigste Frage vor dem Aufbau eines Modells: **"Welche Art von Ausgabe sage ich vorher?"**

### Klassifikation
Sagt eine **diskrete Kategorie** vorher — die Ausgabe gehört zu einer von wenigen festen Klassen.

- **Binäre Klassifikation:** 2 Klassen (Ja/Nein, Spam/Kein Spam, 0/1)
- **Multi-Klassen-Klassifikation:** 3+ Klassen (Hund/Katze/Vogel, A/B/C/D)
- **Multi-Label-Klassifikation:** Mehrere Klassen können gleichzeitig wahr sein (ein Film kann Action UND Komödie sein)

**Ausgabe:** Ein Klassenlabel (und oft eine Wahrscheinlichkeit)

**Bewertungsmetriken:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Beispiele:**
- Wird dieser Kunde abwandern? → Ja/Nein
- Welche Ziffer ist auf diesem Bild? → 0–9
- Welche Sprache ist dieser Text? → DE/EN/FR/...

### Regression
Sagt einen **kontinuierlichen numerischen Wert** vorher — die Ausgabe kann jede Zahl auf einer Skala sein.

**Ausgabe:** Eine Zahl (z.B. 43.200,50)

**Bewertungsmetriken:** R²-Score, MAE, MSE, RMSE

**Beispiele:**
- Für wie viel wird dieses Haus verkauft? → 382.000 €
- Wie warm wird es morgen sein? → 22,4 °C
- Welches Gehalt wird diese Person verdienen? → 67.400 $

### Entscheidungshilfe

```
Ist die Ausgabe eine Kategorie?  → Klassifikation
Ist die Ausgabe eine Zahl?       → Regression
```

> **Tipp:** Manche Probleme können auf beide Weisen formuliert werden. "Wird der Preis steigen?" ist Klassifikation. "Um wie viel wird er steigen?" ist Regression.

---

## 3. Datenstruktur

### 3.1 Das Standardformat: X und y

Scikit-learn erwartet Daten in einem bestimmten Format:

```
X = Feature-Matrix       Form: (n_samples, n_features)
y = Zielvektor           Form: (n_samples,)
```

| Begriff | Auch bekannt als | Bedeutung |
|---------|-----------------|-----------|
| **Sample** | Zeile, Beobachtung, Instanz | Ein Datenpunkt (z.B. eine Person) |
| **Feature** | Spalte, Prädiktor, Variable, Attribut | Eine messbare Eigenschaft (z.B. Alter) |
| **Target** | Label, Zielgröße, Antwort, y | Was wir vorhersagen wollen |
| **n_samples** | Anzahl der Zeilen | Wie viele Beobachtungen wir haben |
| **n_features** | Anzahl der Spalten | Wie viele Eingabevariablen wir haben |

### 3.2 NumPy-Arrays vs. Pandas DataFrames

Scikit-learn verwendet nativ **NumPy-Arrays**, aber **Pandas DataFrames werden in den meisten Fällen auch akzeptiert**.

```python
import numpy as np
import pandas as pd

# NumPy-Array
X = np.array([[1, 2], [3, 4], [5, 6]])   # Form (3, 2)

# Pandas DataFrame — funktioniert auch
df = pd.DataFrame({'alter': [25, 30, 35], 'gehalt': [40000, 60000, 80000]})
X = df[['alter', 'gehalt']]  # funktioniert mit sklearn
```

### 3.3 Wann was verwenden

| Situation | Verwende |
|-----------|----------|
| Rohdaten erkunden | Pandas DataFrame |
| An sklearn übergeben | Beides funktioniert — DataFrame bevorzugt für Lesbarkeit |
| Reine Geschwindigkeit / Mathematik | NumPy-Array |
| Nach Encoding | NumPy-Array (via `.values` oder `np.concatenate`) |

---

## 4. Vollständiger ML-Workflow

Das ist der vollständige End-to-End-Prozess, dem jedes ML-Projekt folgt.

```
1. Problem definieren
2. Daten laden
3. Daten erkunden (EDA)
4. Daten bereinigen & vorverarbeiten
5. In Train/Test aufteilen
6. Modell auswählen und trainieren
7. Modell evaluieren
8. Modell verbessern
9. (Deployment)
```

### Schritt 1: Problem definieren
Bevor du Code anfasst, beantworte:
- Was versuche ich vorherzusagen?
- Ist es Klassifikation oder Regression?
- Welche Daten habe ich?
- Wie sieht "Erfolg" aus? (Welche Metrik ist wichtig)

### Schritt 2: Daten laden

```python
import pandas as pd

# Aus CSV
df = pd.read_csv('daten.csv')

# Aus SQL (siehe Abschnitt 6)
df = pd.read_sql('SELECT * FROM tabelle', engine)

# Eingebaute sklearn-Datensätze (zum Üben)
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
```

### Schritt 3: Daten erkunden (EDA)

```python
df.shape            # (Zeilen, Spalten)
df.dtypes           # Datentypen jeder Spalte
df.head()           # erste 5 Zeilen
df.tail()           # letzte 5 Zeilen
df.describe()       # Statistiken: Mittelwert, Std, Min, Max, Quartile
df.info()           # Überblick: Datentypen + Nicht-Null-Anzahlen
df.isnull().sum()   # fehlende Werte pro Spalte
df.duplicated().sum() # Anzahl doppelter Zeilen
df['spalte'].value_counts()  # Häufigkeitstabelle für eine Spalte
df['spalte'].unique()  # alle eindeutigen Werte
```

**Visualisierung:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df.select_dtypes(include='number'))  # alle numerischen Beziehungen

plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
```

**Wonach suchen:**
- Gibt es fehlende Werte?
- Gibt es Ausreißer? (Min/Max in describe() prüfen)
- Sind Spalten vom falschen Typ? (object statt float)
- Gibt es Korrelationen zwischen Features?
- Ist die Zielgröße ausgeglichen (bei Klassifikation)?

### Schritt 4: Bereinigen & Vorverarbeiten
→ Siehe Abschnitt 5 für alle Details.

### Schritt 5: Train/Test-Aufteilung

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% gehen ins Test-Set
    random_state=42     # fester Seed → reproduzierbare Aufteilung
)
```

**Warum aufteilen?**
- Das Modell trainiert auf `X_train` und `y_train`
- Wir evaluieren es auf `X_test` und `y_test` — Daten, die es **noch nie gesehen** hat
- Ohne das würde man nur messen, wie gut es die Trainingsdaten auswendig gelernt hat (Overfitting)

**Gängige Aufteilungen:** 80/20, 75/25, 70/30

### Schritt 6: Modell trainieren

```python
from sklearn.linear_model import LinearRegression

modell = LinearRegression()      # 1. Initialisieren
modell.fit(X_train, y_train)     # 2. Trainieren
```

Das Muster ist bei **jedem** sklearn-Modell gleich. Nur Import und Klassenname ändern sich.

### Schritt 7: Evaluieren

```python
y_pred = modell.predict(X_test)  # Vorhersagen generieren

# Regression
from sklearn.metrics import r2_score, mean_absolute_error
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)

# Klassifikation
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Schritt 8: Verbessern

Möglichkeiten zur Modellverbesserung:
- Mehr/bessere Daten
- Feature Engineering (neue Features aus bestehenden erstellen)
- Andere Algorithmen ausprobieren
- Hyperparameter tunen
- Regularisierung gegen Overfitting
- Andere Train/Test-Aufteilung

---

## 5. Datenbereinigung & Preprocessing

### 5.1 Datentypen korrigieren

Viele Datensätze importieren Spalten als `object` (String), obwohl sie numerisch sein sollten. Das passiert oft durch nicht-numerische Werte wie `'T'` (Spur), `'-'` oder `'N/A'`.

```python
# Eine Spalte konvertieren
df['spalte'] = pd.to_numeric(df['spalte'], errors='coerce')
# errors='coerce' → nicht konvertierbare Werte werden zu NaN statt Absturz

# Mehrere Spalten auf einmal
spalten = ['sp1', 'sp2', 'sp3']
df[spalten] = df[spalten].apply(pd.to_numeric, errors='coerce')

# Ergebnis prüfen
df.dtypes
```

### 5.2 Fehlende Werte behandeln

**Schritt 1 — Fehlende Werte finden:**
```python
df.isnull().sum()                     # Anzahl pro Spalte
df.isnull().sum() / len(df)           # Prozentsatz pro Spalte
df[df.isnull().any(axis=1)]           # alle Zeilen mit mindestens einem fehlenden Wert
```

**Schritt 2 — Strategie wählen:**

| Situation | Strategie |
|-----------|-----------|
| Spalte hat >10% fehlende Werte | Spalte löschen |
| Zeile hat viele fehlende Werte | Zeile löschen |
| Numerische Daten, Zeitreihe | Interpolieren |
| Numerische Daten, zufällig | Mit Mittelwert oder Median auffüllen |
| Kategorische Daten | Mit häufigster Kategorie oder 'Unbekannt' auffüllen |

```python
# Spalte löschen
df.drop(columns=['schlechte_spalte'], inplace=True)

# Zeilen mit fehlenden Werten löschen
df.dropna(inplace=True)

# Mit Mittelwert auffüllen
df['sp'].fillna(df['sp'].mean(), inplace=True)

# Kategorie mit Label auffüllen
df['sp'].fillna('Unbekannt', inplace=True)

# Lineare Interpolation (gut für Zeitreihen / Wetterdaten)
df_fixed = df.interpolate()
```

### 5.3 Kategorische Variablen encodieren

ML-Algorithmen arbeiten **nur mit Zahlen**. Textkategorien müssen konvertiert werden.

#### One-Hot Encoding (pd.get_dummies) — EMPFOHLEN

Erstellt eine neue binäre Spalte für jede Kategorie. Verwenden, wenn keine natürliche Reihenfolge zwischen Kategorien besteht.

```python
# Vorher: Experience-Spalte hat Werte 'Junior', 'Senior'
# Nachher: Zwei Spalten — Experience_Junior (0/1) und Experience_Senior (0/1)

df = pd.get_dummies(df, columns=['Experience', 'Gender', 'Daltonic'])

# Boolean True/False zu Integer 0/1 konvertieren
df = df.apply(lambda x: x.astype(int) if x.dtype == bool else x)
```

**Warum nicht einfach 0 für Junior und 1 für Senior?**
Weil das implizieren würde "Senior = 2 × Junior", was mathematisch falsch ist. One-Hot Encoding behandelt sie als vollständig getrennte Kategorien.

#### Label Encoding — nur für ordinale Daten

Wenn Kategorien eine natürliche Reihenfolge haben (z.B. Klein < Mittel < Groß):

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['groesse_encoded'] = le.fit_transform(df['groesse'])
# Klein→0, Mittel→1, Groß→2
```

### 5.4 Feature Scaling (Skalierung)

Viele Algorithmen (KNN, SVM, Neuronale Netze) sind empfindlich gegenüber der **Skalierung** von Features. Wenn ein Feature im Bereich 0–1 liegt und ein anderes 0–1.000.000, dominiert das große Feature.

**Standardisierung (StandardScaler)** — transformiert zu Mittelwert=0, Std=1. Standardmäßig verwenden.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # auf Train fitten und transformieren
X_test_scaled = scaler.transform(X_test)         # nur transformieren (nie fitten!)
```

> **Wichtige Regel:** Den Scaler immer nur auf **Trainingsdaten fitten**, dann zum Transformieren der Testdaten verwenden. Wenn du auf allen Daten fittest, leckst du Informationen aus dem Test-Set.

### 5.5 Duplikate entfernen

```python
df.duplicated().sum()            # Wie viele Duplikate?
df.drop_duplicates(inplace=True)
```

### 5.6 Spalten löschen

```python
# Eine Spalte löschen
df.drop(columns=['nutzlose_spalte'], inplace=True)

# Mehrere löschen
df.drop(columns=['sp1', 'sp2', 'sp3'], inplace=True)
```

### 5.7 Vollständige Preprocessing-Pipeline

```python
# 1. Spaltentypen korrigieren
falsche_spalten = ['sp1', 'sp2']
df[falsche_spalten] = df[falsche_spalten].apply(pd.to_numeric, errors='coerce')

# 2. Fehlende Werte behandeln
df.drop(columns=['spalte_mit_zu_vielen_nans'], inplace=True)
df = df.interpolate()

# 3. Kategorische Variablen encodieren
df['text_spalte'].fillna('Unbekannt', inplace=True)
df = pd.get_dummies(df, columns=['kat1', 'kat2'])
df = df.apply(lambda x: x.astype(int) if x.dtype == bool else x)

# 4. X und y aufteilen
X = df.drop(columns=['ziel'])
y = df['ziel']

# 5. Train/Test-Aufteilung
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Features skalieren (wenn nötig)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 6. Arbeiten mit SQL-Daten

### 6.1 MySQL-Verbindung aufbauen

```python
from sqlalchemy import create_engine
import pandas as pd

# Format: 'mysql+pymysql://benutzername:passwort@host/datenbankname'
engine = create_engine('mysql+pymysql://root:deinpasswort@localhost/deine_datenbank')
```

**Benötigte Pakete installieren:**
```bash
pip install sqlalchemy pymysql
```

### 6.2 Komplette Tabelle laden

```python
df = pd.read_sql('SELECT * FROM deine_tabelle', engine)
df.head()
```

### 6.3 Mit einer Query laden

```python
query = """
    SELECT datum, temperatur, luftfeuchtigkeit
    FROM wetter
    WHERE jahr = 2023
    ORDER BY datum
"""
df = pd.read_sql(query, engine)
```

### 6.4 Workflow: MySQL → Python → ML-Modell

```
1. MySQL Workbench öffnen
2. .sql-Datei ausführen, um Datenbank zu erstellen und zu befüllen
3. In Python: Engine mit create_engine() erstellen
4. Tabelle mit pd.read_sql() in DataFrame laden
5. Normalen ML-Workflow fortfahren (bereinigen, encodieren, aufteilen, trainieren)
```

### 6.5 Nützliche SQL-Queries vor dem Laden

Diese in MySQL Workbench ausführen, um Daten zunächst zu verstehen:

```sql
-- Wie viele Zeilen?
SELECT COUNT(*) FROM tabellenname;

-- Maximum/Minimum einer Spalte?
SELECT MAX(spalte), MIN(spalte) FROM tabellenname;

-- Durchschnitt einer Spalte
SELECT AVG(spalte) FROM tabellenname;

-- Top 10 Zeilen sortiert nach einer Spalte
SELECT * FROM tabellenname ORDER BY spalte DESC LIMIT 10;

-- Filtern: Zeilen wo Bedingung wahr ist
SELECT * FROM tabellenname WHERE spalte > 100;

-- Textspalten in Zahlen casten (nötig wenn Spaltentyp VARCHAR ist)
SELECT CAST(spalte AS SIGNED) FROM tabellenname;
```

---

## 7. Scikit-Learn — Alle Methoden erklärt

### 7.1 Die universelle API

Jedes einzelne sklearn-Modell folgt der gleichen Schnittstelle:

```python
modell.fit(X_train, y_train)     # Modell auf Daten trainieren
modell.predict(X)                # Vorhersagen treffen
modell.predict_proba(X)          # Klassenwahrscheinlichkeiten vorhersagen (Klassifikatoren)
modell.score(X, y)               # Bewerten: R² für Regression, Accuracy für Klassifikation
modell.transform(X)              # Daten transformieren (Preprocessing-Objekte)
modell.fit_transform(X)          # Fit + Transform in einem Schritt
```

### 7.2 Modell-Initialisierungsparameter (Hyperparameter)

Diese werden **vor** dem Training gesetzt. Sie steuern, wie das Modell lernt.

```python
# KNN
KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')

# Lineare Regression
LinearRegression(fit_intercept=True)

# Entscheidungsbaum
DecisionTreeClassifier(max_depth=5, min_samples_split=2)
```

### 7.3 Datensätze (zum Üben)

```python
from sklearn.datasets import (
    load_iris,           # 150 Blumen, 4 Features, 3 Klassen — klassische Klassifikation
    load_diabetes,       # 442 Patienten, 10 Features — Regression
    load_digits,         # 8x8 Pixel-Bilder von Ziffern — Bildklassifikation
    load_wine,           # 178 Weine, 13 Features, 3 Klassen
    load_breast_cancer,  # 569 Tumore, 30 Features — binäre Klassifikation
)

data = load_iris()
X = data.data        # Feature-Matrix
y = data.target      # Labels
print(data.DESCR)    # vollständige Beschreibung des Datensatzes
print(data.keys())   # verfügbare Attribute
```

### 7.4 Preprocessing

```python
from sklearn.preprocessing import (
    StandardScaler,      # standardisieren: Mittelwert=0, Std=1
    MinMaxScaler,        # skalieren auf [0, 1]
    LabelEncoder,        # Labels als Ganzzahlen kodieren
    OneHotEncoder,       # kategorische Features one-hot kodieren
    PolynomialFeatures,  # polynomiale Feature-Kombinationen erstellen
)
```

### 7.5 Modellauswahl

```python
from sklearn.model_selection import (
    train_test_split,    # Daten in Train/Test aufteilen
    cross_val_score,     # k-fache Kreuzvalidierung
    GridSearchCV,        # erschöpfende Hyperparameter-Suche
    RandomizedSearchCV,  # zufällige Hyperparameter-Suche
)
```

### 7.6 Metriken

```python
from sklearn.metrics import (
    # Regression
    r2_score,                    # R²-Score: wie viel Varianz erklärt wird (1.0 = perfekt)
    mean_absolute_error,         # MAE: durchschnittlicher absoluter Fehler
    mean_squared_error,          # MSE: durchschnittlicher quadratischer Fehler
    root_mean_squared_error,     # RMSE: Wurzel aus MSE (gleiche Einheit wie Zielgröße)

    # Klassifikation
    accuracy_score,              # % korrekter Vorhersagen
    precision_score,             # von vorhergesagten Positiven: wie viele sind wirklich positiv
    recall_score,                # von allen echten Positiven: wie viele wurden erkannt
    f1_score,                    # harmonisches Mittel aus Precision und Recall
    classification_report,       # vollständiger Bericht mit allen Metriken pro Klasse
    confusion_matrix,            # Tabelle: tatsächlich vs. vorhergesagt
    roc_auc_score,               # Fläche unter der ROC-Kurve (für binäre Klassifikation)
)
```

### 7.7 R²-Score verstehen

R² (R-Quadrat) = der Anteil der Varianz in y, der durch das Modell erklärt wird.

```
R² = 1,0   →  perfektes Modell, erklärt alles
R² = 0,81  →  erklärt 81% der Varianz — gut
R² = 0,0   →  Modell ist so gut wie die Vorhersage des Mittelwerts
R² < 0     →  Modell ist schlechter als die Vorhersage des Mittelwerts — sehr schlecht
```

### 7.8 Accuracy verstehen

```
Accuracy = korrekte Vorhersagen / Gesamtvorhersagen

Beispiel: 85 korrekte von 100 → Accuracy = 0,85 = 85%
```

Accuracy ist irreführend bei **unausgeglichenen Datensätzen** (z.B. 99% der E-Mails sind kein Spam — immer "kein Spam" vorherzusagen gibt 99% Accuracy, ist aber nutzlos).

---

## 8. Modell-Evaluation

### 8.1 Overfitting vs. Underfitting

| Problem | Symptom | Lösung |
|---------|---------|--------|
| **Overfitting** | Hoher Train-Score, niedriger Test-Score | Mehr Daten, einfacheres Modell, Regularisierung |
| **Underfitting** | Niedriger Train-Score, niedriger Test-Score | Komplexeres Modell, mehr Features |
| **Gute Anpassung** | Train ≈ Test-Score, beide hoch | — |

```python
train_score = modell.score(X_train, y_train)
test_score = modell.score(X_test, y_test)

print(f"Train R²: {train_score:.3f}")
print(f"Test R²:  {test_score:.3f}")
# Großer Unterschied zwischen Train und Test → Overfitting
```

### 8.2 Konfusionsmatrix (Klassifikation)

```
                    Vorhergesagt: Nein    Vorhergesagt: Ja
Tatsächlich: Nein   True Negative         False Positive
Tatsächlich: Ja     False Negative        True Positive
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

### 8.3 Kreuzvalidierung

Statt einer einzigen Train/Test-Aufteilung teilt die Kreuzvalidierung die Daten K-mal auf und mittelt den Score. Zuverlässiger, besonders bei kleinen Datensätzen.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(modell, X, y, cv=5)  # 5-fache Kreuzvalidierung
print(f"CV-Scores: {scores}")
print(f"Mittelwert: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 9. Algorithmen-Referenz

### 9.1 Lineare Regression

**Was es macht:** Passt eine Gerade (oder Hyperebene) durch die Daten an, um einen kontinuierlichen Wert vorherzusagen.

**Statistischer Hintergrund:** Minimiert die Summe der quadratischen Residuen (OLS — Kleinste Quadrate).

**Formel:** `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

```python
from sklearn.linear_model import LinearRegression

modell = LinearRegression()
modell.fit(X_train, y_train)

print(modell.intercept_)   # β₀ (y-Achsenabschnitt)
print(modell.coef_)        # β₁, β₂, ... (Koeffizienten)
```

**Wann verwenden:** Basis-Regressionsmodell. Einfach, schnell, interpretierbar. Funktioniert gut, wenn die Beziehung wirklich linear ist.

### 9.2 K-Nearest Neighbors (KNN)

**Was es macht:** Klassifiziert (oder macht Regression für) einen neuen Punkt, indem es seine K nächsten Nachbarn in den Trainingsdaten betrachtet und eine Mehrheitsabstimmung (Klassifikation) oder den Durchschnitt (Regression) berechnet.

**Schritt für Schritt:**
1. Alle Trainingsdaten speichern (kein echtes "Training" findet statt)
2. Für einen neuen Punkt: Abstand zu allen Trainingspunkten berechnen
3. Die K nächsten Nachbarn finden (kleinste Abstände)
4. Klassifikation: Mehrheitsvoting der Nachbar-Labels → vorhergesagte Klasse
5. Regression: Durchschnitt der Nachbar-Werte → vorhergesagter Wert

**Distanzmaß (Standard: Euklidisch):**
```
d = √((x₁-x₁')² + (x₂-x₂')² + ... + (xₙ-xₙ')²)
```

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Klassifikation
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)          # Accuracy
knn.predict_proba(X_test)          # Klassenwahrscheinlichkeiten

# Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
knn_reg.score(X_test, y_test)      # R²-Score
```

**Wichtige Hyperparameter:**

| Parameter | Bedeutung | Standard |
|-----------|-----------|----------|
| `n_neighbors` | Anzahl der Nachbarn K | 5 |
| `metric` | Distanzfunktion | `'euclidean'` |
| `weights` | `'uniform'` = alle gleich; `'distance'` = nähere Nachbarn zählen mehr | `'uniform'` |

**K wählen:**
- Kleines K → komplexe Grenze, sensitiv für Rauschen → **Overfitting**-Risiko
- Großes K → glatte Grenze → **Underfitting**-Risiko
- Faustregel: K ≈ √(n_samples), immer **ungerade** bei Binärklassifikation um Gleichstand zu vermeiden

```python
# Bestes K durch Ausprobieren finden
best_k, best_score = 1, 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    if score > best_score:
        best_k, best_score = k, score
    print(f"K={k:2d}: {score:.3f}")

print(f"\nBestes K: {best_k} mit Score {best_score:.3f}")
```

**⚠️ Features immer vor KNN skalieren!**
Ohne Skalierung dominieren Features mit großem Wertebereich (z.B. Gehalt 0–100.000) Features mit kleinem Wertebereich (z.B. Alter 0–100) komplett.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
```

**Vorteile:** Einfach, keine Trainingszeit, funktioniert für Klassifikation und Regression, multi-class von Haus aus.  
**Nachteile:** Langsam bei Vorhersagen (berechnet alle Abstände), benötigt Skalierung, schwach bei hochdimensionalen Daten.

---

### 9.3 Logistische Regression

Trotz des Namens ist dies ein **Klassifikations**-Algorithmus. Er sagt die **Wahrscheinlichkeit** vorher, dass eine Probe zu einer Klasse gehört.

**Wie es funktioniert:**
Lineare Regression kann beliebige Zahlen ausgeben. Logistische Regression schleust diesen Wert durch die **Sigmoid-Funktion**, um ihn auf eine Wahrscheinlichkeit zwischen 0 und 1 zu bringen.

```
sigmoid(z) = 1 / (1 + e^(-z))    →  gibt immer einen Wert zwischen 0 und 1 zurück
```

Ist die vorhergesagte Wahrscheinlichkeit ≥ 0.5 → Klasse 1. Sonst → Klasse 0.

```python
from sklearn.linear_model import LogisticRegression

modell = LogisticRegression(max_iter=1000)   # max_iter erhöhen falls kein Konvergenz
modell.fit(X_train, y_train)

modell.predict(X_test)                        # vorhergesagte Klassen-Labels
modell.predict_proba(X_test)                  # [[prob_klasse0, prob_klasse1], ...]
modell.score(X_test, y_test)                  # Accuracy

print(modell.coef_)       # Koeffizienten (einer pro Feature und Klasse)
print(modell.intercept_)  # Achsenabschnitt
```

**Wichtige Hyperparameter:**

| Parameter | Bedeutung | Standard |
|-----------|-----------|----------|
| `C` | Kehrwert der Regularisierungsstärke. Kleines C = mehr Regularisierung | `1.0` |
| `max_iter` | Maximale Iterationen des Lösers | `100` |
| `multi_class` | Strategie für Multi-Klasse: `'auto'`, `'ovr'`, `'multinomial'` | `'auto'` |
| `solver` | Optimierungsalgorithmus: `'lbfgs'`, `'liblinear'`, `'saga'` | `'lbfgs'` |

**Multi-Klassen-Klassifikation:**
```python
# Funktioniert automatisch für mehr als 2 Klassen
modell = LogisticRegression(multi_class='auto', max_iter=1000)
modell.fit(X_train, y_train)
# predict_proba gibt eine Wahrscheinlichkeitsspalte pro Klasse zurück
```

**Evaluation:**
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
# Zeigt Precision, Recall, F1-Score für jede Klasse

cm = confusion_matrix(y_test, y_pred)
```

**Vorteile:** Schnell, interpretierbare Koeffizienten, gibt Wahrscheinlichkeiten aus, gutes starkes Basismodell.  
**Nachteile:** Setzt eine lineare Entscheidungsgrenze voraus — versagt wenn Klassen nicht linear trennbar sind.

**Wann verwenden:** Binäre oder Multi-Klassen-Klassifikation, wenn Wahrscheinlichkeiten benötigt werden, als schnelles starkes Basismodell.

---

### 9.4 Entscheidungsbäume (Decision Trees)

**Was es macht:** Lernt einen Baum aus Ja/Nein-Fragen über die Features, um Daten in Gruppen aufzuteilen und Vorhersagen zu treffen.

**Wie es funktioniert:**
1. Alle Daten starten an der Wurzel (Root)
2. Finde das Feature + den Schwellenwert, der die Daten am besten aufteilt (Unreinheit am meisten reduziert)
3. Rekursiv jeden Zweig weiter aufteilen bis eine Stoppbedingung erfüllt ist
4. Jedes Blatt (Leaf) enthält eine Vorhersage (häufigste Klasse oder Durchschnittswert)

**Aufteilungskriterium — Gini-Unreinheit (Standard für Klassifikation):**
Misst wie "gemischt" ein Knoten ist. Ein reiner Knoten (nur eine Klasse) hat Gini = 0.
```
Gini = 1 - Σ(pᵢ²)
```
Wobei pᵢ der Anteil der Klasse i im Knoten ist.

**Aufteilungskriterium — Entropie (Informationsgewinn):**
```
Entropie = -Σ(pᵢ · log₂(pᵢ))
```
Beide Kriterien liefern in der Praxis ähnliche Ergebnisse. Gini ist etwas schneller.

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Klassifikation
baum = DecisionTreeClassifier(
    max_depth=5,           # maximale Tiefe des Baums
    min_samples_split=2,   # Mindestanzahl Samples um einen Knoten aufzuteilen
    min_samples_leaf=1,    # Mindestanzahl Samples in einem Blattknoten
    criterion='gini'       # Kriterium: 'gini' oder 'entropy'
)
baum.fit(X_train, y_train)
baum.predict(X_test)
baum.score(X_test, y_test)

# Regression
baum_reg = DecisionTreeRegressor(max_depth=5)
baum_reg.fit(X_train, y_train)
baum_reg.score(X_test, y_test)   # R²-Score
```

**Wichtige Hyperparameter:**

| Parameter | Bedeutung | Tipp |
|-----------|-----------|------|
| `max_depth` | Maximale Tiefe des Baums. None = unbegrenzt | Mit 3–5 starten um Overfitting zu vermeiden |
| `min_samples_split` | Mindest-Samples um einen Knoten aufzuteilen | Höher = einfacherer Baum |
| `min_samples_leaf` | Mindest-Samples in einem Blattknoten | Höher = einfacherer Baum |
| `criterion` | `'gini'` oder `'entropy'` | Meist kein großer Unterschied |

**Baum visualisieren:**
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
plot_tree(
    baum,
    feature_names=X.columns,
    class_names=['Junior', 'Senior'],
    filled=True,          # Knoten nach Klasse einfärben
    rounded=True,
    fontsize=10
)
plt.show()
```

**Feature Importance:**
```python
# Wie viel jedes Feature zu den Aufteilungen beigetragen hat
importances = pd.Series(baum.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()
```

**Overfitting bei Decision Trees:**
Ohne Einschränkungen wächst ein Baum bis jedes Blatt rein ist (100% Train-Accuracy, miserable Test-Accuracy). `max_depth` ist die wichtigste Kontrolle.

```python
# Train- vs. Test-Score für verschiedene Tiefen vergleichen
for tiefe in range(1, 11):
    baum = DecisionTreeClassifier(max_depth=tiefe, random_state=42)
    baum.fit(X_train, y_train)
    print(f"Tiefe={tiefe:2d} | Train: {baum.score(X_train, y_train):.3f} | Test: {baum.score(X_test, y_test):.3f}")
```

**Vorteile:** Sehr gut interpretierbar (man kann den Baum zeichnen), kein Feature Scaling nötig, erfasst nicht-lineare Zusammenhänge, arbeitet mit numerischen und kategorischen Daten.  
**Nachteile:** Sehr anfällig für Overfitting (max_depth-Tuning nötig), instabil (kleine Datenänderungen → ganz anderer Baum), generell schwächer als Ensemble-Methoden.

**Wann verwenden:** Wenn Interpretierbarkeit entscheidend ist, als schnelles erstes Modell, als Baustein für Random Forests.

---

## 10. Schnell-Import-Cheatsheet

```python
# Grundlagen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Daten laden & aufteilen
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

# Algorithmen
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Metriken
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Feature-Auswahl
from sklearn.feature_selection import RFE
```

---

*Zuletzt aktualisiert: Mai 2026 · Wird kontinuierlich erweitert*
