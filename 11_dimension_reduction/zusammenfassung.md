# Zusammenfassung: Dimensionsreduktion mit PCA

## Warum Dimensionsreduktion?

```
Problem:  Datensatz mit 100 Features → zu viel, zu langsam, schwer visualisierbar
Lösung:   Dimensionsreduktion → weniger Features, aber wichtigste Info bleibt erhalten
```

**Wann nutzen?**
- Visualisierung von hochdimensionalen Daten (auf 2D/3D reduzieren)
- Modell-Performance verbessern (weniger irrelevante Features)
- Rechenzeit reduzieren
- Multikollinearität entfernen (korrelierte Features zusammenfassen)

---

## Was ist PCA?

**Principal Component Analysis (PCA)** = Hauptkomponentenanalyse

PCA findet die **Richtungen der größten Varianz** in den Daten und projiziert die Daten auf diese Richtungen.

```
Originaldaten:  13 Features (z.B. Wine Dataset)
Nach PCA:        2 Features (die 2 wichtigsten Richtungen)
Verlust:        nur ~35% der Varianz — 65% bleibt erhalten!
```

### Analogie:
Stell dir Daten als Wolke von Punkten im 3D-Raum vor. PCA findet die "Schattenwurfrichtung" die die meiste Information behält — wie ein Foto das eine 3D-Szene auf 2D reduziert.

---

## Wichtige Konzepte

### Varianz
```
Varianz = wie stark die Daten vom Mittelwert abweichen
Hohe Varianz → Feature enthält viele Informationen
Niedrige Varianz → Feature ist fast konstant → wenig nützlich
```

### Principal Components (Hauptkomponenten)
- **PC1**: Richtung der maximalen Varianz
- **PC2**: Richtung der zweitgrößten Varianz (orthogonal zu PC1)
- **PC3**: usw.

### Explained Variance Ratio
```python
pca.explained_variance_ratio_
# z.B. [0.36, 0.19, 0.11, ...]
# PC1 erklärt 36%, PC2 19%, PC3 11% der Gesamtvarianz
```

---

## PCA Schritt-für-Schritt

### Algorithmus:
```
1. Daten standardisieren (StandardScaler!)
2. Kovarianzmatrix berechnen
3. Eigenwerte & Eigenvektoren berechnen (Eigen Decomposition)
4. Eigenvektoren nach Eigenwert sortieren (größter zuerst)
5. Top-N Eigenvektoren auswählen
6. Daten auf neuen Raum projizieren
```

### In sklearn (einfacher Weg):
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA auf 2 Komponenten
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Wie viel Varianz erklären diese 2 Komponenten?
print(pca.explained_variance_ratio_)
# [0.73, 0.23] → 96% der Varianz erhalten!
```

---

## Wie viele Komponenten wählen?

### Methode 1: Feste Anzahl
```python
pca = PCA(n_components=2)   # genau 2 Komponenten
```

### Methode 2: Varianz-Schwelle (empfohlen!)
```python
pca = PCA(n_components=0.80)  # so viele Komponenten wie nötig für 80% Varianz
```

### Methode 3: Scree Plot / Explained Variance Plot
```python
pca_full = PCA()
pca_full.fit(X_scaled)

plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Anzahl Komponenten')
plt.ylabel('Kumulierte erklärte Varianz')
plt.axhline(y=0.80, color='red', linestyle='--', label='80% Schwelle')
plt.show()
```

---

## PCA + ML Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(0.80)),               # 80% Varianz behalten
    ('model', DecisionTreeClassifier())
])

score = pipe.fit(X_train, y_train).score(X_test, y_test)
print(f"Accuracy: {score:.2%}")
```

**Wichtig:** PCA kommt NACH dem Scaler, VOR dem Modell.

---

## PCA für Clustering + Visualisierung

```python
# Clustering auf 13 Features → schwer visualisierbar
# Lösung: PCA auf 2D reduzieren, dann plotten

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Cluster in PCA-reduziertem Raum')
plt.show()
```

---

## Vor- und Nachteile

| | PCA |
|--|--|
| ✅ | Reduziert Dimensionen effektiv |
| ✅ | Entfernt Multikollinearität |
| ✅ | Gut für Visualisierung |
| ✅ | Beschleunigt Training |
| ❌ | Features nicht mehr interpretierbar (PC1 hat keine klare Bedeutung) |
| ❌ | Nur lineare Zusammenhänge |
| ❌ | Skalierung notwendig! |

---

## Vergleich: Feature Selection vs. PCA

```
Feature Selection:  Wählt VORHANDENE Features aus → bleibt interpretierbar
PCA (Extraction):   Erstellt NEUE Features → verliert Interpretierbarkeit,
                    gewinnt aber optimale Varianzerfassung
```

---

## Wichtige Imports

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
```
