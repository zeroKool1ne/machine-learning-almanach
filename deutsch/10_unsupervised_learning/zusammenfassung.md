# Zusammenfassung: Unsupervised Learning — Clustering

## Supervised vs. Unsupervised

```
Supervised:    Labels vorhanden → Modell lernt anhand von Beispielen
               z.B. "Das ist eine Katze" → Modell lernt Katzen erkennen

Unsupervised:  KEINE Labels → Modell findet selbst Strukturen/Gruppen
               z.B. "Hier sind 10.000 Kunden — finde ähnliche Gruppen!"
```

**Wann Unsupervised?**
- Kundensegmentierung
- Anomalie-Erkennung (Betrug)
- Genexpressionsanalyse
- Empfehlungssysteme

---

## Clustering-Methoden im Überblick

| Methode | Prinzip | Wann? |
|---------|---------|-------|
| **K-Means** | Punkte → nächster Centroid | Runde, gleich große Cluster |
| **DBSCAN** | Dichte-basiert | Unregelmäßige Formen, Ausreißer |
| **GMM** | Wahrscheinlichkeitsverteilungen | Überlappende Cluster |
| **Hierarchical** | Baumstruktur | Wenn K unbekannt |

---

## K-Means Algorithmus

### Ablauf:
```
1. K festlegen (Anzahl Cluster)
2. K zufällige Punkte als Centroids wählen
3. Wiederhole bis Konvergenz:
   a. Jeden Punkt dem nächsten Centroid zuordnen
   b. Centroids neu berechnen (Mittelpunkt der Gruppe)
4. Stopp wenn Centroids sich nicht mehr bewegen
```

### Code:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,      # K = Anzahl Cluster
    random_state=42,
    n_init=10          # 10x mit verschiedenen Starts → bestes Ergebnis
)
kmeans.fit(X)

labels = kmeans.labels_          # Clusterzugehörigkeit pro Punkt
centroids = kmeans.cluster_centers_  # Koordinaten der Centroids
inertia = kmeans.inertia_        # Güte der Lösung (niedriger = besser)
```

---

## Inertia (Within-Cluster Sum of Squares)

```
Inertia = Summe der quadratischen Abstände jedes Punktes zu seinem Centroid

Niedrige Inertia = kompakte, enge Cluster ✅
Hohe Inertia    = lockere, weitläufige Cluster ❌

ABER: Inertia sinkt immer wenn K steigt → Elbow Method!
```

---

## Die Elbow Method — Optimales K finden

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Anzahl Cluster (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

```
Inertia
  |  \
  |   \
  |    \___
  |        \____________________
  +--+--+--+--+--+--+--+--+--→ K
     1  2  3  4  5  6  7  8

        ↑ Elbow hier → optimales K = 3
```

**Problem:** Manchmal kein klarer Elbow → Domain Knowledge nötig!

---

## K-Means++ Initialisierung

**Problem mit zufälligem Start:**
```
Schlechte Initialisierung → schlechte Cluster → falsches Ergebnis!
```

**Lösung — K-Means++:**
```
1. Ersten Centroid zufällig wählen
2. Nächste Centroids: Punkte mit GROSSEM Abstand zu bestehenden 
   Centroids bekommen höhere Wahrscheinlichkeit
3. → Centroids starten weit auseinander → stabilere Konvergenz
```

```python
# Standard in sklearn — automatisch aktiviert!
kmeans = KMeans(n_clusters=3, init='k-means++')
```

---

## DBSCAN — Dichte-basiertes Clustering

### Parameter:
```
eps         = maximaler Abstand zwischen zwei Punkten (Nachbarschaft)
min_samples = Mindestanzahl Punkte für einen Core Point
```

### Punkt-Typen:
```
Core Point:   > min_samples Nachbarn in Radius eps → Kernpunkt eines Clusters
Border Point: < min_samples Nachbarn, aber Nachbar eines Core Points
Noise Point:  weder Core noch Border → Ausreißer (Label = -1)
```

### Code:
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

labels = dbscan.labels_
# -1 = Noise/Ausreißer
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
```

### Wann DBSCAN?
- Cluster haben unregelmäßige Formen
- Ausreißer sollen erkannt werden
- K ist unbekannt (DBSCAN braucht kein K!)

---

## Gaussian Mixture Model (GMM)

### Hard vs. Soft Clustering:
```
K-Means (Hard):  Jeder Punkt gehört GENAU EINEM Cluster (0 oder 1)
GMM (Soft):      Jeder Punkt gehört JEDEM Cluster mit einer Wahrscheinlichkeit

Punkt X:  Cluster 1: 80% | Cluster 2: 15% | Cluster 3: 5%
```

### Code:
```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

labels = gmm.predict(X)                    # harte Zuweisung
probs = gmm.predict_proba(X)               # weiche Wahrscheinlichkeiten
```

---

## Hierarchical Clustering

```
Agglomerative (Bottom-Up):    Jeder Punkt = eigener Cluster
                               → iterativ die ähnlichsten zusammenführen
                               → Dendrogram entsteht

Divisive (Top-Down):          Alle Punkte = ein Cluster
                               → iterativ aufteilen
```

```python
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3)
labels = hc.fit_predict(X)
```

---

## Clustering Evaluation — Silhouette Score

Da es keine Labels gibt → andere Metriken!

```
a = mittlerer Abstand zum eigenen Cluster (Intra-Cluster)
b = mittlerer Abstand zum nächsten fremden Cluster (Inter-Cluster)

Silhouette Score = (b - a) / max(a, b)

Bereich: -1 bis +1
  +1 = perfekte Trennung ✅
   0 = Punkte auf Clustergrenze
  -1 = falsch zugeordnet ❌
```

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.4f}")

# Optimales K mit Silhouette finden:
scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    scores.append(silhouette_score(X, labels))

best_k = range(2, 11)[scores.index(max(scores))]
print(f"Bestes K: {best_k}")
```

---

## Wichtig: Skalierung vor Clustering!

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DANN clustern!
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
```

K-Means verwendet Abstände → ohne Skalierung dominieren große Spalten!
