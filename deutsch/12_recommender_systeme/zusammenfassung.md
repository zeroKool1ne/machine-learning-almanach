# Zusammenfassung: Recommender Systeme

## Was ist ein Empfehlungssystem?

```
Ziel: Dem Nutzer Items vorschlagen, die er wahrscheinlich mag

Beispiele:
- Netflix: "Weil du X gesehen hast..."
- Amazon: "Kunden kauften auch..."
- Spotify: Dein Discover Weekly
```

---

## Die 3 Haupttypen im Überblick

| Typ | Basis | Beispiel |
|-----|-------|---------|
| **Content-Based Filtering** | Item-Eigenschaften | "Du magst Thriller → hier andere Thriller" |
| **Collaborative Filtering** | Nutzer-Verhalten | "Leute wie du mögen auch..." |
| **Hybrid** | Kombination | Netflix, Amazon |

---

## 1. Content-Based Filtering

```
Idee: Items mit ähnlichen EIGENSCHAFTEN empfehlen

Workflow:
1. Items als Feature-Vektor darstellen (Genre, Regisseur, Schauspieler...)
2. Nutzerprofil = Durchschnitt der bewerteten Items
3. Ähnlichkeit zwischen Items berechnen (Cosine Similarity)
4. Ähnlichste Items empfehlen
```

### Cosine Similarity:
```
cos(A, B) = (A · B) / (|A| × |B|)

Bereich: 0 bis 1  →  1 = identisch, 0 = völlig unterschiedlich
```

### Code:
```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Features vorbereiten
features = pd.get_dummies(df[['genre', 'director', 'actor1']])
scaler = StandardScaler()
features['release_year'] = scaler.fit_transform(features[['release_year']])

# Ähnlichkeitsmatrix berechnen
cosine_sim = cosine_similarity(features, features)

# Empfehlung für einen Film
def recommend(title, cosine_sim=cosine_sim):
    idx = df.index[df['title'] == title][0]
    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:6]]   # Top 5 (ohne sich selbst)
    return df['title'].iloc[top_indices]
```

### Stärken & Schwächen:
```
✅ Keine anderen Nutzer nötig
✅ Transparente Empfehlungen ("weil du X mochtest")
✅ Kaltstartproblem für neue Items lösbar

❌ Filter Bubble (nur Ähnliches wird empfohlen)
❌ Benötigt detaillierte Item-Metadaten
❌ Findet keine überraschenden Entdeckungen
```

---

## 2. Collaborative Filtering

```
Idee: Nutzer mit ähnlichem Geschmack haben dieselben Vorlieben

"Nutzer A und B haben 10 Filme ähnlich bewertet
 → was B gut fand, wird A wahrscheinlich auch mögen"
```

### User-Based vs. Item-Based:
```
User-Based:  Ähnliche NUTZER finden → deren Items empfehlen
Item-Based:  Ähnliche ITEMS finden  → "wenn du X magst, dann Y"
```

### Matrix Factorization (SVD):

```
User-Item-Matrix:
             Film1  Film2  Film3  Film4
  Nutzer1:     5      3      0      1
  Nutzer2:     4      0      4      1
  Nutzer3:     1      1      0      5

Problem: Sehr sparse (viele Nullen = nicht bewertet)
Lösung: SVD zerlegt die Matrix in latente Faktoren
```

```python
from scipy.sparse.linalg import svds
import numpy as np

# User-Item-Matrix erstellen
user_item = df.pivot_table(index='userId', columns='title',
                            values='centered_rating', fill_value=0)

# SVD anwenden
U, sigma, Vt = svds(np.matrix(user_item), k=10)  # k = latente Faktoren
sigma = np.diag(sigma)

# Latente Faktoren für Items und Nutzer
items_df = pd.DataFrame(Vt, columns=user_item.columns).T
user_df  = pd.DataFrame(U)

# Ähnlichkeit berechnen
from sklearn.metrics.pairwise import cosine_similarity
item_similarity = cosine_similarity(items_df, items_df)
user_similarity = cosine_similarity(user_df, user_df)
```

### Rating-Zentrierung (wichtig!):
```python
# Nutzer-Bias entfernen: manche User geben generell höhere/niedrigere Noten
df['mean_rating']     = df.groupby('userId')['rating'].transform('mean')
df['centered_rating'] = df['rating'] - df['mean_rating']
```

### Stärken & Schwächen:
```
✅ Findet überraschende Entdeckungen
✅ Keine Item-Metadaten nötig
✅ Skaliert gut mit Daten

❌ Kaltstartproblem: neue Nutzer/Items haben keine History
❌ Sparsity: meiste Nutzer bewerten nur wenige Items
❌ Popularity Bias: beliebte Items werden öfter empfohlen
```

---

## 3. Surprise Library — ML für Recommender

```python
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Daten laden
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Train/Test Split
trainset, testset = train_test_split(data, test_size=0.25)

# SVD Modell trainieren
model = SVD()
model.fit(trainset)

# Vorhersagen
predictions = model.test(testset)
accuracy.rmse(predictions)

# Einzelne Vorhersage
pred = model.predict(user_id=7, iid=1)
print(f"Vorhergesagtes Rating: {pred.est:.2f}")
```

---

## Dimensionsreduktion mit TruncatedSVD

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=5, random_state=42)
latent_matrix = svd.fit_transform(features)   # Reduzierte Darstellung

# Cosine Similarity auf latenten Faktoren
cosine_sim_svd = cosine_similarity(latent_matrix, latent_matrix)
```

```
Vorteil: Findet verborgene Muster ("latente Faktoren")
         z.B. "Action-Fan" oder "Indie-Liebhaber" — ohne explizite Labels
```

---

## Vergleich der Methoden

| | Content-Based | Collaborative | Hybrid |
|--|--------------|---------------|--------|
| **Daten** | Item-Features | User-Ratings | Beides |
| **Kaltstart (User)** | ✅ kein Problem | ❌ Problem | ✅ |
| **Kaltstart (Item)** | ✅ kein Problem | ❌ Problem | ✅ |
| **Überraschungen** | ❌ selten | ✅ möglich | ✅ |
| **Skalierung** | Mittel | Gut | Gut |

---

## Wichtige Konzepte kurz erklärt

```
Sparse Matrix:      Matrix mit vielen Nullen (typisch bei Ratings)
Latente Faktoren:   Verborgene Eigenschaften (z.B. "Actionlastigkeit")
Kaltstartproblem:   Neuer Nutzer/Item hat keine History → keine Empfehlung möglich
Filter Bubble:      Nutzer bekommt nur ähnliche Inhalte → Vielfalt sinkt
RMSE:               Fehlermaß für vorhergesagte vs. echte Ratings
```
