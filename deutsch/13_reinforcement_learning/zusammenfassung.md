# Zusammenfassung: Reinforcement Learning (RL)

## Was ist Reinforcement Learning?

```
RL = Lernen durch Ausprobieren und Feedback

Ein Agent interagiert mit einer Umgebung:
  → macht eine Aktion
  → bekommt eine Belohnung (positiv oder negativ)
  → passt sein Verhalten an

Ziel: Belohnung über Zeit maximieren
```

**Alltagsanalogie:** Ein Hund lernt durch Belohnung (Leckerli) und Strafe — kein Lehrer gibt ihm die richtige Antwort direkt.

---

## Die 5 Kernkomponenten

| Komponente | Beschreibung | Beispiel (Roboter) |
|-----------|-------------|-------------------|
| **Agent** | Der Lernende | Der Roboter |
| **Environment** | Die Welt | Das Labyrinth |
| **State (s)** | Aktuelle Situation | Position des Roboters |
| **Action (a)** | Mögliche Züge | Hoch/Runter/Links/Rechts |
| **Reward (r)** | Feedback | +1 für Ziel, -0.01 pro Schritt |

---

## Der RL-Kreislauf

```
        ┌─────────────────────────────────────┐
        │                                     │
        ▼                                     │
   [State s_t] → Agent wählt Action a_t       │
                      │                       │
                      ▼                       │
              Environment reagiert            │
                      │                       │
              ┌───────┴───────┐               │
              ▼               ▼               │
        Reward r_t      New State s_{t+1} ────┘
```

**Der Agent passt nach jedem Schritt seine Strategie (Policy) an.**

---

## Vergleich mit anderen ML-Methoden

| | Supervised | Unsupervised | Reinforcement |
|--|:---:|:---:|:---:|
| **Labels** | ✅ Ja | ❌ Nein | ❌ Nein |
| **Feedback** | Direkte Antwort | Keins | Verzögerte Belohnung |
| **Lernt** | Imitation | Muster | Entscheidungen |
| **Daten** | Statisch | Statisch | Dynamisch (Agent erzeugt sie) |
| **Beispiel** | Spam-Filter | Clustering | Schach spielen |

---

## Markov Decision Process (MDP)

RL-Probleme werden formal als **MDP** modelliert:

```
MDP = (S, A, P, R, γ)

S  = alle möglichen Zustände
A  = alle möglichen Aktionen
P  = Übergangswahrscheinlichkeit P(s'|s,a)
R  = Belohnungsfunktion R(s,a,s')
γ  = Discount-Faktor (0 ≤ γ ≤ 1)
```

### Discount-Faktor γ:
```
γ = 0.9: Belohnungen in der Zukunft sind weniger wert
         r_jetzt = 1.0   vs.   r_in_3_Schritten = 0.9³ = 0.729

γ → 0: Agent denkt kurzfristig (nur nächster Schritt)
γ → 1: Agent denkt langfristig (alle zukünftigen Schritte gleich)
```

---

## Value Functions

```
V(s)    = State Value Function
          → Wie gut ist es, in Zustand s zu sein?
          → Erwartete Gesamtbelohnung ab s

Q(s,a)  = Action Value Function (Q-Function)
          → Wie gut ist Aktion a in Zustand s?
          → Basis für Q-Learning
```

### Bellman-Gleichung:
```
V(s) = max_a Σ P(s'|s,a) × [R(s,a,s') + γ × V(s')]

"Der Wert eines Zustands = beste Aktion × (sofortige Belohnung + abgezinkte Zukunft)"
```

---

## Exploration vs. Exploitation

```
Das fundamentale Dilemma:

Exploitation:  Nutze bekannte gute Strategie  → sicher, aber verpasse Besseres
Exploration:   Probiere Neues aus             → riskant, aber könnte besser sein

Lösung: ε-Greedy
  mit Wahrscheinlichkeit ε → zufällige Aktion (Exploration)
  sonst                   → beste bekannte Aktion (Exploitation)
```

```python
import random

epsilon = 0.1  # 10% Exploration

def choose_action(q_values, epsilon):
    if random.random() < epsilon:
        return random.choice(range(len(q_values)))  # zufällig
    else:
        return q_values.index(max(q_values))         # beste bekannte
```

---

## Algorithmen-Überblick

### Temporal Difference (TD) Methods:
```
Q-Learning:  Lernt optimale Q-Werte (off-policy)
SARSA:       Lernt Q-Werte für aktuelle Policy (on-policy)
```

### Deep RL:
```
DQN (Deep Q-Network): Neuronales Netz approximiert Q-Funktion
                      → für hochdimensionale States (z.B. Pixel)
```

### Actor-Critic:
```
Actor:  Lernt welche Aktion gewählt werden soll (Policy)
Critic: Bewertet wie gut die Aktion war (Value Function)
→ Kombination aus beidem: stabiler als reines Q-Learning
```

---

## Reale Anwendungen

| Bereich | Anwendung |
|---------|-----------|
| **Gaming** | AlphaGo, Dota 2, StarCraft II |
| **Robotik** | Laufen lernen, Greifarm |
| **Empfehlungen** | Dynamische Anpassung von Feeds |
| **Finanzen** | Handelsalgorithmen |
| **LLMs** | RLHF (ChatGPT lernt durch menschliches Feedback) |

---

## Wichtige Begriffe

```
Policy π:         Strategie des Agents (s → a)
Episode:          Eine komplette Runde (Start bis Ende)
Trajectory:       Sequenz von (s, a, r, s') in einer Episode
On-Policy:        Agent lernt von eigener aktueller Strategie
Off-Policy:       Agent lernt von gespeicherten Erfahrungen
Experience Replay: Zufällige alte Erfahrungen nochmal verwenden (DQN)
RLHF:             Reinforcement Learning from Human Feedback
```
