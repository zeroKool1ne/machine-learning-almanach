# Machine Learning Bible 🧠
> A complete reference guide — from raw data to trained model.  
> Built to be kept open while working. Updated continuously.

---

## Table of Contents

1. [Types of Machine Learning](#1-types-of-machine-learning)
2. [Classification vs. Regression](#2-classification-vs-regression)
3. [Data Structure](#3-data-structure)
4. [Full ML Workflow](#4-full-ml-workflow)
5. [Data Cleaning & Preprocessing](#5-data-cleaning--preprocessing)
6. [Working with SQL Data](#6-working-with-sql-data)
7. [Scikit-Learn — All Methods Explained](#7-scikit-learn--all-methods-explained)
8. [Model Evaluation](#8-model-evaluation)
9. [Algorithms Reference](#9-algorithms-reference)
10. [Quick Import Cheatsheet](#10-quick-import-cheatsheet)

---

## 1. Types of Machine Learning

Machine Learning is divided into three main paradigms depending on how the algorithm learns from data.

### 1.1 Supervised Learning
The algorithm learns from **labeled data** — meaning every training example has an input (features) and a known correct output (label/target).

**How it works:** The model finds a mapping function `f(X) → y` that generalizes to unseen data.

**Use it when:** You have historical data with known outcomes and want to predict future outcomes.

| Algorithm | Task | Example Use Case |
|-----------|------|-----------------|
| Linear Regression | Regression | Predict house prices |
| Logistic Regression | Classification | Spam detection |
| KNN | Both | Customer segmentation |
| Decision Tree | Both | Loan approval |
| Random Forest | Both | Medical diagnosis |
| SVM | Both | Image classification |

### 1.2 Unsupervised Learning
The algorithm learns from **unlabeled data** — there are no correct answers provided. The model finds hidden patterns or structures on its own.

**Use it when:** You want to discover structure in data without a predefined target.

| Algorithm | Task | Example Use Case |
|-----------|------|-----------------|
| K-Means | Clustering | Customer grouping |
| DBSCAN | Clustering | Anomaly detection |
| PCA | Dimensionality reduction | Feature compression |
| Autoencoders | Feature learning | Image compression |

### 1.3 Reinforcement Learning
An agent learns by **interacting with an environment**. It receives rewards for good actions and penalties for bad ones, and learns to maximize total reward over time.

**Use it when:** You have a sequential decision-making problem where the agent can explore and get feedback.

Examples: Game-playing AI (chess, Go), robot control, recommendation systems.

### 1.4 Semi-Supervised & Self-Supervised Learning
- **Semi-supervised:** Mix of labeled and unlabeled data. Common when labeling is expensive.
- **Self-supervised:** Model generates its own labels from the data (e.g., predict the next word in a sentence — how LLMs are trained).

---

## 2. Classification vs. Regression

The single most important question before building any model: **"What kind of output am I predicting?"**

### Classification
Predicts a **discrete category** — the output belongs to one of a fixed set of classes.

- **Binary classification:** 2 classes (Yes/No, Spam/Not Spam, 0/1)
- **Multi-class classification:** 3+ classes (dog/cat/bird, A/B/C/D)
- **Multi-label classification:** Multiple classes can be true at once (a movie can be Action AND Comedy)

**Output:** A class label (and often a probability)

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Examples:**
- Will this customer churn? → Yes/No
- What digit is this image? → 0–9
- What language is this text? → EN/DE/FR/...

### Regression
Predicts a **continuous numeric value** — the output can be any number on a scale.

**Output:** A number (e.g., 43,200.50)

**Evaluation metrics:** R² Score, MAE, MSE, RMSE

**Examples:**
- What will this house sell for? → €382,000
- What temperature will it be tomorrow? → 22.4°C
- What salary will this person earn? → $67,400

### Decision Guide

```
Is the output a category?     → Classification
Is the output a number?       → Regression
```

> **Tip:** Some problems can be framed both ways. "Will the price go up?" is classification. "By how much will it go up?" is regression.

---

## 3. Data Structure

### 3.1 The Standard Format: X and y

Scikit-learn expects data in a specific format:

```
X = Feature matrix       shape: (n_samples, n_features)
y = Target vector        shape: (n_samples,)
```

| Term | Also called | Meaning |
|------|-------------|---------|
| **Sample** | Row, observation, instance | One data point (e.g., one person) |
| **Feature** | Column, predictor, variable, attribute | One measurable property (e.g., age) |
| **Target** | Label, response, outcome, y | What we want to predict |
| **n_samples** | Number of rows | How many observations we have |
| **n_features** | Number of columns | How many input variables we have |

### 3.2 NumPy Arrays vs. Pandas DataFrames

Scikit-learn natively uses **NumPy arrays**, but **pandas DataFrames are also accepted** in most cases.

```python
import numpy as np
import pandas as pd

# NumPy array
X = np.array([[1, 2], [3, 4], [5, 6]])   # shape (3, 2)

# Pandas DataFrame — also works
df = pd.DataFrame({'age': [25, 30, 35], 'salary': [40000, 60000, 80000]})
X = df[['age', 'salary']]  # still works with sklearn
```

### 3.3 When to use what

| Situation | Use |
|-----------|-----|
| Raw data exploration | Pandas DataFrame |
| Feeding into sklearn | Both work — DataFrame preferred for readability |
| Need pure speed / math ops | NumPy array |
| After encoding | NumPy array (from `.values` or `np.concatenate`) |

---

## 4. Full ML Workflow

This is the complete end-to-end process every ML project follows.

```
1. Define the problem
2. Load the data
3. Explore the data (EDA)
4. Clean & preprocess the data
5. Split into train/test sets
6. Choose and train a model
7. Evaluate the model
8. Improve the model
9. (Deploy)
```

### Step 1: Define the Problem
Before touching any code, answer:
- What am I trying to predict?
- Is it classification or regression?
- What data do I have?
- What does "success" look like? (which metric matters)

### Step 2: Load the Data

```python
import pandas as pd

# From CSV
df = pd.read_csv('data.csv')

# From SQL (see Section 6)
df = pd.read_sql('SELECT * FROM table', engine)

# Built-in sklearn datasets (for practice)
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
```

### Step 3: Explore the Data (EDA)

```python
df.shape            # (rows, columns)
df.dtypes           # data types of each column
df.head()           # first 5 rows
df.tail()           # last 5 rows
df.describe()       # statistics: mean, std, min, max, quartiles
df.info()           # overview: dtypes + non-null counts
df.isnull().sum()   # missing values per column
df.duplicated().sum() # number of duplicate rows
df['col'].value_counts()  # frequency table for a column
df['col'].unique()  # all unique values
```

**Visualization:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df.select_dtypes(include='number'))  # all numeric relationships

plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
```

**What to look for:**
- Are there missing values?
- Are there outliers? (check min/max in describe())
- Are columns the wrong type? (object instead of float)
- Are there correlations between features?
- Is the target variable balanced (for classification)?

### Step 4: Clean & Preprocess
→ See Section 5 for full details.

### Step 5: Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% goes to test set
    random_state=42     # fixed seed → reproducible split
)
```

**Why split?**
- The model trains on `X_train` and `y_train`
- We evaluate it on `X_test` and `y_test` — data it has **never seen**
- Without this, you'd just be measuring how well it memorized training data (overfitting)

**Common splits:** 80/20, 75/25, 70/30

### Step 6: Train the Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()      # 1. Initialize
model.fit(X_train, y_train)     # 2. Train
```

This is the same pattern for **every** sklearn model. Only the import and class name changes.

### Step 7: Evaluate

```python
y_pred = model.predict(X_test)  # Generate predictions

# Regression
from sklearn.metrics import r2_score, mean_absolute_error
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)

# Classification
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Step 8: Improve

Options for improving a model:
- More/better data
- Feature engineering (create new features from existing ones)
- Try different algorithms
- Tune hyperparameters
- Use regularization to fight overfitting
- Change train/test split ratio

---

## 5. Data Cleaning & Preprocessing

### 5.1 Fixing Data Types

Many datasets import columns as `object` (string) when they should be numeric. This often happens because of non-numeric values like `'T'` (trace), `'-'`, or `'N/A'`.

```python
# Convert one column
df['column'] = pd.to_numeric(df['column'], errors='coerce')
# errors='coerce' → converts non-convertible values to NaN instead of crashing

# Convert multiple columns at once
cols_to_fix = ['col1', 'col2', 'col3']
df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors='coerce')

# Check result
df.dtypes
```

### 5.2 Handling Missing Values

**Step 1 — Find missing values:**
```python
df.isnull().sum()                     # count per column
df.isnull().sum() / len(df)           # percentage per column
df[df.isnull().any(axis=1)]           # all rows with at least one missing value
```

**Step 2 — Decide strategy:**

| Situation | Strategy |
|-----------|----------|
| Column has >10% missing | Drop the column |
| Row has many missing values | Drop the row |
| Numerical data, time series | Interpolate |
| Numerical data, random | Fill with mean or median |
| Categorical data | Fill with mode or 'Unknown' |

```python
# Drop column
df.drop(columns=['bad_column'], inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Fill with mean
df['col'].fillna(df['col'].mean(), inplace=True)

# Fill categorical with a label
df['col'].fillna('Unknown', inplace=True)

# Linear interpolation (great for time series / weather data)
df_fixed = df.interpolate()
```

### 5.3 Encoding Categorical Variables

Machine learning algorithms work with **numbers only**. Text categories must be converted.

#### One-Hot Encoding (pd.get_dummies) — RECOMMENDED

Creates a new binary column for each category. Use when there is no natural order between categories.

```python
# Before: Experience column has values 'Junior', 'Senior'
# After: Two columns — Experience_Junior (0/1) and Experience_Senior (0/1)

df = pd.get_dummies(df, columns=['Experience', 'Gender', 'Daltonic'])

# Convert boolean True/False to integer 0/1
df = df.apply(lambda x: x.astype(int) if x.dtype == bool else x)
```

**Why not just use 0 and 1 for Junior/Senior directly?**
Because that would imply "Senior = 2 × Junior" which is mathematically wrong. One-hot encoding treats them as completely separate categories.

#### Label Encoding — use only for ordinal data

When categories have a natural order (e.g., Small < Medium < Large):

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])
# Small→0, Medium→1, Large→2
```

#### OneHotEncoder from sklearn

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
encoded = enc.fit_transform(df[['Experience']]).toarray()
```

### 5.4 Feature Scaling

Many algorithms (KNN, SVM, Neural Networks) are sensitive to the **scale** of features. If one feature ranges 0–1 and another ranges 0–1,000,000, the large-scale feature dominates.

**Standardization (StandardScaler)** — transforms to mean=0, std=1. Use this by default.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train, transform train
X_test_scaled = scaler.transform(X_test)         # only transform test (never fit!)
```

**Min-Max Scaling (MinMaxScaler)** — scales to range [0, 1].

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

> **Important rule:** Always fit the scaler on **training data only**, then use it to transform test data. If you fit on all data, you leak information from the test set.

### 5.5 Removing Duplicates

```python
df.duplicated().sum()           # how many duplicates?
df.drop_duplicates(inplace=True)
```

### 5.6 Dropping Columns

```python
# Drop one column
df.drop(columns=['useless_col'], inplace=True)

# Drop multiple
df.drop(columns=['col1', 'col2', 'col3'], inplace=True)
```

### 5.7 Complete Preprocessing Pipeline

```python
# 1. Fix column types
wrong_cols = ['col1', 'col2']
df[wrong_cols] = df[wrong_cols].apply(pd.to_numeric, errors='coerce')

# 2. Handle missing values
df.drop(columns=['col_with_too_many_nans'], inplace=True)
df = df.interpolate()

# 3. Encode categoricals
df['text_col'].fillna('Unknown', inplace=True)
df = pd.get_dummies(df, columns=['cat1', 'cat2'])
df = df.apply(lambda x: x.astype(int) if x.dtype == bool else x)

# 4. Split X and y
X = df.drop(columns=['target'])
y = df['target']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale features (if needed)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 6. Working with SQL Data

### 6.1 Setting Up the MySQL Connection

```python
from sqlalchemy import create_engine
import pandas as pd

# Format: 'mysql+pymysql://username:password@host/database_name'
engine = create_engine('mysql+pymysql://root:yourpassword@localhost/your_database')
```

**Install required packages if needed:**
```bash
pip install sqlalchemy pymysql
```

### 6.2 Loading a Full Table

```python
df = pd.read_sql('SELECT * FROM your_table', engine)
df.head()
```

### 6.3 Loading with a Query

```python
query = """
    SELECT date, temperature, humidity
    FROM weather
    WHERE year = 2023
    ORDER BY date
"""
df = pd.read_sql(query, engine)
```

### 6.4 Workflow: MySQL → Python → ML Model

```
1. Open MySQL Workbench
2. Run your .sql file to create and populate the database
3. In Python: create engine with create_engine()
4. Load table into DataFrame with pd.read_sql()
5. Proceed with normal ML workflow (clean, encode, split, train)
```

### 6.5 Useful SQL Queries Before Loading

Run these in MySQL Workbench to understand your data first:

```sql
-- How many rows?
SELECT COUNT(*) FROM table_name;

-- What's the max/min of a column?
SELECT MAX(column), MIN(column) FROM table_name;

-- Average of a column
SELECT AVG(column) FROM table_name;

-- Top 10 rows sorted by a column
SELECT * FROM table_name ORDER BY column DESC LIMIT 10;

-- Filter: rows where condition is true
SELECT * FROM table_name WHERE column > 100;

-- Casting text columns to numbers (needed when column type is VARCHAR)
SELECT CAST(column AS SIGNED) FROM table_name;
```

---

## 7. Scikit-Learn — All Methods Explained

### 7.1 The Universal API

Every single sklearn model follows the same interface:

```python
model.fit(X_train, y_train)      # Train the model on data
model.predict(X)                  # Make predictions
model.predict_proba(X)            # Predict class probabilities (classifiers)
model.score(X, y)                 # Evaluate: R² for regression, accuracy for classification
model.transform(X)                # Transform data (preprocessing objects)
model.fit_transform(X)            # Fit + transform in one step
```

### 7.2 Model Initialization Parameters (Hyperparameters)

These are set **before** training. They control how the model learns.

```python
# KNN
KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')

# Linear Regression
LinearRegression(fit_intercept=True)

# Decision Tree
DecisionTreeClassifier(max_depth=5, min_samples_split=2)
```

### 7.3 Datasets (for Practice)

```python
from sklearn.datasets import (
    load_iris,          # 150 flowers, 4 features, 3 classes — classic classification
    load_diabetes,      # 442 patients, 10 features — regression
    load_digits,        # 8x8 pixel images of digits — image classification
    load_wine,          # 178 wines, 13 features, 3 classes
    load_breast_cancer, # 569 tumors, 30 features — binary classification
    make_classification, # generate a random classification dataset
    make_regression      # generate a random regression dataset
)

data = load_iris()
X = data.data        # feature matrix
y = data.target      # labels
print(data.DESCR)    # full description of the dataset
print(data.keys())   # available attributes
```

### 7.4 Preprocessing

```python
from sklearn.preprocessing import (
    StandardScaler,      # standardize: mean=0, std=1
    MinMaxScaler,        # scale to [0, 1]
    LabelEncoder,        # encode labels as integers
    OneHotEncoder,       # one-hot encode categorical features
    PolynomialFeatures,  # create polynomial feature combinations
)
```

### 7.5 Model Selection

```python
from sklearn.model_selection import (
    train_test_split,    # split data into train/test
    cross_val_score,     # k-fold cross-validation
    GridSearchCV,        # exhaustive hyperparameter search
    RandomizedSearchCV,  # random hyperparameter search
)
```

### 7.6 Metrics

```python
from sklearn.metrics import (
    # Regression
    r2_score,                    # R² score: how much variance is explained (1.0 = perfect)
    mean_absolute_error,         # MAE: average absolute error
    mean_squared_error,          # MSE: average squared error
    root_mean_squared_error,     # RMSE: square root of MSE (same unit as target)

    # Classification
    accuracy_score,              # % of correct predictions
    precision_score,             # of predicted positives, how many are actually positive
    recall_score,                # of actual positives, how many were predicted positive
    f1_score,                    # harmonic mean of precision and recall
    classification_report,       # full report with all metrics per class
    confusion_matrix,            # table: actual vs predicted
    roc_auc_score,               # area under ROC curve (for binary classification)
)
```

### 7.7 Understanding R² Score

R² (R-squared) = the proportion of variance in y that is explained by the model.

```
R² = 1.0   →  perfect model, explains everything
R² = 0.81  →  explains 81% of the variance — good
R² = 0.0   →  model is as good as just predicting the mean
R² < 0     →  model is worse than predicting the mean — terrible
```

### 7.8 Understanding Accuracy

```
Accuracy = correct predictions / total predictions

Example: 85 correct out of 100 → accuracy = 0.85 = 85%
```

Accuracy is misleading on **imbalanced datasets** (e.g., 99% of emails are not spam — predicting "not spam" always gives 99% accuracy but is useless).

### 7.9 Feature Selection

```python
from sklearn.feature_selection import RFE  # Recursive Feature Elimination

model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=3)
rfe.fit(X_train, y_train)

print(rfe.ranking_)   # 1 = most important, higher = less important
print(rfe.support_)   # True/False for selected features
```

---

## 8. Model Evaluation

### 8.1 Overfitting vs. Underfitting

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Overfitting** | High train score, low test score | More data, simpler model, regularization |
| **Underfitting** | Low train score, low test score | More complex model, more features |
| **Good fit** | Train ≈ test score, both high | — |

```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R²: {train_score:.3f}")
print(f"Test R²:  {test_score:.3f}")
# Large gap between train and test → overfitting
```

### 8.2 Confusion Matrix (Classification)

```
                Predicted: No    Predicted: Yes
Actual: No      True Negative    False Positive
Actual: Yes     False Negative   True Positive
```

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

### 8.3 Cross-Validation

Instead of a single train/test split, cross-validation splits the data K times and averages the score. More reliable, especially on small datasets.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross validation
print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 9. Algorithms Reference

### 9.1 Linear Regression

**What it does:** Fits a straight line (or hyperplane) through the data to predict a continuous value.

**Statistical background:** Minimizes the sum of squared residuals (OLS — Ordinary Least Squares).

**Formula:** `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_)   # β₀
print(model.coef_)        # β₁, β₂, ...
```

**When to use:** Baseline regression model. Simple, fast, interpretable. Works well when relationship is truly linear.

### 9.2 K-Nearest Neighbors (KNN)

**What it does:** Classifies (or regresses) a new point by looking at its K closest neighbors in the training data and taking a majority vote (classification) or average (regression).

**How it works step by step:**
1. Store all training data (no actual "training" happens)
2. For a new point: calculate the distance to every training point
3. Find the K nearest neighbors (smallest distances)
4. Classification: majority vote of neighbors' labels → predicted class
5. Regression: average of neighbors' values → predicted value

**Distance metric (default: Euclidean):**
```
d = √((x₁-x₁')² + (x₂-x₂')² + ... + (xₙ-xₙ')²)
```

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Classification
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)          # accuracy
knn.predict_proba(X_test)          # class probabilities

# Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)
knn_reg.score(X_test, y_test)      # R² score
```

**Key hyperparameters:**

| Parameter | What it does | Default |
|-----------|-------------|---------|
| `n_neighbors` | Number of neighbors K | 5 |
| `metric` | Distance function | `'euclidean'` |
| `weights` | `'uniform'` = all equal; `'distance'` = closer neighbors count more | `'uniform'` |

**Choosing K:**
- Small K → complex boundary, sensitive to noise → **overfitting** risk
- Large K → smooth boundary → **underfitting** risk
- Rule of thumb: K ≈ √(n_samples), always **odd** for binary classification to avoid ties

```python
# Find best K by testing a range
from sklearn.metrics import accuracy_score

best_k, best_score = 1, 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    if score > best_score:
        best_k, best_score = k, score
    print(f"K={k:2d}: {score:.3f}")

print(f"\nBest K: {best_k} with score {best_score:.3f}")
```

**⚠️ Always scale features before KNN!**
Without scaling, features with large ranges (e.g. salary 0–100,000) completely dominate features with small ranges (e.g. age 0–100).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
```

**Pros:** Simple, no training time, naturally handles multi-class, works for both classification and regression.  
**Cons:** Slow at prediction time (must compute all distances), needs feature scaling, struggles with high-dimensional data (curse of dimensionality).

---

### 9.3 Logistic Regression

Despite the name, this is a **classification** algorithm. It predicts the **probability** that a sample belongs to a class.

**How it works:**
Linear regression can output any number. Logistic Regression passes that number through the **sigmoid function** to squeeze it into a probability between 0 and 1.

```
sigmoid(z) = 1 / (1 + e^(-z))    →  always outputs between 0 and 1
```

If the predicted probability is ≥ 0.5 → class 1. If < 0.5 → class 0.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)   # increase max_iter if it doesn't converge
model.fit(X_train, y_train)

model.predict(X_test)                        # predicted class labels
model.predict_proba(X_test)                  # [[prob_class0, prob_class1], ...]
model.score(X_test, y_test)                  # accuracy

print(model.coef_)       # coefficients (one per feature per class)
print(model.intercept_)  # intercept
```

**Key hyperparameters:**

| Parameter | What it does | Default |
|-----------|-------------|---------|
| `C` | Inverse of regularization strength. Smaller C = more regularization | `1.0` |
| `max_iter` | Maximum iterations for the solver | `100` |
| `multi_class` | Strategy for multi-class: `'auto'`, `'ovr'`, `'multinomial'` | `'auto'` |
| `solver` | Optimization algorithm: `'lbfgs'`, `'liblinear'`, `'saga'` | `'lbfgs'` |

**Multi-class classification:**
```python
# Works out of the box for more than 2 classes
model = LogisticRegression(multi_class='auto', max_iter=1000)
model.fit(X_train, y_train)
# predict_proba returns one probability column per class
```

**Evaluation:**
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
# Shows precision, recall, f1-score for each class

cm = confusion_matrix(y_test, y_pred)
```

**Pros:** Fast, interpretable coefficients, outputs probabilities, works well with many features, strong baseline.  
**Cons:** Assumes a linear decision boundary — fails when classes are not linearly separable.

**When to use:** Binary or multi-class classification, when you need probabilities, as a strong and fast baseline before trying complex models.

---

### 9.4 Decision Trees

**What it does:** Learns a tree of yes/no questions about the features to split the data into groups and make predictions.

**How it works:**
1. Start with all data at the root
2. Find the feature + threshold that best splits the data (reduces impurity the most)
3. Recursively split each branch until a stopping condition is met
4. Each leaf node contains a prediction (majority class or mean value)

**Splitting criterion — Gini Impurity (default for classification):**
Measures how "mixed" a node is. A pure node (all one class) has Gini = 0.
```
Gini = 1 - Σ(pᵢ²)
```
Where pᵢ is the proportion of class i in the node.

**Splitting criterion — Entropy (information gain):**
```
Entropy = -Σ(pᵢ · log₂(pᵢ))
```
Both criteria produce similar results in practice. Gini is slightly faster.

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
tree = DecisionTreeClassifier(
    max_depth=5,           # maximum depth of the tree
    min_samples_split=2,   # minimum samples needed to split a node
    min_samples_leaf=1,    # minimum samples required at a leaf node
    criterion='gini'       # splitting criterion: 'gini' or 'entropy'
)
tree.fit(X_train, y_train)
tree.predict(X_test)
tree.score(X_test, y_test)

# Regression
tree_reg = DecisionTreeRegressor(max_depth=5)
tree_reg.fit(X_train, y_train)
tree_reg.score(X_test, y_test)   # R² score
```

**Key hyperparameters:**

| Parameter | What it does | Tip |
|-----------|-------------|-----|
| `max_depth` | Maximum depth of tree. None = unlimited | Start with 3–5 to avoid overfitting |
| `min_samples_split` | Min samples to split a node | Higher = simpler tree |
| `min_samples_leaf` | Min samples in a leaf node | Higher = simpler tree |
| `criterion` | `'gini'` or `'entropy'` | Usually doesn't matter much |

**Visualizing the tree:**
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=['Junior', 'Senior'],
    filled=True,          # color nodes by class
    rounded=True,
    fontsize=10
)
plt.show()
```

**Feature importance:**
```python
# How much each feature contributed to the splits
importances = pd.Series(tree.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()
```

**Overfitting with Decision Trees:**
Without constraints, a decision tree will grow until every leaf is pure (perfect training accuracy, terrible test accuracy). Use `max_depth` to control this.

```python
# Compare train vs test score for different depths
for depth in range(1, 11):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    print(f"depth={depth:2d} | train: {tree.score(X_train, y_train):.3f} | test: {tree.score(X_test, y_test):.3f}")
```

**Pros:** Highly interpretable (you can draw the tree), no feature scaling needed, handles both numeric and categorical data naturally, captures non-linear relationships.  
**Cons:** Very prone to overfitting (requires max_depth tuning), unstable (small data change can produce very different tree), generally weaker than ensemble methods.

**When to use:** When interpretability is critical, as a quick first model, as a building block for Random Forests.

---

## 10. Quick Import Cheatsheet

```python
# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading & splitting
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

# Algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Metrics
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Feature selection
from sklearn.feature_selection import RFE
```

---

*Last updated: May 2026 · Continuously extended*
