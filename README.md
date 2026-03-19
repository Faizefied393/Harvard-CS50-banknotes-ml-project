# Banknotes — Machine Learning Classification

A supervised machine learning project that classifies banknotes as **authentic** or **counterfeit** using four numerical features extracted from wavelet-transformed images.

---

## Overview

This project uses a labeled dataset of banknotes and applies several machine learning classification models to predict whether a banknote is genuine or fake.

**The repository includes:**
- A final `K-Nearest Neighbors` implementation
- Two starter-style comparison scripts
- A multi-model comparison script
- The dataset
- Setup instructions for running the project locally

---

## Project Structure

```text
ai-banknotes-project/
│
├── banknotes.py          # Final KNN implementation
├── banknotes0.py         # Starter: manual shuffle & holdout split
├── banknotes1.py         # Starter: train_test_split version
├── compare_models.py     # Multi-model side-by-side comparison
├── banknotes.csv         # Dataset
├── requirements.txt      # Python dependencies
├── README.md
├── .gitignore
└── venv/                 # Local only — not pushed to GitHub
```

---

## Dataset

Each banknote is described by 4 numerical features derived from wavelet-transformed images:

| Feature    | Description                              |
|------------|------------------------------------------|
| `variance` | Variance of the wavelet-transformed image |
| `skewness` | Skewness of the wavelet-transformed image |
| `curtosis` | Curtosis of the wavelet-transformed image |
| `entropy`  | Entropy of the image                     |

**Label mapping:**
- `0` → Authentic
- `1` → Counterfeit

**Dataset size:** 1,372 samples (1,373 lines including header)

---

## Concepts Demonstrated

- Supervised learning
- Binary classification
- Train/test splitting
- Model fitting and prediction
- Accuracy-based evaluation
- Comparing multiple ML algorithms

---

## Models Used

| Model | Description |
|-------|-------------|
| **K-Nearest Neighbors (KNN)** | Classifies based on proximity to training examples |
| **Perceptron** | Linear classifier inspired by neural architecture |
| **Support Vector Machine (SVM)** | Finds optimal decision boundary with maximum margin |
| **Gaussian Naive Bayes** | Probabilistic classifier assuming feature independence |

---

## How It Works

1. Load the dataset from `banknotes.csv`
2. Split each row into:
   - `evidence` — first 4 columns (features)
   - `label` — last column (authentic or counterfeit)
3. Convert feature values to floating-point numbers
4. Split data into training and testing sets
5. Train a machine learning model
6. Predict labels on the testing set
7. Evaluate results using:
   - Correct predictions
   - Incorrect predictions
   - Accuracy percentage

---

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

**Run the final KNN implementation:**
```bash
python banknotes.py
```

**Run the starter-style manual split version:**
```bash
python banknotes0.py
```

**Run the starter-style `train_test_split` version:**
```bash
python banknotes1.py
```

**Run the full model comparison:**
```bash
python compare_models.py
```

---

## Example Results

### `banknotes.py` — KNN
```
Results for model KNeighborsClassifier
Correct: 549
Incorrect: 0
Accuracy: 100.00%
```

### `banknotes0.py` — Gaussian Naive Bayes
```
Results for model GaussianNB
Correct: 466
Incorrect: 82
Accuracy: 85.04%
```

### `banknotes1.py` — Perceptron
```
Results for model Perceptron
Correct: 536
Incorrect: 13
Accuracy: 97.63%
```

### `compare_models.py` — All Models
```
Model: KNeighborsClassifier
Correct: 549 | Incorrect: 0 | Accuracy: 100.00%

Model: Perceptron
Correct: 543 | Incorrect: 6 | Accuracy:  98.91%

Model: SVC
Correct: 545 | Incorrect: 4 | Accuracy:  99.27%

Model: GaussianNB
Correct: 458 | Incorrect: 91 | Accuracy: 83.42%
```

---

## Notes

- The dataset file must be comma-separated (`.csv`), not tab-separated.
- The `venv/` folder should remain local and must **not** be pushed to GitHub.
- Results may vary slightly across runs in scripts that do not use a fixed random seed.

---

## Author

**Faiz Tariq**