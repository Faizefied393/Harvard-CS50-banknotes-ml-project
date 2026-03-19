# Banknotes — Machine Learning Classification Project

A supervised machine learning project that classifies banknotes as **authentic** or **counterfeit** using four numerical features extracted from wavelet-transformed images.

---

## Features

Each banknote is described by four continuous numerical attributes:

| Feature | Description |
|---|---|
| Variance | Variance of the wavelet-transformed image |
| Skewness | Skewness of the wavelet-transformed image |
| Curtosis | Curtosis of the wavelet-transformed image |
| Entropy | Entropy of the image |

Labels are mapped as follows:

- `0` → **Authentic**
- `1` → **Counterfeit**

---

## Project Structure

```
banknotes/
├── banknotes.py        # Final K-nearest neighbors implementation
├── compare_models.py   # Compares multiple ML models side-by-side
├── banknotes.csv       # Dataset (1,372 samples)
├── requirements.txt    # Python dependencies
└── .gitignore          # Excludes venv and cache files
```

---

## How It Works

1. **Load** the dataset from `banknotes.csv`
2. **Split** into evidence (first 4 columns) and labels (last column)
3. **Partition** data into training and testing sets
4. **Train** a model on the training data
5. **Predict** labels for the test set
6. **Evaluate** by counting correct and incorrect predictions

---

## Models

This project implements and compares several classification algorithms:

- **K-Nearest Neighbors (KNN)** — primary model in `banknotes.py`
- **Perceptron** — linear binary classifier
- **Support Vector Machine (SVM)** — maximum-margin classifier
- **Gaussian Naive Bayes** — probabilistic classifier based on Bayes' theorem

Model performance is measured using **accuracy** — the proportion of correctly classified banknotes on the held-out test set.

---

## Concepts Demonstrated

- Supervised learning
- Binary classification
- Train/test split
- Model evaluation with accuracy metrics
- Comparison of multiple ML algorithms

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd banknotes
```

### 2. Create and activate a virtual environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the KNN classifier:

```bash
python banknotes.py
```

Run the full model comparison:

```bash
python compare_models.py
```

---

## Dataset

The dataset contains **1,372 samples**, each with 4 numerical features and a binary label. It is derived from the [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication), where features were extracted using a Haar wavelet transform applied to 400×400 pixel greyscale images of genuine and forged banknotes.