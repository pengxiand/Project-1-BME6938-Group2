# Parkinson's Disease Detection using Voice Measurements

A machine learning pipeline for detecting Parkinson's Disease using acoustic voice measurements extracted from phonation recordings.

---

# Clinical Context

Parkinson's Disease (PD) is a progressive neurodegenerative disorder that affects movement, speech, and motor control. Early diagnosis is important for monitoring disease progression and enabling timely clinical intervention.

This project investigates whether voice measurements extracted from phonation recordings can be used to automatically classify Parkinson's Disease using machine learning methods.

The system is intended for:

- Researchers studying Parkinson's Disease biomarkers
- Clinical data scientists developing diagnostic tools
- Healthcare AI researchers exploring non-invasive screening methods

The goal is to evaluate multiple machine learning algorithms and identify the most effective approach for distinguishing healthy individuals from patients with Parkinson's Disease using voice features.

---

# Quick Start

## Installation

### Requirements

- Python 3.10+
- pip
- Jupyter Notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

```
openml>=0.15
scikit-learn>=1.4
pandas>=2.2
matplotlib>=3.8
numpy>=1.26
jupyter>=1.0
```

---

## Run the Pipeline

Open the notebook and run all cells:

```bash
jupyter notebook project1.ipynb
```

The notebook will automatically:

1. Download the dataset from OpenML
2. Perform preprocessing
3. Train multiple machine learning models
4. Perform cross-validation comparison
5. Train the final model
6. Evaluate the model on the test set

---

## Expected Runtime and Computational Requirements

- Runtime: 1–3 minutes
- Hardware: CPU only
- Memory usage: < 1GB RAM

The dataset is small and can be processed on any standard laptop.

---

# Usage Guide

Follow these steps to reproduce the machine learning pipeline.

---

## Step 1 — Load Dataset

The dataset is retrieved directly from OpenML.

```python
import openml

DATASET_ID = 1488
dataset = openml.datasets.get_dataset(DATASET_ID)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)
```

Expected output:

- Feature matrix `X`
- Target vector `y`

---

## Step 2 — Data Preprocessing

The preprocessing pipeline includes:

- Removal of non-numeric features
- Median imputation for missing values
- Feature scaling for applicable models
- Binary transformation of target labels

Train-test split:

```
80% training
20% testing
Stratified sampling
random_state = 42
```

This ensures reproducibility.

---

## Step 3 — Model Training and Comparison

Seven machine learning models are evaluated using **5-fold Stratified Cross Validation**.

| Model | Notes |
|------|------|
| Logistic Regression | Balanced class weights |
| SVM (RBF Kernel) | Probability estimates enabled |
| K-Nearest Neighbors | Distance-based classifier |
| Multi-Layer Perceptron | Neural network classifier |
| Random Forest | 600 estimators |
| Gradient Boosting | Ensemble boosting model |
| Histogram Gradient Boosting | Efficient tree boosting |

Two preprocessing pipelines are used.

### Scaled Models

```
Median Imputation
→ StandardScaler
```

Used for:

- Logistic Regression
- SVM
- KNN
- MLP

### Tree-Based Models

```
Median Imputation only
```

Used for:

- Random Forest
- Gradient Boosting
- Histogram Gradient Boosting

---

## Step 4 — Model Evaluation

Each model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Cross-validation results are used to compare model performance.

---

## Step 5 — Final Model Selection

The Multi-Layer Perceptron (MLP) achieved the best overall performance.

Final configuration:

```python
MLPClassifier(
    hidden_layer_sizes=(32,16),
    activation="relu",
    solver="adam",
    max_iter=3000,
    random_state=42
)
```

Pipeline:

```
Median Imputation
→ StandardScaler
→ MLP Classifier
```

---

## Step 6 — Test Set Evaluation

The final model is evaluated on the held-out test set.

Generated outputs include:

- Confusion matrix
- ROC curve
- Final evaluation metrics

Outputs are saved in the **Results/** directory.

---

# Data Description

## Data Source

Dataset: **Parkinson's Disease (Voice Measurements)**  
OpenML Dataset ID: **1488**

Dataset link:

https://www.openml.org/d/1488

Original paper:

Max A. Little, Patrick E. McSharry, Eric J. Hunter, Jennifer L. Spielman, Lorraine O. Ramig (2008).  
Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.  
IEEE Transactions on Biomedical Engineering.

---

## Data Format

The dataset contains:

- 195 samples
- 22 numeric features

These features represent acoustic characteristics extracted from voice recordings.

Examples include:

- Fundamental frequency
- Frequency variation
- Jitter
- Shimmer
- Harmonics-to-noise ratio

---

## Target Variable

Binary classification:

| Class | Meaning |
|------|------|
| 0 | Healthy individual |
| 1 | Parkinson's Disease |

---

## Data License

The dataset is publicly available through OpenML for research and academic use.

Please cite the original publication when using the dataset.

---

# Results Summary

Across seven machine learning models, the **MLP neural network achieved the best performance** on this dataset.

Key observations:

- Neural networks capture nonlinear relationships in voice features
- Feature scaling significantly improves performance for distance-based models
- Cross-validation provides robust performance estimates

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Confusion matrices and ROC curves confirm reliable classification performance.

---

# Project Structure

```
Project-1-BME6938-Group2/

project1.ipynb
Main notebook containing the full analysis pipeline

Code/
    project1.ipynb
    Backup copy of notebook

Database/
    php4ylQmK.arff
    Local dataset file

Results/
    Output figures and evaluation results

requirements.txt
    Python dependencies

environment.yml
    Conda environment configuration

README.md
    Project documentation
```

---

# Authors and Contributions

**Group 2 — BME 6938**

**Jialu Liang**  
Department of Health Outcomes and Biomedical Informatics  
University of Florida  

Contributions:

- Designed the machine learning pipeline
- Implemented data preprocessing and model training
- Performed model comparison and evaluation
- Implemented visualization and final analysis
- Wrote code and project documentation

---

**James Boyd**  
University of Florida  
Email: jamesboyd@ufl.edu  

Contributions:
- Explored dataset features and interpretation
- Assisted with evaluation metrics analysis
- Contributed Results and Disscussion

---

**Benjamin Tondre**  
University of Florida  

Contributions:

- Assisted with model comparison experiments
- Validated code outputs project documentation
- Contributed Abstract, Introduction, Literature Review, and Methods

---

# Dependencies

Complete list of libraries:

```
openml>=0.15
scikit-learn>=1.4
pandas>=2.2
matplotlib>=3.8
numpy>=1.26
jupyter>=1.0
```

Install using:

```bash
pip install -r requirements.txt
```

