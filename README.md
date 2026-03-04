# Parkinson's Disease Detection using Voice Measurements

A machine learning project to classify Parkinson's Disease status based on voice measurements using multiple classification algorithms.

## Dataset

This project uses the **Parkinson's Disease (Voice Measurements) dataset** from OpenML. The dataset is loaded directly using the OpenML package with **Dataset ID: 1488**.

### About the Dataset
The dataset contains voice measurements from individuals with and without Parkinson's Disease. It includes features extracted from phonation recordings used to detect and monitor progression of the disease. The target variable is binary:
- **Class 0**: Healthy individuals
- **Class 1**: Individuals with Parkinson's Disease

The voice measurements include acoustic characteristics such as frequency, amplitude, and other signal processing features that can indicate the presence of Parkinson's Disease.

## Project Workflow

### 1. Data Loading and Exploration
The dataset is loaded using OpenML's `get_dataset()` function:
```python
import openml

DATASET_ID = 1488
dataset = openml.datasets.get_dataset(DATASET_ID)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute
)
```

### 2. Data Preprocessing
- **Non-numeric feature removal**: Drops any non-numeric columns to focus on numeric models
- **Target variable transformation**: Converts multi-class labels (1, 2) to binary classification (0, 1)
- **Train-test split**: Stratified 80-20 split with random state 42 to maintain class distribution

### 3. Model Comparison
Seven different classification models are trained and evaluated using 5-fold Stratified K-Fold Cross-Validation:

| Model | Preprocessing | Notes |
|-------|---------------|-------|
| Logistic Regression | StandardScaler | with balanced class weights |
| SVM (RBF kernel) | StandardScaler | with probability estimates |
| K-Nearest Neighbors | StandardScaler | - |
| Multi-Layer Perceptron (MLP) | StandardScaler | hidden layers (32, 16) |
| Random Forest | Median Imputation Only | 600 estimators, balanced weights |
| Gradient Boosting | Median Imputation Only | - |
| Histogram Gradient Boosting | Median Imputation Only | - |

**Preprocessing Pipeline**:
- **Scaled models** (LR, SVM, KNN, MLP): Median imputation + StandardScaler
- **Tree-based models**: Median imputation only (trees are scale-invariant)

### 4. Evaluation Metrics
Models are evaluated using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

### 5. Selected Model: MLP (Multi-Layer Perceptron)
After comparing all models, the **MLP** classifier achieved the best overall performance. The final model configuration:
```python
mlp_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=3000,
        random_state=42
    ))
])
```

### 6. Test Set Evaluation
The final MLP model is evaluated on held-out test data with:
- Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC curve visualization

## Project Structure

```
Project-1-BME6938-Group2/
├── project1.ipynb          # Main notebook with full analysis
├── Code/
│   └── project1.ipynb      # Copy of main notebook
├── Database/
│   └── php4ylQmK.arff      # Local copy of dataset (ARFF format)
├── Results/                # Output results and visualizations
├── requirements.txt        # Python package dependencies
├── environment.yml         # Conda environment specification
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Using pip
```bash
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate project
```

### Required Packages
```
openml>=0.15           # For dataset loading
scikit-learn>=1.4      # ML models and preprocessing
pandas>=2.2            # Data manipulation
matplotlib>=3.8        # Visualization
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook project1.ipynb
```

2. Run all cells to:
   - Load the Parkinson's dataset from OpenML
   - Preprocess and prepare the data
   - Train all 7 classification models
   - Perform cross-validation and comparison
   - Train the final MLP model
   - Evaluate on test set with visualizations

## Key Findings

- The **MLP neural network** outperformed other algorithms on this dataset
- The model successfully identifies Parkinson's Disease from voice measurements
- Cross-validation provides robust estimates of generalization performance
- Proper preprocessing (scaling for distance-based/linear models) is crucial for performance
- Confusion matrix and ROC curves confirm model reliability for both classes

## References

- OpenML Dataset 1488: https://www.openml.org/d/1488
- Dataset paper: Max A. Little, Patrick E. McSharry, Eric J. Hunter, Jennifer L. Spielman, Lorraine O. Ramig (2008). Suitability of dysphonia measurements for telemonitoring of Parkinson's disease. IEEE Transactions on Biomedical Engineering, 56(4), 1015-1022.

## Authors

Group 2 - BME 6938

## License

Academic project - BME 6938 Course
