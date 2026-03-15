## Group Eight Project
- Perform hyperparameter tuning using both grid search and random search on a complex
model. Compare the results and discuss the trade-offs in performance and computation time.
- Merge two or more datasets with related information (e.g., customer transactions and demographics).
- Discuss challenges and approaches for dealing with inconsistent or missing data during
integration.

## Project Description

This project builds an end-to-end machine learning pipeline to predict **high-value customers** from synthetic retail data. Two datasets — customer demographics and transaction records — are generated, integrated, cleaned, and used to train a **Random Forest classifier**. The model is then optimised using two hyperparameter tuning strategies, **Grid Search** and **Random Search**, and the results are compared by accuracy, F1 score, and runtime.

## Project Structure

GROUP 8 MLDM/
├── main.py                  # Entry point — orchestrates the full pipeline
├── config.py                # Shared constants (random state, test size, CV folds)
├── data/
│   └── data_generation.py   # Synthetic demographics & transaction dataset factory
├── processing/
│   ├── integration.py       # Dataset merging and aggregation
│   ├── cleaning.py          # Missing value handling and target creation
│   └── feature_engineering.py  # Ratio features and column pruning
├── modeling/
│   ├── preprocessing.py     # Train-test split and ColumnTransformer pipeline
│   ├── tuning.py            # GridSearchCV and RandomizedSearchCV
│   └── evaluation.py        # Accuracy, F1, and confusion matrix
└── reporting/
    └── report.py            # Console summary report

## Technologies Used

Python is the main programming language used to implement the machine learning system. It is widely used in data science because it has many libraries that support data processing, machine learning, and visualization.

**Pandas**
Pandas is a Python library used for data manipulation and analysis. It helps load datasets, clean data, merge multiple datasets, and organize data in tables called DataFrames.

**NumPy**
NumPy is a numerical computing library used for performing mathematical operations on arrays and matrices. It supports efficient calculations required during data preprocessing and model training

**Scikit-learn**
Scikit-learn is a machine learning library in Python. It provides tools for training models, preprocessing data, performing hyperparameter tuning, and evaluating model performance.

**Matplotlib**
Matplotlib is a visualization library used to create graphs and plots. In this project, it is used to display results such as performance metrics and confusion matrices.

## Requirements
Python 3.8 or higher is required. Install all dependencies with:

bash
pip install pandas numpy scikit-learn matplotlib seaborn


## How to Run

1. **Clone the repository**
   bash
   git clone https://github.com/barrietwingithub/MachineLearningProject
   cd MachineLearningProject
   

2. **Install dependencies**
   bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   

3. **Run the pipeline**
   bash
   python main.py
   

The script will print dataset previews, tuning progress, evaluation metrics, display the confusion matrix, and finish with a final summary report.



## Configuration

Key parameters are centralised in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_STATE` | `42` | Seed for reproducibility |
| `TEST_SIZE` | `0.2` | Fraction of data held out for testing |
| `CV_FOLDS` | `3` | Number of cross-validation folds |
| `N_ITER_RANDOM` | `15` | Iterations for RandomizedSearchCV |



## Key Findings

- **Grid Search** exhaustively evaluates every parameter combination — slower but thorough.
- **Random Search** samples combinations randomly — typically faster with comparable performance.
- Both methods tune `n_estimators` and `max_depth` of a `RandomForestClassifier` and are evaluated using the **F1 score** to account for any class imbalance in the target.


## GitHub Repository

[https://github.com/barrietwingithub/MachineLearningProject]



## Authors
Assanatu Barrie
Mohamed John Kanu
Zechariah Bayoh
Victor Kwesi Barber-Richards
Moses B. Koroma
Lamarana Shaw

**Group 8** — Machine Learning & Data Mining module


4.	Run the Python script
Execute the main program file:
python main.py

5.	View the results
The program will process the datasets, train the machine learning model, perform hyperparameter tuning, and display the model evaluation results.