from data.data_generation import create_sample_data
from processing.integration import integrate_data
from processing.cleaning import clean_data
from processing.feature_engineering import engineer_features
from modeling.preprocessing import prepare_data, create_preprocessor
from modeling.tuning import tune_model
from modeling.evaluation import evaluate_models
from reporting.report import generate_report


def main():
    print("Generating sample data...")
    demographics, transactions = create_sample_data()

    print("\nDataset 1 (Demographics) Preview:")
    print(demographics.head())

    print("\nDataset 2 (Transactions) Preview:")
    print(transactions.head())

    print("Integrating data...")
    merged = integrate_data(demographics, transactions)

    print("\nMerged Dataset Preview:")
    print(merged.head())

    print("Cleaning data...")
    cleaned = clean_data(merged)

    print("Engineering features...")
    final_df = engineer_features(cleaned)

    print("Preparing data for modeling...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        cat_cols,
        num_cols
    ) = prepare_data(final_df)

    preprocessor = create_preprocessor(cat_cols, num_cols)

    print("Performing hyperparameter tuning...")
    (
        grid_model,
        random_model,
        grid_time,
        random_time
    ) = tune_model(X_train, y_train, preprocessor)

    print("Evaluating models...")
    evaluation_results = evaluate_models(
        grid_model.best_estimator_,
        random_model.best_estimator_,
        X_test,
        y_test
    )

    generate_report(grid_time, random_time, evaluation_results)


if __name__ == "__main__":
    main()
