from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_models(grid_model, random_model, X_test, y_test):
    results = {}

    y_pred_grid = grid_model.predict(X_test)
    grid_accuracy = accuracy_score(y_test, y_pred_grid)
    grid_f1 = f1_score(y_test, y_pred_grid)

    y_pred_random = random_model.predict(X_test)
    random_accuracy = accuracy_score(y_test, y_pred_random)
    random_f1 = f1_score(y_test, y_pred_random)

    results["GridSearch"] = {
        "accuracy": grid_accuracy,
        "f1": grid_f1
    }

    results["RandomSearch"] = {
        "accuracy": random_accuracy,
        "f1": random_f1
    }

    print("\nModel Evaluation")
    print("----------------------------")
    print(
        f"Grid Search - Accuracy: {grid_accuracy:.4f}, F1: {grid_f1:.4f}"
    )
    print(
        f"Random Search - Accuracy: {random_accuracy:.4f}, F1: {random_f1:.4f}"
    )

    if grid_f1 >= random_f1:
        best_preds = y_pred_grid
        best_name = "Grid Search"
    else:
        best_preds = y_pred_random
        best_name = "Random Search"

    cm = confusion_matrix(y_test, best_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return results
