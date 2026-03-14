
def generate_report(grid_time, random_time, evaluation_results):

    print("\n" + "=" * 60)
    print("FINAL PROJECT SUMMARY")
    print("=" * 60)

    print("\nHyperparameter Tuning Comparison")
    print("---------------------------------")
    print(f"Grid Search Time: {grid_time:.2f} seconds")
    print(f"Random Search Time: {random_time:.2f} seconds")

    print("\nPerformance Results")
    print("---------------------------------")

    for method, metrics in evaluation_results.items():
        print(f"{method}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

    print("\nConclusion")
    print("---------------------------------")
    print("Grid Search checks all parameter combinations.")
    print("Random Search samples combinations randomly.")
    print("Random Search is usually faster.")
    print("Grid Search may find slightly better parameters.")
    print("=" * 60)
