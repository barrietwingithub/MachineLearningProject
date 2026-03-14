import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def tune_model(X_train, y_train, preprocessor):

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )

    start = time.time()
    grid.fit(X_train, y_train)
    grid_time = time.time() - start

    random = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=4,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )

    start = time.time()
    random.fit(X_train, y_train)
    random_time = time.time() - start

    return grid, random, grid_time, random_time,
