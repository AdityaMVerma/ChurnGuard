from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

def train_knn_model(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()

    # Try different values for hyperparameters
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(model, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("KNN Model Results:")
    print(classification_report(y_test, y_pred))

    # Save the best model
    joblib.dump(best_model, '../../models/trained/knn_model.pkl')

    return best_model
