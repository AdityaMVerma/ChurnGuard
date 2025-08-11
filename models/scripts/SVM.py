from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

def train_svm_model(X_train, y_train, X_test, y_test):
    # Define base model
    model = SVC(class_weight='balanced', probability=True)

    # Define grid for tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],  # linear vs radial basis function
        'gamma': ['scale', 'auto']   # only applies to 'rbf'
    }

    # Grid search to find best combination
    grid = GridSearchCV(model, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("SVM Model Results:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(best_model, '../../models/trained/svm_model.pkl')

    return best_model
