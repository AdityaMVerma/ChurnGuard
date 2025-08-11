from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(class_weight='balanced', max_iter=1000)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }

    grid = GridSearchCV(model, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(best_model, '../../models/trained/logistic_regression_model.pkl')

    return best_model
