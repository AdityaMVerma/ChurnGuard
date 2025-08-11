# models/xgboost_model.py

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import joblib

def train_xgboost_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best F1 Score: {f1_score(y_test, best_model.predict(X_test)):.4f}")
    joblib.dump(best_model, '../../models/trained/xgboost_model.pkl')

    return best_model
