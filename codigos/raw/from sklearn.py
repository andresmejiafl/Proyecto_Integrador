from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm
import xgboost as xgb
from scipy.stats import randint as sp_randint

# Definir el espacio de búsqueda de hiperparámetros para cada modelo
param_dist_rf = {
    'n_estimators': sp_randint(100, 1000),
    'max_depth': sp_randint(2, 10),
    'min_samples_split': sp_randint(2, 20),
    'min_samples_leaf': sp_randint(1, 20),
}

param_dist_lgbm = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'num_leaves': sp_randint(10, 100),
    'max_depth': sp_randint(2, 10),
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': sp_randint(100, 1000),
}

param_dist_xgb = {
    'booster': ['gbtree', 'gblinear'],
    'max_depth': sp_randint(2, 10),
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': sp_randint(100, 1000),
}

# Crear los clasificadores
rf_clf = RandomForestClassifier()
lgbm_clf = lgbm.LGBMClassifier()
xgb_clf = xgb.XGBClassifier()

# Definir la búsqueda aleatoria con validación cruzada
random_search_rf = RandomizedSearchCV(rf_clf, param_distributions=param_dist_rf, n_iter=10, cv=5)
random_search_lgbm = RandomizedSearchCV(lgbm_clf, param_distributions=param_dist_lgbm, n_iter=10, cv=5)
random_search_xgb = RandomizedSearchCV(xgb_clf, param_distributions=param_dist_xgb, n_iter=10, cv=5)

# Entrenar y ajustar los modelos
random_search_rf.fit(X, y)  # Reemplaza X e y con tus datos de entrenamiento
random_search_lgbm.fit(X, y)
random_search_xgb.fit(X, y)

# Imprimir los mejores hiperparámetros y puntajes para cada modelo
print("Mejores hiperparámetros para Random Forest:")
print(random_search_rf.best_params_)
print("Mejor puntaje para Random Forest:", random_search_rf.best_score_)
print()

print("Mejores hiperparámetros para LightGBM:")
print(random_search_lgbm.best_params_)
print("Mejor puntaje para LightGBM:", random_search_lgbm.best_score_)
print()

print("Mejores hiperparámetros para XGBoost:")
print(random_search_xgb.best_params_)
print("Mejor puntaje para XGBoost:", random_search_xgb.best_score_)
