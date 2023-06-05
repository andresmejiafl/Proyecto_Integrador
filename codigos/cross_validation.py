from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

def compare_models(X,y,k):
    """
    Esta función realiza un torneo de modelos con hiperparametros por defecto
    con validación cruzada para comparar los diferentes algoritmos, presenta
    metricas de auc roc, precision, accuracy, recall y f1.
    Tiene como parametros de entrada: las caracteristicas y labels
    X: Caracteristicas
    y: Label
    cv: Cantidad de kFolds de validación
    """
    print('******************************************************************** ')
    print('Iniciando torneo de modelos con validación cruzada')
    print('******************************************************************** ')

    models = {
        'Random Forest': RandomForestClassifier(n_jobs = -1),
        'Naive Bayes': GaussianNB(),
        'SVM': svm.SVC(probability=True),
        'LGBM': LGBMClassifier(device = 'gpu'),
        'XGBoost': XGBClassifier(tree_method = 'gpu_hist', gpu_id = 0)
    }

    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']

    results = []

    for model_name, model in models.items():

        print('Modelo: ' + model_name)

        cv_results = cross_validate(model, X, y, cv = k, scoring = metrics, return_train_score = True, error_score = 'raise')      
      
        train_results = {metric: cv_results['train_' + metric].mean() for metric in metrics}
        test_results = {metric: cv_results['test_' + metric].mean() for metric in metrics}
        
        results.append({
            'Model': model_name,
            'Train Accuracy': train_results['accuracy'],
            'Train precision_macro': train_results['precision_macro'],
            'Train recall_macro': train_results['recall_macro'],
            'Train f1_macro': train_results['f1_macro'],
            'Train AUC-ROC': train_results['roc_auc_ovr'],
            'Test Accuracy': test_results['accuracy'],
            'Test precision_macro': test_results['precision_macro'],
            'Test recall_macro': test_results['recall_macro'],
            'Test f1_macro': test_results['f1_macro'],
            'Test AUC-ROC': test_results['roc_auc_ovr']
        })

    df_results = pd.DataFrame(results)

    print('******************************************************************** ')
    print('Finalizando torneo de modelos con validación cruzada')
    print('******************************************************************** ')

    return df_results