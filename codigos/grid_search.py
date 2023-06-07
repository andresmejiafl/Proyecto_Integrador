from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import random

def train_models(X_train, y_train, X_test, y_test, n_iter, parametros):
	"""
	Esta función realiza el entrenamiento de los algoritmos 
	seleccionados por validación cruzada, la ejecución se
	realiza sobre una malla gruesa de hiperparametros.
	Tiene como parametros de entrada:
	X_train: Caracteristicas de entrenamiento
	y_train: Labels de entrenamiento
	X_test: Caracteristicas de validación
	y_test: Labels de validación
	n_iter: Número de iteraciones
	parametros: Diccionario de hiperparametros y valores
	"""
	print('******************************************************************** ')
	print('Iniciando busqueda de hiperparametros')
	print('******************************************************************** ')
	
	random.seed(42)

	learning_ra = parametros['learning_ra']
	estimadores = parametros['estimadores']
	profundidad = parametros['profundidad']
	min_data_le = parametros['min_data_le']

	params_lgbm = {
		'n_estimators'     : estimadores,
		'max_depth'        : profundidad,
		'min_data_in_leaf' : min_data_le,
		'learning_rate'    : learning_ra
	}

	param_rf = {
		'n_estimators'     : estimadores,
		'max_depth'        : profundidad,
		'min_samples_leaf' : min_data_le,
		'criterion'        : ['gini', 'entropy']
	}

	param_xgb = {
		'n_estimators'	   : estimadores,
		'max_depth'		   : profundidad,
		'learning_rate'	   : learning_ra
	}

	dict_hpparams = {'lgbm':params_lgbm, 'rf':param_rf, 'xgb':param_xgb}

	parameters = []; 
	auc_train_l = []; auc_test_l = []; 
	accuracy_train_l = []; accuracy_test_l = []; 
	precision_train_l = []; precision_test_l = []; 
	recall_train_l = []; recall_test_l = []; 
	f1_train_l = []; f1_test_l = [];	
	matrix_train_l = []; matrix_test_l = []; 
	typemodel_l = []

	for i in range(n_iter):

		if i % 10 == 0:
			print('******************************************************************** ')
			print('Iteración: ' + str(i))
			print('******************************************************************** ')

		for key in dict_hpparams:

			try:
				if key == 'lgbm':
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					model1 = lgb.LGBMClassifier(
					learning_rate = hyperparameters['learning_rate'],
					n_estimators = hyperparameters['n_estimators'],
					max_depth = hyperparameters['max_depth'],
					min_data_in_leaf = hyperparameters['min_data_in_leaf'],
					n_jobs = -3,
					verbose = -1,
					seed = 42
					)
					typemodel = 'Lighgbm'

				elif key == 'rf':
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					model1 = RandomForestClassifier(
					n_estimators = hyperparameters['n_estimators'],
					max_depth = hyperparameters['max_depth'],
					criterion = hyperparameters['criterion'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					n_jobs = -3,
					random_state = 42
					)
					typemodel = 'RandomForest'

				elif key == 'xgb' :
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					model1 = xgb.XGBClassifier(
					learning_rate = hyperparameters['learning_rate'],
					n_estimators = hyperparameters['n_estimators'],
					max_depth = hyperparameters['max_depth'],
					n_jobs = -3,
					seed = 42
					)
					typemodel = 'XGBoost'

				model1.fit(X_train, y_train)

				y_prob_train = model1.predict_proba(X_train)[:,1]
				y_prob_test  = model1.predict_proba(X_test)[:,1]

				y_pred_train = model1.predict(X_train)
				y_pred_test  = model1.predict(X_test)

				auc_train = roc_auc_score(y_train, y_prob_train)

				accuracy_train = accuracy_score(y_train, y_pred_train)
				precision_train = precision_score(y_train, y_pred_train)
				recall_train = recall_score(y_train, y_pred_train)
				f1_train = f1_score(y_train, y_pred_train)
				matrix_train = confusion_matrix(y_train, y_pred_train)

				auc_test = roc_auc_score(y_test, y_prob_test)
				accuracy_test = accuracy_score(y_test, y_pred_test)
				precision_test = precision_score(y_test, y_pred_test)
				recall_test = recall_score(y_test, y_pred_test)
				f1_test = f1_score(y_test, y_pred_test)
				matrix_test = confusion_matrix(y_test, y_pred_test)

				auc_train_l.append(auc_train)
				auc_test_l.append(auc_test)
				accuracy_train_l.append(accuracy_train)
				accuracy_test_l.append(accuracy_test)
				precision_train_l.append(precision_train)
				precision_test_l.append(precision_test)
				recall_train_l.append(recall_train)
				recall_test_l.append(recall_test)
				f1_train_l.append(f1_train)
				f1_test_l.append(f1_test)
				matrix_train_l.append(matrix_train)
				matrix_test_l.append(matrix_test)

				typemodel_l.append(typemodel)
				parameters.append(hyperparameters)

			except:
				pass

	resultados = {
		'parameters'   : parameters,
		'modelo' 	   : typemodel_l,      			       
		'auc_tr'	   : auc_train_l, 
		'accuracy_tr'  : accuracy_train_l,
		'precision_tr' : precision_train_l,
		'recall_tr'	   : recall_train_l,
		'f1_tr'		   : f1_train_l,
		'Matriz_tr'    : matrix_train_l,
		'auc_te'	   : auc_test_l, 
		'accuracy_te'  : accuracy_test_l,
		'precision_te' : precision_test_l,
		'recall_te'	   : recall_test_l,
		'f1_te'		   : f1_test_l,
		'Matriz_te'    : matrix_test_l
	}

	res_df = pd.DataFrame.from_dict(resultados)

	print('******************************************************************** ')
	print('Finalizando busqueda de hiperparametros')
	print('******************************************************************** ')

	return res_df