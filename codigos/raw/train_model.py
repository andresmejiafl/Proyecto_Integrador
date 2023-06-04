# from cgi import print_form
# import time

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statistics
# import ast
# from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import random
import lightgbm as lgb

# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from sklearn import tree
# from matplotlib import pyplot
# import codigos_iniciales.EvaluadorModelos as paquetest
# import shap

def train_grid(X_train, y_train, n_iter, parametros):
	#########################################################
	# Se fija la semilla
	random.seed(42)

	#########################################################
	# Hiperparámetros Algoritmos
	#########################################################
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
		'min_child_weight' : min_data_le,
		'learning_rate'	   : learning_ra
	}

	dict_hpparams = {'lgbm':params_lgbm, 'rf':param_rf, 'xgb':param_xgb}

	#########################################################
	# Resultados a medir: totales
	#########################################################



	#########################################################
	# Iteraciones
	#########################################################
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

				else :
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					model1 = xgb.XGBClassifier(
					learning_rate = hyperparameters['learning_rate'],
					n_estimators = hyperparameters['n_estimators'],
					max_depth = hyperparameters['max_depth'],
					min_child_weight = hyperparameters['min_data_in_leaf'],
					n_jobs = -3,
					verbose = -1,
					seed = 42
					)

					typemodel = 'XGBoost'

				#########################################################
				# Entrenamiento
				#########################################################
				model1.fit(X_train, np.ravel(y_train))

				#########################################################
				################# REPORTE DE RESULTADOS #################
				#########################################################

				#########################################################
				# Predicciones: Totales
				#########################################################
				y_score_train  =  model1.predict_proba(X_train)
				y_score_test   =  model1.predict_proba(X_test)
				y_score_oot    =  model1.predict_proba(X_oot)

				y_score_train  =  [item[1] for item in y_score_train]
				y_score_test   =  [item[1] for item in y_score_test]
				y_score_oot    =  [item[1] for item in y_score_oot]

				base_train_pre[nom_proba] = y_score_train
				base_test_pre[nom_proba]  = y_score_test
				base_oot_pre[nom_proba]   = y_score_oot

				#########################################################
				# Métricas precisión, mapeo y concentraciones: Totales
				#########################################################
				if flg_estricto :
				#########################################################
				# Métricas estrictas, valores solo mayores al anterior
				#########################################################
					_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango_estricta( base_train_pre, nom_proba, target[0], 'f_analisis' )
					_, enrango_test, ordena_test, prct_desorden_test   = metricas_ordena_rango_estricta( base_test_pre,  nom_proba, target[0], 'f_analisis' )
					_, enrango_oot, ordena_oot, prct_desorden_oot     = metricas_ordena_rango_estricta( base_oot_pre,   nom_proba, target[0], 'f_analisis' )
					_,_,_,ordena_train_q, ordena_test_q, ordena_oot_q = metricas_ordena_q_estricta(base_train_pre, base_test_pre, base_oot_pre, q, nom_proba, target[0], 'f_analisis')

				else:
				#########################################################
				# Métricas no estrictas, valores mayores o iguales
				#########################################################
					_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango( base_train_pre, nom_proba, target[0], 'f_analisis' )
					_, enrango_test, ordena_test, prct_desorden_test   = metricas_ordena_rango( base_test_pre,  nom_proba, target[0], 'f_analisis' )
					_, enrango_oot, ordena_oot, prct_desorden_oot     = metricas_ordena_rango( base_oot_pre,   nom_proba, target[0], 'f_analisis' )
					_,_,_,ordena_train_q, ordena_test_q, ordena_oot_q = metricas_ordena_q(base_train_pre, base_test_pre, base_oot_pre, q, nom_proba, target[0], 'f_analisis')

				auc_train = metrics.roc_auc_score( y_train, y_score_train )
				auc_test  = metrics.roc_auc_score( y_test,  y_score_test )
				auc_oot   = metrics.roc_auc_score( y_oot,   y_score_oot )


				if flg_poblacion :
					#########################################################
					# Predicciones: Especificas
					#########################################################
					y_score_train  =  model1.predict_proba(X_train_e)
					y_score_test   =  model1.predict_proba(X_test_e)
					y_score_oot    =  model1.predict_proba(X_oot_e)

					y_score_train  =  [item[1] for item in y_score_train]
					y_score_test   =  [item[1] for item in y_score_test]
					y_score_oot    =  [item[1] for item in y_score_oot]

					base_train_pre_e[nom_proba] = y_score_train
					base_test_pre_e[nom_proba] = y_score_test
					base_oot_pre_e[nom_proba] = y_score_oot

					#########################################################
					# Métricas precisión, mapeo y concentraciones: Especificas
					#########################################################
					_, enrango_train_ee, ordena_train_ee = metricas_ordena_rango( base_train_pre_e, nom_proba, target[0], 'f_analisis' )
					_, enrango_test_ee, ordena_test_ee   = metricas_ordena_rango( base_test_pre_e,  nom_proba, target[0], 'f_analisis' )
					_, enrango_oot_ee, ordena_oot_ee     = metricas_ordena_rango( base_oot_pre_e,   nom_proba, target[0], 'f_analisis' )

					auc_train_ee = metrics.roc_auc_score( y_train_e, y_score_train )
					auc_test_ee  = metrics.roc_auc_score( y_test_e, y_score_test )
					auc_oot_ee   = metrics.roc_auc_score( y_oot_e, y_score_oot )


				# Escalas en rango
				enrango_train_l.append(enrango_train)
				enrango_test_l.append(enrango_test)
				enrango_oot_l.append(enrango_oot)

				# Escalas ordenadas
				ordena_train_l.append(ordena_train)
				ordena_test_l.append( ordena_test)
				ordena_oot_l.append(  ordena_oot)

				# Escalas ordenadas q
				ordena_train_q_l.append(ordena_train_q)
				ordena_test_q_l.append( ordena_test_q)
				ordena_oot_q_l.append(  ordena_oot_q)

				# Escalas ordenadas q
				prct_desorden_train_l.append(prct_desorden_train)
				prct_desorden_test_l.append( prct_desorden_test)
				prct_desorden_oot_l.append(  prct_desorden_oot)

				# AUC
				auc_train_l.append(auc_train)
				auc_test_l.append(auc_test)
				auc_oot_l.append(auc_oot)


				if flg_poblacion :
					# Escalas en rango
					enrango_train_l_e.append(enrango_train_ee)
					enrango_test_l_e.append( enrango_test_ee)
					enrango_oot_l_e.append(  enrango_oot_ee)

					# Escalas ordenadas
					ordena_train_l_e.append(ordena_train_ee)
					ordena_test_l_e.append( ordena_test_ee)
					ordena_oot_l_e.append(  ordena_oot_ee)

					# AUC
					auc_train_l_e.append(auc_train_ee)
					auc_test_l_e.append( auc_test_ee)
					auc_oot_l_e.append(  auc_oot_ee)

				typemodel_l.append(typemodel)
				paramters.append(hyperparameters)

			except:
				pass

	resultados = {'parameters': paramters,
						'auc_tr': auc_train_l, 'auc_te': auc_test_l, 'auc_o': auc_oot_l,
						'enrango_tr': enrango_train_l, 'enrango_te': enrango_test_l, 'enrango_o': enrango_oot_l,
						'ordena_tr': ordena_train_l, 'ordena_te': ordena_test_l, 'ordena_o': ordena_oot_l,
						'prct_desorden_tr': prct_desorden_train_l, 'prct_desorden_te': prct_desorden_test_l, 'prct_desorden_o': prct_desorden_oot_l,
						'ordena_tr_q': ordena_train_q_l, 'ordena_te_q': ordena_test_q_l, 'ordena_o_q': ordena_oot_q_l,
						'modelo' : typemodel_l
						}
	res_df = pd.DataFrame.from_dict(resultados)

	res_df['auc_dif'] = round(abs(res_df['auc_tr'] - res_df['auc_te']), 4)
	res_df['ar_tr'] = 2*res_df['auc_tr'] - 1
	res_df['ar_te'] = 2*res_df['auc_te'] - 1
	res_df['ar_o']  = 2*res_df['auc_o']  - 1
	res_df['enrango'] = res_df['enrango_tr'] * res_df['enrango_te']
	res_df['ordena'] = res_df['ordena_tr'] * res_df['ordena_te']
	res_df['ordena_q'] = res_df['ordena_tr_q'] * res_df['ordena_te_q']

	if escala_c == False : 
		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif',
					'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o',
					'prct_desorden_tr','prct_desorden_te', 'prct_desorden_o',
					'ordena_q','ordena_tr_q', 'ordena_te_q', 'ordena_o_q' ]
	else:
		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif',
					'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o']

	return res_df[columnas]

