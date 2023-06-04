from cgi import print_form
import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import statistics
import random
import ast
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn import tree
#from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

from matplotlib import pyplot
#import paquetest
import codigos_iniciales.EvaluadorModelos as paquetest

#import codigos_iniciales.utilspsi as utp

import shap

# sargumed: Obtiene la pd y pd_ajustada(si se requiere) para un modelo dado
def calificar_base(base, model, feats, zpredict = None, li_p=0, ls_p=1):
	df = base.copy()
	df['prob'] = [item[1] for item in model.predict_proba(df[feats])]

	if zpredict is not None:
		y_0 = [ i if (i >= li_p and i <= ls_p) else li_p if i < li_p else ls_p for i in df['prob'] ]
		yp =  zpredict(y_0)		
		df['prob_aj'] = yp
		df['c'] = df['prob_aj'].apply( escala_pj )
	else:
		df['c'] = df['prob'].apply( escala_pj )
	return df


# Función para graficar los hiperparámetros de un modelo vs una métrica
def hiperplot(df_plot, metric_, hue_='l_enrango_p'):
	fig, axs = plt.subplots(2, 3, figsize=(20, 10))
	sns.scatterplot(data=df_plot, x="p_learning_rate", y=metric_, hue=hue_, ax=axs[0, 0])
	sns.scatterplot(data=df_plot, x="p_max_bins", y=metric_, hue=hue_, ax=axs[0, 1])
	sns.scatterplot(data=df_plot, x="p_n_estimators", y=metric_, hue=hue_, ax=axs[0, 2])
	sns.scatterplot(data=df_plot, x="p_num_leaves", y=metric_, hue=hue_, ax=axs[1, 0])
	sns.scatterplot(data=df_plot, x="p_max_depth", y=metric_, hue=hue_, ax=axs[1, 1])
	sns.scatterplot(data=df_plot, x="p_min_data_in_leaf", y=metric_, hue=hue_, ax=axs[1, 2])


def tdos(base, var, variable_proba):
	c_entr = base.groupby(var).agg(
	{'f_analisis': 'count', variable_proba: ['min', 'max'], 'def_default_12m': 'sum'}).reset_index()
	c_entr.columns = [var, 'cantidad', 'min', 'max', 'default']
	c_entr['TDO'] = round(c_entr['default'] / c_entr['cantidad'] , 7)
	c_entr['Porc'] = round(c_entr['cantidad'] / sum(c_entr['cantidad']) * 100, 2)
	c_entr['Escala_Obs'] = c_entr['TDO'].apply(lambda x: escala_pj(x))
	return c_entr


# Escala maestra de la C 
def escala_pj(item):
	e = 'C20'
	# print(item)
	if item <= 0.0027:
		e = 'C01'
	elif item <= 0.0038:
		e = 'C02'
	elif item <= 0.0054:
		e = 'C03'
	elif item <= 0.0077:
		e = 'C04'
	elif item <= 0.0106:
		e = 'C05'
	elif item <= 0.0153:
		e = 'C06'
	elif item <= 0.0216:
		e = 'C07'
	elif item <= 0.0255:
		e = 'C08'
	elif item <= 0.0305:
		e = 'C09'
	elif item <= 0.0357:
		e = 'C10'
	elif item <= 0.0431:
		e = 'C11'
	elif item <= 0.0518:
		e = 'C12'
	elif item <= 0.0658:
		e = 'C13'
	elif item <= 0.0859:
		e = 'C14'
	elif item <= 0.1123:
		e = 'C15'
	elif item <= 0.17:
		e = 'C16'
	elif item <= 0.23:
		e = 'C17'
	elif item <= 0.5:
		e = 'C18'
	elif item <= 1:
		e = 'C19'
	return e


# Límites de la escala maestra de la C
def probabilidades_pj(c):
	if c == 'C01':
		l_inf = 0; l_sup = 0.0027
	elif c == 'C02':
		l_inf = 0.0027; l_sup = 0.0038
	elif c == 'C03':
		l_inf = 0.0038; l_sup = 0.0054
	elif c == 'C04':
		l_inf = 0.0054; l_sup = 0.0077
	elif c == 'C05':
		l_inf = 0.0077; l_sup = 0.0106
	elif c == 'C06':
		l_inf = 0.0106; l_sup = 0.0153
	elif c == 'C07':
		l_inf = 0.0153; l_sup = 0.0216
	elif c == 'C08':
		l_inf = 0.0216; l_sup = 0.0255
	elif c == 'C09':
		l_inf = 0.0255; l_sup = 0.0305
	elif c == 'C10':
		l_inf = 0.0305; l_sup = 0.0357
	elif c == 'C11':
		l_inf = 0.0357; l_sup = 0.0431
	elif c == 'C12':
		l_inf = 0.0431; l_sup = 0.0518
	elif c == 'C13':
		l_inf = 0.0518; l_sup = 0.0658
	elif c == 'C14':
		l_inf = 0.0658; l_sup = 0.0859
	elif c == 'C15':
		l_inf = 0.0859; l_sup = 0.1123
	elif c == 'C16':
		l_inf = 0.1123; l_sup = 0.17
	elif c == 'C17':
		l_inf = 0.17; l_sup = 0.23
	elif c == 'C18':
		l_inf = 0.23; l_sup = 0.5
	elif c == 'C19':
		l_inf = 0.5; l_sup = 1
	return l_inf, l_sup


# Determina la mejor semilla para particionar las bases (Train y Test)
def best_random_state( df, nro_samples, start, stop, test_size = 0.2):

	import pandas as pd
	import datetime
	from sklearn.metrics import mean_squared_error
	from sklearn.model_selection import train_test_split

	resultados  = pd.DataFrame(columns = ['random_state','rmse'])
	samples_list = pd.Series(range(start, stop)).sample(nro_samples).tolist()

	# Lista de IDs que hacen parte del train y test
	list_ID = df.loc[df.train_test_e1.isin([0,1])].llave_nombre.unique()

	for i in samples_list:
		# Se sacan dos muestras aleatorias de IDs
		train_ID, test_ID = train_test_split(list_ID, train_size= 1 - test_size, test_size=test_size, random_state=i, shuffle=True)

		# Se dejan nuevamente los IDs en un dataframe para agregar la variable train_test_e2 a la base final
		new_train_test = pd.DataFrame(list(zip((train_ID, test_ID), [1, 0])), columns=['llave_nombre', 'train_test_e2'])
		new_train_test = new_train_test.explode('llave_nombre')

		# Se adiciona la variable train_test_e2 a la base final para revisar la TDO con esta nueva partición
		df_2 = df.loc[:, ['llave_nombre', 'f_analisis', 'train_test_e1', 'def_default_12m']].merge(new_train_test, how='left', on='llave_nombre')

		df_2['train_test_e2'] = np.where(df_2['train_test_e1'] == 2, df_2['train_test_e1'], df_2['train_test_e2'])

		tdo_res3 = df_2.groupby(['f_analisis','train_test_e2'])['def_default_12m'].mean().reset_index()
		tdo_res4 = pd.pivot_table(tdo_res3, index = 'f_analisis', columns = 'train_test_e2').reset_index()
		tdo_res4.columns = tdo_res4.columns.map(lambda x: '_'.join([str(i) for i in x]))
		tdo_res4.rename(columns = {'f_analisis_':'f_analisis'
							   , 'def_default_12m_1.0':'Train'
							   , 'def_default_12m_0.0':'Test'
							   , 'def_default_12m_2.0':'Oot'}, inplace = True)
		tdo_res4.dropna(subset= ['Train', 'Test'], inplace= True)
		rms = np.sqrt(mean_squared_error(tdo_res4.Train, tdo_res4.Test))

		resultados = resultados.append({'random_state': i, 'rmse': rms}, ignore_index=True)

	return resultados


def test_two_models(bse_train_pre_ ,bse_test_pre_, bse_oot_pre_, target, var_saldo, modelos, nombre_exp = ''):

	res_group_df = pd.DataFrame()
	res_df       = pd.DataFrame()

	#%%######################################################
	# Resultados a medir
	#########################################################
	semilla = []

	paramters = []
	res_vars_l = []

	auc_train_l = []
	auc_test_l = []
	auc_oot_l = []

	enrango_train_l = []
	enrango_test_l = []
	enrango_oot_l = []
	enrango_l = []

	estab_train_l = []
	estab_test_l = []
	estab_oot_l = []

	estabmm1_train_l = []
	estabmm1_test_l = []
	estabmm1_oot_l = []

	estab_an_train_l = []
	estab_an_test_l = []
	estab_an_oot_l = []

	estabmm1_an_train_l = []
	estabmm1_an_test_l = []
	estabmm1_an_oot_l = []

	ordena_train_l = []
	ordena_test_l = []
	ordena_oot_l = []
	ordena_l = []

	sub_train_l = []
	sub_test_l = []
	sub_oot_l = []

	sob_train_l = []
	sob_test_l = []
	sob_oot_l = []

	mc_train_l = []
	mc_test_l = []
	mc_oot_l = []

	ps_c1_train_l = []
	ps_c1_test_l = []
	ps_c1_oot_l = []

	variance_auc_l = []

	#psi_tr_oo_l = []
	#psi_tr_te_l = []

	typemodel_l = []
	group_l     = []


	for k, mo in modelos.items() :

		try :
			#%%######################################################
			# Entrenamiento modelo
			#########################################################
			# Bases
			base_train_pre___ = bse_train_pre_.copy()
			base_test_pre___  = bse_test_pre_.copy()
			base_oot_pre___   = bse_oot_pre_.copy()

			if mo['algoritmo'] == 'lgbm' :
				 model1 = lgb.LGBMClassifier(
				 learning_rate = mo['parametros']['learning_rate'],
				 max_bin = mo['parametros']['max_bin'],
				 n_estimators = mo['parametros']['n_estimators'],
				 num_leaves = mo['parametros']['num_leaves'],
				 max_depth = mo['parametros']['max_depth'],
				 min_data_in_leaf = mo['parametros']['min_data_in_leaf'],
				 n_jobs = -3,
				 silent = True,
				 verbose = -1,
				 seed = 42
				 )
				 model1.fit(base_train_pre__t[mo['variables']], np.ravel(base_train_pre__t[target]))

			elif mo['algoritmo']  == 'hgb' :
				 model1 = HistGradientBoostingClassifier(
				 loss = mo['parametros']['loss'],
				 learning_rate = mo['parametros']['learning_rate'],
				 max_iter = mo['parametros']['max_iter'],
				 max_leaf_nodes = mo['parametros']['max_leaf_nodes'],
				 max_depth = mo['parametros']['max_depth'],
				 min_samples_leaf = mo['parametros']['min_samples_leaf'],
				 l2_regularization = mo['parametros']['l2_regularization'],
				 max_bins = mo['parametros']['max_bins'],
				 tol = mo['parametros']['tol']
				 )
				 model1.fit(base_train_pre__t[mo['variables']], np.ravel(base_train_pre__t[target]))

			#%%######################################################
			# Preparación Datos
			#########################################################
			# deja los dataframes solo con las vars explicativas
			X_train = base_train_pre___[mo['variables']]
			y_train = base_train_pre___[target]

			X_test = base_test_pre___[mo['variables']]
			y_test = base_test_pre___[target]

			X_oot = base_oot_pre___[mo['variables']]
			y_oot = base_oot_pre___[target]

			X_train.reset_index(inplace=True, drop= True)
			y_train.reset_index(inplace=True, drop= True)
			X_test.reset_index(inplace=True, drop= True)
			y_test.reset_index(inplace=True, drop= True)
			X_oot.reset_index(inplace=True, drop= True)
			y_oot.reset_index(inplace=True, drop= True)

			#print('datos ok')

			#%%######################################################
			# Predicciones
			#########################################################
			y_score_train  =  model1.predict_proba(X_train)
			y_score_test   =  model1.predict_proba(X_test)
			y_score_oot    =  model1.predict_proba(X_oot)

			y_score_train  =  [item[1] for item in y_score_train]
			y_score_test   =  [item[1] for item in y_score_test]
			y_score_oot    =  [item[1] for item in y_score_oot]

			# Se define punto de corte
			cuto = 0.025

			nom_proba = 'proba'
			base_train_pre___[nom_proba] = y_score_train
			base_train_pre___['pred'] = np.where(base_train_pre___[nom_proba] >= cuto, 1, 0)

			base_test_pre___[nom_proba] = y_score_test
			base_test_pre___['pred'] = np.where(base_test_pre___[nom_proba] >= cuto, 1, 0)

			base_oot_pre___[nom_proba] = y_score_oot
			base_oot_pre___['pred'] = np.where(base_oot_pre___[nom_proba] >= cuto, 1, 0)

			paquetest.Calificador().califica(base_train_pre___,nom_proba ,paquetest.Calificador.escala_c ,'calificacion_c')
			paquetest.Calificador().califica(base_test_pre___, nom_proba ,paquetest.Calificador.escala_c ,'calificacion_c')
			paquetest.Calificador().califica(base_oot_pre___,  nom_proba ,paquetest.Calificador.escala_c ,'calificacion_c')
			#print('calificador ok')

			#%%######################################################
			# Métricas precisión, mapeo y concentraciones
			#########################################################
			tst_scl_train, enrango_train, num_ordena_train, num_sub_train, num_sob_train, \
			estab_diag_train, estab_diagmm1_train, estab_diag_train_an, estab_diagmm1_train_an, \
			m_c_train, s_c1_train, auc_train = metricasBase \
					(base_train_pre____, mo['variables'], 0, 'proba', 'pred', var_saldo )

			tst_scl_test, enrango_test,  num_ordena_test,  num_sub_test,  num_sob_test, \
					estab_diag_test, estab_diagmm1_test, estab_diag_test_an, estab_diagmm1_test_an, \
					m_c_test, s_c1_test, auc_test = metricasBase \
					(base_test_pre____, mo['variables'], 0, 'proba', 'pred', var_saldo )

			tst_scl_oot, enrango_oot,   num_ordena_oot,   num_sub_oot,   num_sob_oot, \
					estab_diag_oot, estab_diagmm1_oot, estab_diag_oot_an, estab_diagmm1_oot_an, \
					m_c_oot, s_c1_oot, auc_oot = metricasBase \
					(base_oot_pre____, mo['variables'], 0, 'proba', 'pred', var_saldo )
			#print('metricas ok')

			# Escalas en rango
			enrango_train_l.append(enrango_train)
			enrango_test_l.append(enrango_test)
			enrango_oot_l.append(enrango_oot)
			enrango_l.append(enrango_train*enrango_test*enrango_oot)

			# Escalas ordenadas
			ordena_train_l.append(num_ordena_train)
			ordena_test_l.append(num_ordena_test)
			ordena_oot_l.append(num_ordena_oot)
			ordena_l.append(num_ordena_train*num_ordena_test*num_ordena_oot)

			# Escalas subestimadas
			sub_train_l.append(num_sub_train)
			sub_test_l.append(num_sub_test)
			sub_oot_l.append(num_sub_oot)

			# Escalas sobreestimadas
			sob_train_l.append(num_sob_train)
			sob_test_l.append(num_sob_test)
			sob_oot_l.append(num_sob_oot)

			# Estabilidad mensual en la diagonal
			estab_train_l.append(int(estab_diag_train *100))
			estab_test_l.append(int(estab_diag_test *100))
			estab_oot_l.append(int(estab_diag_oot *100))

			# Estabilidad mensual en la diagonal +- 1
			estabmm1_train_l.append(int(estab_diagmm1_train *100))
			estabmm1_test_l.append(int(estab_diagmm1_test *100))
			estabmm1_oot_l.append(int(estab_diagmm1_oot *100))

			# Estabilidad anual en la diagonal
			estab_an_train_l.append(int(estab_diag_train_an *100))
			estab_an_test_l.append(int(estab_diag_test_an *100))
			estab_an_oot_l.append(int(estab_diag_oot_an *100))

			# Estabilidad anual en la diagonal +- 1
			estabmm1_an_train_l.append(int(estab_diagmm1_train_an *100))
			estabmm1_an_test_l.append(int(estab_diagmm1_test_an *100))
			estabmm1_an_oot_l.append(int(estab_diagmm1_oot_an *100))

			# Máxima concentración en una escala
			mc_train_l.append(m_c_train)
			mc_test_l.append(m_c_test)
			mc_oot_l.append(m_c_oot)

			# Participación saldo en C1
			ps_c1_train_l.append(s_c1_train)
			ps_c1_test_l.append(s_c1_test)
			ps_c1_oot_l.append(s_c1_oot)

			# AUC
			auc_train_l.append(auc_train)
			auc_test_l.append(auc_test)
			auc_oot_l.append(auc_oot)

			variance_auc = statistics.variance([auc_train, auc_test, auc_oot])
			variance_auc_l.append(variance_auc)

			paramters.append(mo['parametros'])
			typemodel_l.append(algoritmo)
			group_l.append(Group)

		except :
			pass

	#print('Consolidacion')
	resultados = {'paramters': paramters, 'var_auc':variance_auc_l,
					  'auc_tr': auc_train_l, 'auc_te': auc_test_l, 'auc_o': auc_oot_l,
					  'enrango': enrango_l, 'enrango_tr': enrango_train_l, 'enrango_te': enrango_test_l, 'enrango_o': enrango_oot_l,
					  'ordena': ordena_l, 'ordena_tr': ordena_train_l, 'ordena_te': ordena_test_l, 'ordena_o': ordena_oot_l,
					  'estab_tr': estab_train_l, 'estab_te': estab_test_l, 'estab_o': estab_oot_l,
					  'mc_tr': mc_train_l ,'mc_te': mc_test_l, 'mc_o': mc_oot_l,
					  'sld_c1_tr' : ps_c1_train_l, 'sld_c1_te' : ps_c1_test_l, 'sld_c1_o' : ps_c1_oot_l,
					  'modelo' : typemodel_l,
					  'Group' : group_l
					 #'nombre_experimento' : nombre_exp
					 }


	res_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in resultados.items() ]))

	#res_df['group'] = Group
	res_df['nombre_exp'] = nombre_exp

	return res_df, resultados


def extrae_hiperparmetros( x ):
	param_dict = ast.literal_eval(x['parameters'])
	if x['modelo'] == 'Lighgbm' :
		learning_r = param_dict['learning_rate']
		estimadores = param_dict['n_estimators']
		profundidad = param_dict['max_depth']
		min_data = param_dict['min_data_in_leaf']
		bins = param_dict['max_bin']
		n_leaves = param_dict['num_leaves']
	elif x['modelo'] == 'HistGradientBoosting' :
		learning_r = param_dict['learning_rate']
		estimadores = param_dict['max_iter']
		profundidad = param_dict['max_depth']
		min_data = param_dict['min_samples_leaf']
		bins = param_dict['max_bins']
		n_leaves = param_dict['max_leaf_nodes']
	else :
		return None
	return [learning_r, estimadores, profundidad, min_data, bins, n_leaves]


########################################################################################################################
# Esta funciones genera un resumen de las métricas del modelo de ML: retorna el ordenamiento y el mapeo
# Autor: morodrig
########################################################################################################################
def mapeo_escala( modelo, df_train, df_test, df_oot, features, escala_c = True, q_ = 20):
	y_score_train  =  modelo.predict_proba(df_train[features])
	y_score_test   =  modelo.predict_proba(df_test[features])
	y_score_oot    =  modelo.predict_proba(df_oot[features])

	y_score_train  =  [item[1] for item in y_score_train]
	y_score_test   =  [item[1] for item in y_score_test]
	y_score_oot    =  [item[1] for item in y_score_oot]

	nom_proba = 'proba'
	df_train[nom_proba] = y_score_train
	df_test[nom_proba]  = y_score_test
	df_oot[nom_proba]   = y_score_oot

	mtr, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango( df_train, nom_proba, 'def_default_12m', 'f_analisis' )
	mte, enrango_test_ee, ordena_test_ee,_   = metricas_ordena_rango( df_test,  nom_proba, 'def_default_12m', 'f_analisis' )
	mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango( df_oot,   nom_proba, 'def_default_12m', 'f_analisis' )

	mtr['rango'] = mtr['c'] == mtr['c_obs']
	mte['rango'] = mte['c'] == mte['c_obs']
	mo['rango']  = mo[ 'c'] == mo[ 'c_obs']

	rangos_escala_pj = pd.DataFrame.from_dict( {'c':['C01',	'C02',	'C03',	'C04',	'C05',	'C06',	'C07',	'C08',	'C09',	'C10',	'C11',	'C12',	'C13',	'C14',	'C15',	'C16',	'C17',	'C18',	'C19'],
												'l_inf': [0,	0.0027,	0.0038,	0.0054,	0.0077,	0.0106,	0.0153,	0.0216,	0.0255,	0.0305,	0.0357,	0.0431,	0.0518,	0.0658,	0.0859,	0.1123,	0.17,	0.23,	0.5],
												'l_sup': [0.0027,	0.0038,	0.0054,	0.0077,	0.0106,	0.0153,	0.0216,	0.0255,	0.0305,	0.0357,	0.0431,	0.0518,	0.0658,	0.0859,	0.1123,	0.17,	0.23,	0.5,	1] } )

	cols_1 = ['c']
	cols_n = ['tdo','prct','rango','ordena','cantidad']

	mtr = mtr[cols_1+cols_n]
	mte = mte[cols_1+cols_n]
	mo  = mo[ cols_1+cols_n]

	mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
	mte.columns = cols_1 + [x+'_te' for x in cols_n]
	mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

	df_consolidado = rangos_escala_pj.merge(mtr, how='outer').merge(mte, how='outer').merge(mo, how='outer')

	if escala_c == False: 
		mtr, mte, mo, ordena_train_ee, ordena_test_ee, ordena_oot_ee = metricas_ordena_q( df_train, df_test, df_oot, q_, nom_proba, 'def_default_12m', 'f_analisis' )

		cols_1 = ['q', 'q_range']
		cols_n = ['tdo','prct','ordena','cantidad']

		mtr = mtr[cols_1+cols_n]
		mte = mte[cols_1+cols_n]
		mo  = mo[ cols_1+cols_n]

		mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
		mte.columns = cols_1 + [x+'_te' for x in cols_n]
		mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

		df_consolidado = mtr.merge(mte, how='outer', on = ['q','q_range']).merge(mo, how='outer', on = ['q','q_range'])

	return df_consolidado

def mapeo_escala_estricto( modelo, df_train, df_test, df_oot, features, escala_c = True, q_ = 20):
	y_score_train  =  modelo.predict_proba(df_train[features])
	y_score_test   =  modelo.predict_proba(df_test[features])
	y_score_oot    =  modelo.predict_proba(df_oot[features])

	y_score_train  =  [item[1] for item in y_score_train]
	y_score_test   =  [item[1] for item in y_score_test]
	y_score_oot    =  [item[1] for item in y_score_oot]

	nom_proba = 'proba'
	df_train[nom_proba] = y_score_train
	df_test[nom_proba]  = y_score_test
	df_oot[nom_proba]   = y_score_oot

	mtr, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango_estricta( df_train, nom_proba, 'def_default_12m', 'f_analisis' )
	mte, enrango_test_ee, ordena_test_ee,_   = metricas_ordena_rango_estricta( df_test,  nom_proba, 'def_default_12m', 'f_analisis' )
	mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango_estricta( df_oot,   nom_proba, 'def_default_12m', 'f_analisis' )

	mtr['rango'] = mtr['c'] == mtr['c_obs']
	mte['rango'] = mte['c'] == mte['c_obs']
	mo['rango']  = mo[ 'c'] == mo[ 'c_obs']

	rangos_escala_pj = pd.DataFrame.from_dict( {'c':['C01',	'C02',	'C03',	'C04',	'C05',	'C06',	'C07',	'C08',	'C09',	'C10',	'C11',	'C12',	'C13',	'C14',	'C15',	'C16',	'C17',	'C18',	'C19'],
												'l_inf': [0,	0.0027,	0.0038,	0.0054,	0.0077,	0.0106,	0.0153,	0.0216,	0.0255,	0.0305,	0.0357,	0.0431,	0.0518,	0.0658,	0.0859,	0.1123,	0.17,	0.23,	0.5],
												'l_sup': [0.0027,	0.0038,	0.0054,	0.0077,	0.0106,	0.0153,	0.0216,	0.0255,	0.0305,	0.0357,	0.0431,	0.0518,	0.0658,	0.0859,	0.1123,	0.17,	0.23,	0.5,	1] } )

	cols_1 = ['c']
	cols_n = ['tdo','prct','rango','ordena','cantidad']

	mtr = mtr[cols_1+cols_n]
	mte = mte[cols_1+cols_n]
	mo  = mo[ cols_1+cols_n]

	mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
	mte.columns = cols_1 + [x+'_te' for x in cols_n]
	mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

	df_consolidado = rangos_escala_pj.merge(mtr, how='outer').merge(mte, how='outer').merge(mo, how='outer')

	if escala_c == False: 
		mtr, mte, mo, ordena_train_ee, ordena_test_ee, ordena_oot_ee = metricas_ordena_q_estricta( df_train, df_test, df_oot, q_, nom_proba, 'def_default_12m', 'f_analisis' )

		cols_1 = ['q', 'q_range']
		cols_n = ['tdo','prct','ordena','cantidad']

		mtr = mtr[cols_1+cols_n]
		mte = mte[cols_1+cols_n]
		mo  = mo[ cols_1+cols_n]

		mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
		mte.columns = cols_1 + [x+'_te' for x in cols_n]
		mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

		df_consolidado = mtr.merge(mte, how='outer', on = ['q','q_range']).merge(mo, how='outer', on = ['q','q_range'])

	return df_consolidado

########################################################
# Mapeo escala CV
# Realiza el mapeo de escalas para para CrossValidation con el mapeo de manera estricta
# Nota: No tiene flag de estricta, solo actua de manera estricta
########################################################
def mapeo_escala_cv( modelo, df_train_test, df_oot, features, escala_c = True, q_ = 20):
	y_score_train_test  =  modelo.predict_proba(df_train_test[features]) 
	y_score_oot    =  modelo.predict_proba(df_oot[features])

	y_score_train_test  =  [item[1] for item in y_score_train_test]
	y_score_oot    =  [item[1] for item in y_score_oot]

	nom_proba = 'proba'
	df_train_test[nom_proba] = y_score_train_test
	df_oot[nom_proba]   = y_score_oot

	mtrte, enrango_train_test_ee, ordena_train_test_ee,_ = metricas_ordena_rango_estricta( df_train_test, nom_proba, 'def_default_12m', 'f_analisis' )
	mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango_estricta( df_oot,   nom_proba, 'def_default_12m', 'f_analisis' )

	mtrte['rango'] = mtrte['c'] == mtrte['c_obs']
	mo['rango']  = mo[ 'c'] == mo[ 'c_obs']

	rangos_escala_pj = pd.DataFrame.from_dict( {'c':['C01',	'C02',	'C03',	'C04',	'C05',	'C06',	'C07',	'C08',	'C09',	'C10',	'C11',	'C12',	'C13',	'C14',	'C15',	'C16',	'C17',	'C18',	'C19'],
												'l_inf': [0,	0.0027,	0.0038,	0.0054,	0.0077,	0.0106,	0.0153,	0.0216,	0.0255,	0.0305,	0.0357,	0.0431,	0.0518,	0.0658,	0.0859,	0.1123,	0.17,	0.23,	0.5],
												'l_sup': [0.0027,	0.0038,	0.0054,	0.0077,	0.0106,	0.0153,	0.0216,	0.0255,	0.0305,	0.0357,	0.0431,	0.0518,	0.0658,	0.0859,	0.1123,	0.17,	0.23,	0.5,	1] } )

	cols_1 = ['c']
	cols_n = ['tdo','prct','rango','ordena','cantidad']

	mtrte = mtrte[cols_1+cols_n]
	mo  = mo[ cols_1+cols_n]

	mtrte.columns = cols_1 + [x+'_trte' for x in cols_n]
	mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

	df_consolidado = rangos_escala_pj.merge(mtrte, how='outer').merge(mo, how='outer')

	if escala_c == False: 
		mtrte, mo, ordena_train_test_ee, ordena_oot_ee = metricas_ordena_q_cv_estricta( df_train_test, df_oot, q_, nom_proba, 'def_default_12m', 'f_analisis' )

		cols_1 = ['q', 'q_range']
		cols_n = ['tdo','prct','ordena','cantidad']

		mtrte = mtrte[cols_1+cols_n]
		mo  = mo[ cols_1+cols_n]

		mtrte.columns = cols_1 + [x+'_trte' for x in cols_n]
		mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

		df_consolidado = mtrte.merge(mo, how='outer', on = ['q','q_range'])

	return df_consolidado


def metricas( hyperparameters, base_x, base_y, base_x_tr, base_x_te, base_x_o, varis, tipo = 'lgbm' ):

	if tipo == 'lgbm' :
		modelx = lgb.LGBMClassifier( learning_rate = hyperparameters['learning_rate'], max_bin = hyperparameters['max_bin'],
								n_estimators = hyperparameters['n_estimators'], num_leaves = hyperparameters['num_leaves'],
								max_depth = hyperparameters['max_depth'], min_data_in_leaf = hyperparameters['min_data_in_leaf'],
								n_jobs = -3, silent = True, seed = 42 )
	elif tipo == 'hgb' :
		modelx = HistGradientBoostingClassifier( loss = hyperparameters['loss'],learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],random_state = 42 )
	modelx.fit(base_x, base_y)

	df_ = mapeo_escala( modelx, base_x_tr, base_x_te, base_x_o, varis )
	print('Ordenamiento en train ' + str(df_['ordena_tr'].sum(skipna = True)))
	print('Ordenamiento en test  ' + str(df_['ordena_te'].sum(skipna = True)))
	print('Ordenamiento en oot   ' + str(df_['ordena_o'].sum(skipna = True)))
	print('Rango en train ' + str(df_['rango_tr'].sum(skipna = True)))
	print('Rango en test  ' + str(df_['rango_te'].sum(skipna = True)))
	print('Rango en oot   ' + str(df_['rango_o'].sum(skipna = True)))

	return df_, modelx


########################################################################################################################
# Esta función ejecuta la función "metricas" para el top de modelos definidos por su ordenamiento en test y por su auc_dif
# Autor: jaimolin
########################################################################################################################
def metricas_top_models(base_x, base_y, base_x_tr, base_x_te, base_x_o, varis, ronda_i, num_top_models = 10, ronda_completa = False, cota_var2 = 14, cota_auc_dif = 0.05):
	"""
	Esta función devuelve el listado de métricas y modelos que se entrenan a partir de la función 
	fm.metricas para varios modelos. Sin necesidad de copiar y pegar los parametros uno a uno.
	
	Se obtienen los primeros n_top_models en base a su ordenamiento en test y su auc_dif.
	
	Parametros:
	- ronda_i: base de datos con los modelos de una iteración (ronda)
	- num_top_models: número de modelos a comparar
	- ordena_by: metrica para usar en el top
	- ascending: ordena de forma ascendente o descendente
	
	Resultado: Lista donde cada elemento es una tupla -> [(tv_0, model_0), ..., (tv_n, model_n)]    
	"""

	if ronda_completa : 
		ronda_i_top = ronda_i.loc[(ronda_i.auc_dif <= cota_auc_dif) & (ronda_i.ordena_te >= cota_ordena)]
	else: 
		ronda_i_top = ronda_i.loc[ronda_i.ordena_te.isin(ronda_i.sort_values('ordena_te', ascending = False).head(50).ordena_te.unique())].copy()

	ronda_i_top = ronda_i_top.sort_values(["ordena_te","auc_dif"], ascending=[False, True])
	ronda_i_top['rn_ordena']=tuple(zip(ronda_i_top.ordena_te,ronda_i_top.auc_dif))
	ronda_i_top['rn_ordena'] = ronda_i_top[['rn_ordena']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top = ronda_i_top.sort_values(["auc_dif", "ordena_te"], ascending=[True, False])
	ronda_i_top['rn_auc'] = tuple(zip(ronda_i_top.ordena_te,ronda_i_top.auc_dif))
	ronda_i_top['rn_auc'] = ronda_i_top[['rn_auc']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top['rn'] = ronda_i_top['rn_ordena'] * ronda_i_top['rn_auc']

	ronda_i_top = ronda_i_top.sort_values(by="rn", ascending=True).head(num_top_models)

	modelos = []
	for index, row in ronda_i_top.iterrows():
		# Tipo del modelo
		if str(ronda_i_top.loc[index,:]['modelo']) == 'HistGradientBoosting':
			tipo_modelo = "hgb"
		else:
			tipo_modelo = "lgbm"
		# Hiper-parametros del modelo row
		hyperparameters_row = eval(str(ronda_i_top.loc[index,:]['parameters']))
		# Creción del modelo y sus métricas
		print("modelo_", index, sep = "")
		tv_row, model_row = metricas(hyperparameters_row, base_x, base_y, base_x_tr, base_x_te, base_x_o, varis, tipo = tipo_modelo)
		tv_row = tv_row.fillna(0)
		print('Porcentaje de registros no ordena train: ' + str(sum(tv_row[tv_row['ordena_tr']==0]['prct_tr'])))
		print('Porcentaje de registros no ordena test: ' + str(sum(tv_row[tv_row['ordena_te']==0]['prct_te'])))
		print('Porcentaje de registros no ordena oot: ' + str(sum(tv_row[tv_row['ordena_o' ]==0]['prct_o'])))
		print("\n")
		# Se guarda la información de los modelos en una lista
		modelos.append((tv_row, model_row))
	return modelos


def metricas_top_models_q(base_x, base_y, base_x_tr, base_x_te, base_x_o, varis, ronda_i, num_top_models = 10, ronda_completa = False, var1= ["auc_dif", 0.05, True], var2 = ["ordena_te_q", 14, False], q = 20, estricto = True):
	"""
	Esta función devuelve el listado de métricas y modelos que se entrenan a partir de la función 
	fm.metricas para varios modelos. Sin necesidad de copiar y pegar los parametros uno a uno.
	Se obtienen los primeros n_top_models en base a su ordenamiento en test y su auc_dif.
	Var1 siempre debe ser ascending = True y var2 siempre debe ser ascending = False
	Parametros:
	- ronda_i: base de datos con los modelos de una iteración (ronda)
	- num_top_models: número de modelos a comparar
	- ordena_by: metrica para usar en el top
	- ascending: ordena de forma ascendente o descendente
	Resultado: Lista donde cada elemento es una tupla -> [(tv_0, model_0), ..., (tv_n, model_n)]    
	"""

	if ronda_completa : 
		ronda_i_top = ronda_i.loc[(ronda_i[var1[0]] <= var1[1]) & (ronda_i[var2[0]] >= var2[1])]
	else: 
		ronda_i_top = ronda_i.loc[ronda_i['ordena_te_q'].isin(ronda_i.sort_values('ordena_te_q', ascending = False).head(50)['ordena_te_q'].unique())]
 
	ronda_i_top = ronda_i_top.sort_values([var1[0], var2[0]], ascending=[var1[2], var2[2]])
	ronda_i_top['rn_var1'] = tuple(zip(ronda_i_top[var1[0]], ronda_i_top[var2[0]]))
	ronda_i_top['rn_var1'] = ronda_i_top[['rn_var1']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top = ronda_i_top.sort_values([var2[0], var1[0]], ascending=[var2[2], var1[2]])
	ronda_i_top['rn_var2']=tuple(zip(ronda_i_top[var2[0]], ronda_i_top[var1[0]]))
	ronda_i_top['rn_var2'] = ronda_i_top[['rn_var2']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top['rn'] = ronda_i_top['rn_var2'] * ronda_i_top['rn_var1']

	ronda_i_top = ronda_i_top.sort_values(by="rn", ascending=True).head(num_top_models)
	
	modelos = []
	for index, row in ronda_i_top.iterrows():
		
		hyperparameters = eval(str(ronda_i_top.loc[index,:]['parameters']))
		
		# Tipo del modelo
		if str(ronda_i_top.loc[index,:]['modelo']) == 'HistGradientBoosting':
			tipo_modelo = "hgb"
			modelx = HistGradientBoostingClassifier( loss = hyperparameters['loss'],learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],random_state = 42 )
		else:
			tipo_modelo = "lgbm"
			modelx = lgb.LGBMClassifier( learning_rate = hyperparameters['learning_rate'], max_bin = hyperparameters['max_bin'],
								n_estimators = hyperparameters['n_estimators'], num_leaves = hyperparameters['num_leaves'],
								max_depth = hyperparameters['max_depth'], min_data_in_leaf = hyperparameters['min_data_in_leaf'],
								n_jobs = -3, silent = True, seed = 42 )
		
		model_row = modelx.fit(base_x, base_y)
		
		# Creción del modelo y sus métricas
		print("modelo_", index, sep = "")

		if estricto:
			tv_row = mapeo_escala_estricto( model_row, base_x_tr, base_x_te, base_x_o, varis , escala_c = False, q_ = q)
		else:
			tv_row = mapeo_escala( model_row, base_x_tr, base_x_te, base_x_o, varis , escala_c = False, q_ = q)
		#tv_row = tv_row.fillna(0)
		
		# Se guarda la información de los modelos en una lista
		modelos.append((tv_row, model_row))
	
	return modelos


########################################################################################################################
# metricas_top_models_q cv
########################################################################################################################
def metricas_top_models_q_cv(base_x, base_y, base_x_tr_te, base_x_o, varis, ronda_i, num_top_models = 10, ronda_completa = False, var1= ["auc_dif", 0.05, True], var2 = ["ordena_te_q", 14, False], q = 20):
	"""
	Esta función devuelve el listado de métricas y modelos que se entrenan a partir de la función 
	fm.metricas para varios modelos. Sin necesidad de copiar y pegar los parametros uno a uno.
	Se obtienen los primeros n_top_models en base a su ordenamiento en test y su auc_dif.
	Var1 siempre debe ser ascending = True y var2 siempre debe ser ascending = False
	Parametros:
	- ronda_i: base de datos con los modelos de una iteración (ronda)
	- num_top_models: número de modelos a comparar
	- ordena_by: metrica para usar en el top
	- ascending: ordena de forma ascendente o descendente
	Resultado: Lista donde cada elemento es una tupla -> [(tv_0, model_0), ..., (tv_n, model_n)]    
	"""

	if ronda_completa : 
		ronda_i_top = ronda_i.loc[(ronda_i[var1[0]] <= var1[1]) & (ronda_i[var2[0]] >= var2[1])]
	else: 
		ronda_i_top = ronda_i.loc[ronda_i['ordena_te_q'].isin(ronda_i.sort_values('ordena_te_q', ascending = False).head(50)['ordena_te_q'].unique())]
 
	ronda_i_top = ronda_i_top.sort_values([var1[0], var2[0]], ascending=[var1[2], var2[2]])
	ronda_i_top['rn_var1'] = tuple(zip(ronda_i_top[var1[0]], ronda_i_top[var2[0]]))
	ronda_i_top['rn_var1'] = ronda_i_top[['rn_var1']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top = ronda_i_top.sort_values([var2[0], var1[0]], ascending=[var2[2], var1[2]])
	ronda_i_top['rn_var2']=tuple(zip(ronda_i_top[var2[0]], ronda_i_top[var1[0]]))
	ronda_i_top['rn_var2'] = ronda_i_top[['rn_var2']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top['rn'] = ronda_i_top['rn_var2'] * ronda_i_top['rn_var1']

	ronda_i_top = ronda_i_top.sort_values(by="rn", ascending=True).head(num_top_models)
	
	modelos = []
	for index, row in ronda_i_top.iterrows():
		
		hyperparameters = eval(str(ronda_i_top.loc[index,:]['parameters']))
		
		# Tipo del modelo
		if str(ronda_i_top.loc[index,:]['modelo']) == 'HistGradientBoosting':
			tipo_modelo = "hgb"
			modelx = HistGradientBoostingClassifier( loss = hyperparameters['loss'],learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],random_state = 42 )
		else:
			tipo_modelo = "lgbm"
			modelx = lgb.LGBMClassifier( learning_rate = hyperparameters['learning_rate'], max_bin = hyperparameters['max_bin'],
								n_estimators = hyperparameters['n_estimators'], num_leaves = hyperparameters['num_leaves'],
								max_depth = hyperparameters['max_depth'], min_data_in_leaf = hyperparameters['min_data_in_leaf'],
								n_jobs = -3, silent = True, seed = 42 )
		
		model_row = modelx.fit(base_x, base_y)
		
		# Creción del modelo y sus métricas
		print("modelo_", index, sep = "")

		tv_row = mapeo_escala_cv( model_row, base_x_tr_te, base_x_o, varis , escala_c = False, q_ = q)
		#tv_row = tv_row.fillna(0)
		
		# Se guarda la información de los modelos en una lista
		modelos.append((tv_row, model_row))
	
	return modelos


########################################################################################################################
# Esta funcion categoriza una variable utilizando un árbol de decisión
# Retorna los cortes sugeridos por el árbol y el código de python para categorizar la variable
# Autor: morodrig
########################################################################################################################
def cortes( base_cruce, var, profundidad = 4 ):
	clf = tree.DecisionTreeClassifier( max_depth = profundidad, class_weight = 'balanced',
									   min_samples_leaf = max(10, int(round(base_cruce.shape[0]*0.01, 0 )) ) )
	clf = clf.fit(base_cruce[[var]], base_cruce['def_default_12m'])

	cortes  = []

	n_nodes = clf.tree_.node_count
	children_left = clf.tree_.children_left
	children_right = clf.tree_.children_right
	threshold = clf.tree_.threshold

	node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
	is_split = np.zeros(shape=n_nodes, dtype=bool)
	is_leaves = np.zeros(shape=n_nodes, dtype=bool)

	stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
	while len(stack) > 0:
		# `pop` ensures each node is only visited once
		node_id, depth = stack.pop()

		is_split_node = children_left[node_id] != children_right[node_id]
		is_split[node_id] = is_split_node

		if is_split_node:
			stack.append((children_left[node_id], depth + 1))
			stack.append((children_right[node_id], depth + 1))
		else:
			is_leaves[node_id] = True

	thrs = list(threshold)
	cortes_i = [ thrs[x] for x in range(len(thrs)) if is_split[x] ]
	cortes_i.sort()
	cortes.append( cortes_i )

	cortes_ = cortes[0]
	c_0 = round(cortes_[0], 3)
	com_str = 'def agrupacion_analitica_au( x ): \n\tif x <= ' + str(c_0) + ' : return ' + str(c_0) + ' \n'
	for c in cortes_[1:]:
		c_i = round(c , 3)
		com_str = com_str + '\telif x <= ' + str(c_i) + ' : return ' + str(c_i) + ' \n'
	com_str = com_str + '\telse: return 99999999999'

	return com_str, cortes_


########################################################################################################################
# Esta funicion genera el valor del AR basado en la medicion de una regresión logistica donde la variable independiente
# es la calificación del modelo generado.
# Autor: Gerencia de riesgo de modelos.
########################################################################################################################
def calculo_ar(datos_cal,var_x,var_y):
	from sklearn.linear_model import LogisticRegression
	regresion_logistica = LogisticRegression()
	if datos_cal[var_y].sum() > 2 and len(set(datos_cal[var_x].tolist())) > 1:
		escala_riesgo_d = pd.get_dummies(datos_cal[var_x],drop_first=False) ##escala de riesgo
		X = np.array(escala_riesgo_d)
		Y =  np.array(datos_cal[var_y]) ## variable respuesta... default 12m
		regresion_logistica.fit(X,Y)
		roc_res = roc_auc_score(Y, regresion_logistica.predict_proba(X)[:,1])
		resultado_metricas = [roc_res,roc_res*2 - 1]
	else:
		resultado_metricas = [None,None]
	return(resultado_metricas)


########################################################################################################################
# Esta funicion calcula la tabla de validación del modelo y el cálculo de cuántas escalas están ordenadas y
# cuantas están en rango
# Autor: morodrig.
########################################################################################################################
def metricas_ordena_rango( base, nom_proba, var_default, var_conteo ):
	base['c'] = base[nom_proba].apply( escala_pj )
	c_entr = base.groupby(['c']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	c_entr.columns = ['c','cantidad','default']
	c_entr['tdo'] = round( c_entr['default'] / c_entr['cantidad']*100,2)
	c_entr['prct'] = round( c_entr['cantidad'] / sum(c_entr['cantidad'])*100,2)
	c_entr['c_obs'] = c_entr['tdo'].apply( lambda x: escala_pj(x/100) )

	en_rango = sum(c_entr['c']==c_entr['c_obs'])

	c_entr.sort_values(by='c', inplace=True)
	c_entr['before_max_tdo'] = c_entr['tdo'].expanding(1).max()
	c_entr['ordena'] = (c_entr['tdo'] >= c_entr['before_max_tdo']).astype(int)

	ordena = sum(c_entr['ordena'])

	prct_desorden = sum(c_entr[c_entr['ordena']==0]['prct'])

	return c_entr, en_rango, ordena, prct_desorden


def metricas_ordena_q( base_train_pre_, base_test_pre_, base_oot_pre_, q__, nom_proba, var_default, var_conteo ):
	
	## Train
	base_train_pre_['q'] = pd.qcut(base_train_pre_[nom_proba], q__, labels = False, duplicates='drop')
	base_train_pre_['q_range'], bins = pd.qcut(base_train_pre_[nom_proba], q__, retbins=True, duplicates='drop')
	cat = base_train_pre_[['q','q_range']].drop_duplicates()

	q_train = base_train_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_train.columns = ['q','cantidad','default']
	q_train['tdo'] = round( q_train['default'] / q_train['cantidad']*100,2)
	q_train['prct'] = round( q_train['cantidad'] / sum(q_train['cantidad'])*100,2)

	q_train.sort_values(by='q', inplace=True)
	q_train['before_max_tdo'] = q_train['tdo'].expanding(1).max()
	q_train['ordena'] = (q_train['tdo'] >= q_train['before_max_tdo']).astype(int)
	q_train = pd.merge(q_train, cat, how = 'left', on = 'q')
	ordena_train_q = sum(q_train['ordena'])

	## Test
	base_test_pre_['q'] = pd.cut(base_test_pre_[nom_proba], bins, labels = False)
	base_test_pre_['q_range'] = pd.cut(base_test_pre_[nom_proba], bins)

	q_test = base_test_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_test.columns = ['q','cantidad','default']
	q_test['tdo'] = round( q_test['default'] / q_test['cantidad']*100,2)
	q_test['prct'] = round( q_test['cantidad'] / sum(q_test['cantidad'])*100,2)

	q_test.sort_values(by='q', inplace=True)
	q_test['before_max_tdo'] = q_test['tdo'].expanding(1).max()
	q_test['ordena'] = (q_test['tdo'] >= q_test['before_max_tdo']).astype(int)
	q_test = pd.merge(q_test, cat, how = 'left', on = 'q')
	ordena_test_q = sum(q_test['ordena'])

	## OOT
	base_oot_pre_['q'] = pd.cut(base_oot_pre_[nom_proba], bins, labels = False)
	base_oot_pre_['q_range'] = pd.cut(base_oot_pre_[nom_proba], bins)

	q_oot = base_oot_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_oot.columns = ['q','cantidad','default']
	q_oot['tdo'] = round( q_oot['default'] / q_oot['cantidad']*100,2)
	q_oot['prct'] = round( q_oot['cantidad'] / sum(q_oot['cantidad'])*100,2)

	q_oot.sort_values(by='q', inplace=True)
	q_oot['before_max_tdo'] = q_oot['tdo'].expanding(1).max()
	q_oot['ordena'] = (q_oot['tdo'] >= q_oot['before_max_tdo']).astype(int)
	q_oot = pd.merge(q_oot, cat, how = 'left', on = 'q')
	ordena_oot_q = sum(q_oot['ordena'])

	return q_train, q_test, q_oot, ordena_train_q, ordena_test_q, ordena_oot_q

def metricas_ordena_q_estricta( base_train_pre_, base_test_pre_, base_oot_pre_, q__, nom_proba, var_default, var_conteo ):
	
	## Train
	base_train_pre_['q'] = pd.qcut(base_train_pre_[nom_proba], q__, labels = False)
	base_train_pre_['q_range'], bins = pd.qcut(base_train_pre_[nom_proba], q__, retbins=True )
	cat = base_train_pre_[['q','q_range']].drop_duplicates()

	q_train = base_train_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_train.columns = ['q','cantidad','default']
	q_train['tdo'] = round( q_train['default'] / q_train['cantidad']*100,2)
	q_train['prct'] = round( q_train['cantidad'] / sum(q_train['cantidad'])*100,2)

	q_train.sort_values(by='q', inplace=True)
	max_val_train = q_train['tdo'].expanding(1).max()
	ini_val_train = pd.Series(-1)
	q_train['before_max_tdo'] = ini_val_train.append(max_val_train).reset_index(drop=True).drop(labels=len(q_train))
	q_train['ordena'] = (q_train['tdo'] > q_train['before_max_tdo']).astype(int)
	q_train = pd.merge(q_train, cat, how = 'left', on = 'q')
	ordena_train_q = sum(q_train['ordena'])

	## Test
	base_test_pre_['q'] = pd.cut(base_test_pre_[nom_proba], bins, labels = False)
	base_test_pre_['q_range'] = pd.cut(base_test_pre_[nom_proba], bins)

	q_test = base_test_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_test.columns = ['q','cantidad','default']
	q_test['tdo'] = round( q_test['default'] / q_test['cantidad']*100,2)
	q_test['prct'] = round( q_test['cantidad'] / sum(q_test['cantidad'])*100,2)

	q_test.sort_values(by='q', inplace=True)
	max_val_test = q_test['tdo'].expanding(1).max()
	ini_val_test = pd.Series(-1)
	q_test['before_max_tdo'] = ini_val_test.append(max_val_test).reset_index(drop=True).drop(labels=len(q_test))
	q_test['ordena'] = (q_test['tdo'] > q_test['before_max_tdo']).astype(int)
	q_test = pd.merge(q_test, cat, how = 'left', on = 'q')
	ordena_test_q = sum(q_test['ordena'])

	## OOT
	base_oot_pre_['q'] = pd.cut(base_oot_pre_[nom_proba], bins, labels = False)
	base_oot_pre_['q_range'] = pd.cut(base_oot_pre_[nom_proba], bins)

	q_oot = base_oot_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_oot.columns = ['q','cantidad','default']
	q_oot['tdo'] = round( q_oot['default'] / q_oot['cantidad']*100,2)
	q_oot['prct'] = round( q_oot['cantidad'] / sum(q_oot['cantidad'])*100,2)

	q_oot.sort_values(by='q', inplace=True)
	max_val_oot = q_oot['tdo'].expanding(1).max()
	ini_val_oot = pd.Series(-1)
	q_oot['before_max_tdo'] = ini_val_oot.append(max_val_oot).reset_index(drop=True).drop(labels=len(q_oot))
	q_oot['ordena'] = (q_oot['tdo'] > q_oot['before_max_tdo']).astype(int)
	q_oot = pd.merge(q_oot, cat, how = 'left', on = 'q')
	ordena_oot_q = sum(q_oot['ordena'])

	return q_train, q_test, q_oot, ordena_train_q, ordena_test_q, ordena_oot_q

########################################################################################################################
# metricas ordena q CV estricta
# Autor: mialopez
########################################################################################################################
def metricas_ordena_q_cv_estricta( base_train_test_pre_, base_oot_pre_, q__, nom_proba, var_default, var_conteo ):

	## Train-test
	base_train_test_pre_['q'] = pd.qcut(base_train_test_pre_[nom_proba], q__, labels = False, duplicates='drop')
	base_train_test_pre_['q_range'], bins = pd.qcut(base_train_test_pre_[nom_proba], q__, retbins=True, duplicates='drop' )
	cat = base_train_test_pre_[['q','q_range']].drop_duplicates()

	q_train_test = base_train_test_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_train_test.columns = ['q','cantidad','default']
	q_train_test['tdo'] = round( q_train_test['default'] / q_train_test['cantidad']*100,2)
	q_train_test['prct'] = round( q_train_test['cantidad'] / sum(q_train_test['cantidad'])*100,2)

	q_train_test.sort_values(by='q', inplace=True)
	max_val_train_test = q_train_test['tdo'].expanding(1).max()
	ini_val_train_test = pd.Series(-1)
	q_train_test['before_max_tdo'] = ini_val_train_test.append(max_val_train_test).reset_index(drop=True).drop(labels=len(q_train_test))
	q_train_test['ordena'] = (q_train_test['tdo'] > q_train_test['before_max_tdo']).astype(int)
	q_train_test = pd.merge(q_train_test, cat, how = 'left', on = 'q')
	ordena_train_test_q = sum(q_train_test['ordena'])

	## OOT
	base_oot_pre_['q'] = pd.cut(base_oot_pre_[nom_proba], bins, labels = False)
	base_oot_pre_['q_range'] = pd.cut(base_oot_pre_[nom_proba], bins)

	q_oot = base_oot_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_oot.columns = ['q','cantidad','default']
	q_oot['tdo'] = round( q_oot['default'] / q_oot['cantidad']*100,2)
	q_oot['prct'] = round( q_oot['cantidad'] / sum(q_oot['cantidad'])*100,2)

	q_oot.sort_values(by='q', inplace=True)
	max_val_oot = q_oot['tdo'].expanding(1).max()
	ini_val_oot = pd.Series(-1)
	q_oot['before_max_tdo'] = ini_val_oot.append(max_val_oot).reset_index(drop=True).drop(labels=len(q_oot))
	q_oot['ordena'] = (q_oot['tdo'] > q_oot['before_max_tdo']).astype(int)
	q_oot = pd.merge(q_oot, cat, how = 'left', on = 'q')
	ordena_oot_q = sum(q_oot['ordena'])
	return q_train_test, q_oot, ordena_train_test_q, ordena_oot_q


########################################################################################################################
# Esta funcion genera la transformacion de q a C: retorna el ordenamiento y el mapeo
# Autor: wiareval
########################################################################################################################
def mapeo_q_a_c(base_final, feats_total_, var_partition, model, escala_maestra, zpredict = None, li_p=0, ls_p=1):
	"""Funcion que permite mapear de Q calibrada a C
	Input:
		base_final: dataframe pandas, base completa con la que se entrena el modelo
		feats_final: lista, Lista con la variables finales que se utilizan para entrenar el modelo
		var_partition: string, variable por la que se particiona Train, Test y OOT
		model: model sklearn, modelo entrenado
		zpredict: vector de transformacion sklearn, vector de calibracion.
		escala_maestra: dataframe pandas con escala maestra, cols: C, l_inf,l_sup
	Output:
		df_consolidado: dataframe, devuelve las escalas mapeadas a C.
	"""
	base_final_ = base_final.copy()
	base_final_['prob_q'] = [item[1] for item in model.predict_proba(base_final_[feats_total_])]

	if zpredict is not None:
		y_0 = [ i if (i >= li_p and i <= ls_p) else li_p if i < li_p else ls_p for i in base_final_['prob_q'] ]
		yp =  zpredict(y_0)		
		base_final_['prob_aj'] = yp
		mtr, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango( base_final_[base_final_[var_partition]==1], 'prob_aj', 'def_default_12m', 'f_analisis' )
		mte, enrango_test_ee, ordena_test_ee,_   = metricas_ordena_rango( base_final_[base_final_[var_partition]==0],  'prob_aj', 'def_default_12m', 'f_analisis' )
		mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango( base_final_[base_final_[var_partition]==2],   'prob_aj', 'def_default_12m', 'f_analisis' )
	else:
		mtr, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango( base_final_[base_final_[var_partition]==1], 'prob_q', 'def_default_12m', 'f_analisis' )
		mte, enrango_test_ee, ordena_test_ee,_   = metricas_ordena_rango( base_final_[base_final_[var_partition]==0],  'prob_q', 'def_default_12m', 'f_analisis' )
		mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango( base_final_[base_final_[var_partition]==2],   'prob_q', 'def_default_12m', 'f_analisis' )
	
	mtr['rango'] = mtr['c'] == mtr['c_obs']
	mte['rango'] = mte['c'] == mte['c_obs']
	mo['rango']  = mo[ 'c'] == mo[ 'c_obs']
	cols_1 = ['c']
	cols_n = ['tdo','prct','rango','ordena','cantidad']
	mtr = mtr[cols_1+cols_n]
	mte = mte[cols_1+cols_n]
	mo  = mo[ cols_1+cols_n]

	mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
	mte.columns = cols_1 + [x+'_te' for x in cols_n]
	mo.columns  = cols_1 + [x+'_o'  for x in cols_n]
	
	df_consolidado = escala_maestra.merge(mtr, how='outer').merge(mte, how='outer').merge(mo, how='outer')
	return df_consolidado


########################################################################################################################
# Esta funcion genera la transformacion de q a C: retorna el ordenamiento y el mapeo
# Autor: wiareval
########################################################################################################################
def mapeo_q_a_c_cv(base_train_test, base_oot, feats_total_, model, escala_maestra, zpredict = None):
	"""Funcion que permite mapear de Q calibrada a C
	Input:
		base_final: dataframe pandas, base completa con la que se entrena el modelo
		feats_final: lista, Lista con la variables finales que se utilizan para entrenar el modelo
		var_partition: string, variable por la que se particiona Train, Test y OOT
		model: model sklearn, modelo entrenado
		zpredict: vector de transformacion sklearn, vector de calibracion.
		escala_maestra: dataframe pandas con escala maestra, cols: C, l_inf,l_sup
	Output:
		df_consolidado: dataframe, devuelve las escalas mapeadas a C.
	"""
	base_train_test['prob_q'] = [item[1] for item in model.predict_proba(base_train_test[feats_total_])]
	base_oot['prob_q'] = [item[1] for item in model.predict_proba(base_oot[feats_total_])]
	if zpredict is not None:
		base_train_test['prob_aj'] = zpredict(base_train_test['prob_q'])
		base_oot['prob_aj'] = zpredict(base_oot['prob_q'])
		mtrte, enrango_train_test_ee, ordena_train_test_ee,_ = metricas_ordena_rango( base_train_test, 'prob_aj', 'def_default_12m', 'f_analisis' )
		mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango( base_oot,   'prob_aj', 'def_default_12m', 'f_analisis' )
	else:
		mtrte, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango( base_train_test, 'prob_q', 'def_default_12m', 'f_analisis' )
		mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango( base_oot,   'prob_q', 'def_default_12m', 'f_analisis' )
	mtrte['rango'] = mtrte['c'] == mtrte['c_obs']
	mo['rango']  = mo[ 'c'] == mo[ 'c_obs']
	cols_1 = ['c']
	cols_n = ['tdo','prct','rango','ordena','cantidad']
	mtrte = mtrte[cols_1+cols_n]
	mo  = mo[ cols_1+cols_n]

	mtrte.columns = cols_1 + [x+'_trte' for x in cols_n]
	mo.columns  = cols_1 + [x+'_o'  for x in cols_n]
	
	df_consolidado = escala_maestra.merge(mtrte, how='outer').merge(mo, how='outer')
	return df_consolidado


########################################################################################################################
# Esta funcion separa las X y de la y en las bases de entrenamiento, prueba y fuera de tiempo
# Autor: morodrig.
########################################################################################################################
def prepara_bases(base_train_pre_, base_test_pre_, base_oot_pre_, target, feats_train):
	base_train_pre = base_train_pre_.copy()
	base_test_pre = base_test_pre_.copy()
	base_oot_pre = base_oot_pre_.copy()

	# deja los dataframes solo con las vars explicativas
	X_train = base_train_pre[feats_train]
	y_train = base_train_pre[target]

	X_test = base_test_pre[feats_train]
	y_test = base_test_pre[target]

	X_oot = base_oot_pre[feats_train]
	y_oot = base_oot_pre[target]

	X_train.reset_index(inplace=True, drop= True)
	y_train.reset_index(inplace=True, drop= True)
	X_test.reset_index(inplace=True, drop= True)
	y_test.reset_index(inplace=True, drop= True)
	X_oot.reset_index(inplace=True, drop= True)
	y_oot.reset_index(inplace=True, drop= True)

	return base_train_pre, X_train, y_train, base_test_pre, X_test, y_test, base_oot_pre, X_oot, y_oot

def prepara_bases_sector(base_train_pre_, base_test_pre_, base_oot_pre_, sector,  target, feats_train):

	base_train_pre = base_train_pre_.copy()
	base_test_pre = base_test_pre_.copy()
	base_oot_pre = base_oot_pre_.copy()

	# deja los dataframes solo con las vars explicativas
	X_train = base_train_pre[feats_train]
	X_train_sec = base_train_pre.query(f'cli_sec_riesgo_{sector}==1')[feats_train]
	y_train = base_train_pre[target]
	y_train_sec = base_train_pre.query(f'cli_sec_riesgo_{sector}==1')[target]

	X_test = base_test_pre[feats_train]
	y_test = base_test_pre[target]

	X_oot = base_oot_pre[feats_train]
	y_oot = base_oot_pre[target]

	X_train.reset_index(inplace=True, drop= True)
	y_train.reset_index(inplace=True, drop= True)
	y_train_sec.reset_index(inplace=True, drop= True)

	X_test.reset_index(inplace=True, drop= True)
	y_test.reset_index(inplace=True, drop= True)

	X_oot.reset_index(inplace=True, drop= True)
	y_oot.reset_index(inplace=True, drop= True)

	return base_train_pre, X_train, X_train_sec, y_train, y_train_sec, base_test_pre, X_test, y_test, base_oot_pre, X_oot, y_oot

########################################################################################################################
# Esta función realiza el entrenamiento de n_iter x 2 (lightgbm y HistGradientBoosting). Retorna tres métricas:
# (1) número de escalas ordenadas
# (2) número de escalas en rango
# (3) auc
# Estas metricas se calculan para cada muestra: entrenamiento, prueba y fuera de tiempo. El parámetro flg_poblacion
# se usa para poder medir el desempeño del modelo en una población específica, para la que se retornan las mismas
# metricas.
#
# Autor: morodrig.
########################################################################################################################
def train_nueva_era_segint(base_train_pre_ ,base_test_pre_, base_oot_pre_, target, n_iter, seed, feats_train, parametros, muestras_en_intervalo, flg_poblacion = False, seg_interes = '' , escala_c = True, q = 20, flg_estricto = True):
	#########################################################
	# Se fija la semilla
	random.seed(seed)
	#########################################################
	# Preparación Datos: Totales
	#########################################################
	base_train_pre, X_train, y_train, base_test_pre, X_test, y_test, base_oot_pre, X_oot, y_oot = prepara_bases(base_train_pre_, base_test_pre_, base_oot_pre_, target, feats_train)

	if flg_poblacion :
		#########################################################
		# Preparación Datos: Especificos
		#########################################################
		base_train_pre_e, X_train_e, y_train_e, base_test_pre_e, X_test_e, y_test_e, base_oot_pre_e, X_oot_e, y_oot_e = prepara_bases(base_train_pre_[base_train_pre_[seg_interes]>0],
																																	  base_test_pre_[ base_test_pre_[ seg_interes]>0],
																																	  base_oot_pre_[  base_oot_pre_[  seg_interes]>0],
																																	  target, feats_train)
		#print("bases ok")

	#########################################################
	# Hiperparámetros Algoritmos
	#########################################################
	learning_ra = parametros['learning_ra']
	estimadores = parametros['estimadores']
	profundidad = parametros['profundidad']
	min_data_le = parametros['min_data_le']
	max_bins___ = parametros['max_bins___']
	n_leaves___ = parametros['n_leaves___']

	params_lgbm = {
		'learning_rate'    : learning_ra,
		'max_bin'          : max_bins___,
		'n_estimators'     : estimadores,
		'num_leaves'       : n_leaves___,
		'max_depth'        : profundidad,
		'min_data_in_leaf' : min_data_le,
	}

	param_rf = {
		'n_estimators'     : estimadores,
		'max_features'     : ['auto', 'sqrt', 'log2'],
		'max_depth'        : profundidad,
		'criterion'        : ['gini', 'entropy'],
		'min_samples_leaf' : min_data_le
	}

	param_hgb = {
			'loss'               : ['auto'],
			'learning_rate'      : learning_ra,
			'max_iter'           : estimadores,
			'max_leaf_nodes'     : n_leaves___,
			'max_depth'          : profundidad,
			'min_samples_leaf'   : min_data_le,
			'l2_regularization'  : [float(x) for x in np.linspace(start = 0, stop = 0.5, num = muestras_en_intervalo)],
			'max_bins'           : max_bins___,
			'random_state'       : [seed],
			'tol'                : [1e-7]
			}

	dict_hpparams = {'lgbm':params_lgbm, 'hgb': param_hgb}

	#########################################################
	# Resultados a medir: totales
	#########################################################
	semilla = []; paramters = [];
	auc_train_l = [];     auc_test_l = [];     auc_oot_l = [];
	enrango_train_l = []; enrango_test_l = []; enrango_oot_l = [];
	ordena_train_l = [];  ordena_test_l = [];  ordena_oot_l = [];
	ordena_train_q_l = [];  ordena_test_q_l = [];  ordena_oot_q_l = []; 
	prct_desorden_train_l = []; prct_desorden_test_l = []; prct_desorden_oot_l = [];
	typemodel_l = []

	if flg_poblacion :
		#########################################################
		# Resultados a medir: especificos
		#########################################################
		auc_train_l_e = [];     auc_test_l_e = [];     auc_oot_l_e = []
		enrango_train_l_e = []; enrango_test_l_e = []; enrango_oot_l_e = []
		ordena_train_l_e = [];  ordena_test_l_e = [];  ordena_oot_l_e = []

	# Se define punto de corte
	cuto = 0.0859
	nom_proba = 'proba'

	#########################################################
	# Iteraciones
	#########################################################
	for i in range(n_iter):

		if i % 10 == 0:
			print('******************************************************************** ')
			print('******************************************************************** ')
			print('Iteración: ' + str(i))
			print('******************************************************************** ')
			print('******************************************************************** ')

		for key in dict_hpparams:

			try:
				if key == 'lgbm':
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)

					model1 = lgb.LGBMClassifier(
					learning_rate = hyperparameters['learning_rate'],
					max_bin = hyperparameters['max_bin'],
					n_estimators = hyperparameters['n_estimators'],
					num_leaves = hyperparameters['num_leaves'],
					max_depth = hyperparameters['max_depth'],
					min_data_in_leaf = hyperparameters['min_data_in_leaf'],
					n_jobs = -3,
					#silent = True,
					verbose=-1,
					seed = seed
					)

					typemodel = 'Lighgbm'

				elif  key == 'param_rf':
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)

					model1 = RandomForestClassifier(
					n_estimators = hyperparameters['n_estimators'],
					max_features = hyperparameters['max_features'],
					criterion = hyperparameters['criterion'],
					min_samples_split = hyperparameters['min_samples_split'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					n_jobs = -3,
					)

					typemodel = 'RandomForest'

				else : ##'hgb
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)

					model1 = HistGradientBoostingClassifier(
					loss = hyperparameters['loss'],
					learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],
					max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],
					max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],
					#validation_fraction = hyperparameters['validation_fraction'],
					#n_jobs = -3,
					random_state = seed
					)

					typemodel = 'HistGradientBoosting'



				#########################################################
				# Entrenamiento
				#########################################################
				model1.fit(X_train, np.ravel(y_train))#, verbose=False)
				#print("modelo ok")

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

	if flg_poblacion :
		resultados = {'paramters': paramters,
						  'auc_tr': auc_train_l, 'auc_te': auc_test_l, 'auc_o': auc_oot_l,
						  'enrango_tr': enrango_train_l, 'enrango_te': enrango_test_l, 'enrango_o': enrango_oot_l,
						  'ordena_tr': ordena_train_l, 'ordena_te': ordena_test_l, 'ordena_o': ordena_oot_l,
						  'auc_tr_e': auc_train_l_e, 'auc_te_e': auc_test_l_e, 'auc_o_e': auc_oot_l_e,
						  'enrango_tr_e': enrango_train_l_e, 'enrango_te_e': enrango_test_l_e, 'enrango_o_e': enrango_oot_l_e,
						  'ordena_tr_e': ordena_train_l_e, 'ordena_te_e': ordena_test_l_e, 'ordena_o_e': ordena_oot_l_e,
						  'modelo' : typemodel_l
						  }

		res_df = pd.DataFrame.from_dict(resultados)

		res_df['auc_dif'] = round(abs(res_df['auc_tr'] - res_df['auc_te']), 4)
		res_df['ar_tr'] = 2*res_df['auc_tr'] - 1
		res_df['ar_te'] = 2*res_df['auc_te'] - 1
		res_df['ar_o']  = 2*res_df['auc_o']  - 1
		res_df['ar_tr_e'] = 2*res_df['auc_tr_e'] - 1
		res_df['ar_te_e'] = 2*res_df['auc_te_e'] - 1
		res_df['ar_o_e']  = 2*res_df['auc_o_e']  - 1
		res_df['enrango'] = res_df['enrango_tr'] * res_df['enrango_te']
		res_df['ordena'] = res_df['ordena_tr'] * res_df['ordena_te']

		res_df['auc_dif_e'] = round(abs(res_df['auc_tr_e'] - res_df['auc_te_e']), 4)
		res_df['enrango_e'] = res_df['enrango_tr_e'] * res_df['enrango_te_e']
		res_df['ordena_e'] = res_df['ordena_tr_e'] * res_df['ordena_te_e']

		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif', 'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o',
					'ar_tr_e','ar_te_e','ar_o_e', 'auc_dif_e', 'enrango_e', 'enrango_tr_e', 'enrango_te_e', 'enrango_o_e', 'ordena_e','ordena_tr_e', 'ordena_te_e', 'ordena_o_e']


	else :
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

########################################################################################################################
# Esta función realiza el entrenamiento con base completa de n_iter x 2 (lightgbm y HistGradientBoosting). Retorna tres métricas:
# (1) número de escalas ordenadas por método estricto
# (2) número de escalas en rango por método estricto
# (3) auc - calculado evaluando en el sector específico
# Estas metricas se calculan para cada muestra: entrenamiento, prueba y fuera de tiempo. El parámetro flg_poblacion
# se usa para poder medir el desempeño del modelo en una población específica, para la que se retornan las mismas
# metricas.
#
# Autor: morodrig. Ajustado a sectores de pocos datos: andremej, jaimolin, mebernal
########################################################################################################################
def train_nueva_era_segint_full_sector(base_train_pre_ ,base_test_pre_, base_oot_pre_, sector, target, n_iter, seed, feats_train, parametros, muestras_en_intervalo, flg_poblacion = False, seg_interes = '' , escala_c = False, q = 10):
	#########################################################
	# Se fija la semilla
	random.seed(seed)
	#########################################################
	# Preparación Datos: Totales
	#########################################################
	base_train_pre, X_train, X_train_sec, y_train, y_train_sec, base_test_pre, X_test, y_test, base_oot_pre, X_oot, y_oot = prepara_bases_sector(base_train_pre_, base_test_pre_, base_oot_pre_, sector, target, feats_train)
	if flg_poblacion :
		#########################################################
		# Preparación Datos: Especificos
		#########################################################
		base_train_pre_e, X_train_e, y_train_e, base_test_pre_e, X_test_e, y_test_e, base_oot_pre_e, X_oot_e, y_oot_e = prepara_bases(base_train_pre_[base_train_pre_[seg_interes]>0],
																																	  base_test_pre_[ base_test_pre_[ seg_interes]>0],
																																	  base_oot_pre_[  base_oot_pre_[  seg_interes]>0],
																																	  target, feats_train)
		#print("bases ok")
	#########################################################
	# Hiperparámetros Algoritmos
	#########################################################
	learning_ra = parametros['learning_ra']
	estimadores = parametros['estimadores']
	profundidad = parametros['profundidad']
	min_data_le = parametros['min_data_le']
	max_bins___ = parametros['max_bins___']
	n_leaves___ = parametros['n_leaves___']
	params_lgbm = {
		'learning_rate'    : learning_ra,
		'max_bin'          : max_bins___,
		'n_estimators'     : estimadores,
		'num_leaves'       : n_leaves___,
		'max_depth'        : profundidad,
		'min_data_in_leaf' : min_data_le,
	}
	param_rf = {
		'n_estimators'     : estimadores,
		'max_features'     : ['auto', 'sqrt', 'log2'],
		'max_depth'        : profundidad,
		'criterion'        : ['gini', 'entropy'],
		'min_samples_leaf' : min_data_le
	}
	param_hgb = {
			'loss'               : ['auto'],
			'learning_rate'      : learning_ra,
			'max_iter'           : estimadores,
			'max_leaf_nodes'     : n_leaves___,
			'max_depth'          : profundidad,
			'min_samples_leaf'   : min_data_le,
			'l2_regularization'  : [float(x) for x in np.linspace(start = 0, stop = 0.5, num = muestras_en_intervalo)],
			'max_bins'           : max_bins___,
			'random_state'       : [seed],
			'tol'                : [1e-7]
			}
	dict_hpparams = {'lgbm':params_lgbm, 'hgb': param_hgb}
	#########################################################
	# Resultados a medir: totales
	#########################################################
	semilla = []; paramters = [];
	auc_train_l = [];     auc_test_l = [];     auc_oot_l = [];
	enrango_train_l = []; enrango_test_l = []; enrango_oot_l = [];
	ordena_train_l = [];  ordena_test_l = [];  ordena_oot_l = [];
	ordena_train_q_l = [];  ordena_test_q_l = [];  ordena_oot_q_l = []; 
	prct_desorden_train_l = []; prct_desorden_test_l = []; prct_desorden_oot_l = [];
	typemodel_l = []
	if flg_poblacion :
		#########################################################
		# Resultados a medir: especificos
		#########################################################
		auc_train_l_e = [];     auc_test_l_e = [];     auc_oot_l_e = []
		enrango_train_l_e = []; enrango_test_l_e = []; enrango_oot_l_e = []
		ordena_train_l_e = [];  ordena_test_l_e = [];  ordena_oot_l_e = []
	# Se define punto de corte
	cuto = 0.0859
	nom_proba = 'proba'
	#########################################################
	# Iteraciones
	#########################################################
	for i in range(n_iter):
		if i % 10 == 0:
			print('******************************************************************** ')
			print('******************************************************************** ')
			print('Iteración: ' + str(i))
			print('******************************************************************** ')
			print('******************************************************************** ')
		for key in dict_hpparams:
			try:
				if key == 'lgbm':
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)
					model1 = lgb.LGBMClassifier(
					learning_rate = hyperparameters['learning_rate'],
					max_bin = hyperparameters['max_bin'],
					n_estimators = hyperparameters['n_estimators'],
					num_leaves = hyperparameters['num_leaves'],
					max_depth = hyperparameters['max_depth'],
					min_data_in_leaf = hyperparameters['min_data_in_leaf'],
					n_jobs = -3,
					#silent = True,
					verbose=-1,
					seed = seed
					)
					typemodel = 'Lighgbm'
				elif  key == 'param_rf':
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					model1 = RandomForestClassifier(
					n_estimators = hyperparameters['n_estimators'],
					max_features = hyperparameters['max_features'],
					criterion = hyperparameters['criterion'],
					min_samples_split = hyperparameters['min_samples_split'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					n_jobs = -3,
					)
					typemodel = 'RandomForest'
				else : ##'hgb
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)
					model1 = HistGradientBoostingClassifier(
					loss = hyperparameters['loss'],
					learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],
					max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],
					max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],
					#validation_fraction = hyperparameters['validation_fraction'],
					#n_jobs = -3,
					random_state = seed
					)
					typemodel = 'HistGradientBoosting'
				#########################################################
				# Entrenamiento
				#########################################################
				model1.fit(X_train, np.ravel(y_train))#, verbose=False)
				#print("modelo ok")
				#########################################################
				################# REPORTE DE RESULTADOS #################
				#########################################################
				#########################################################
				# Predicciones: Totales
				#########################################################
				y_score_train = model1.predict_proba(X_train)
				y_score_train_sec = model1.predict_proba(X_train_sec)
				y_score_test = model1.predict_proba(X_test)
				y_score_oot = model1.predict_proba(X_oot)
				y_score_train = [item[1] for item in y_score_train]
				y_score_train_sec = [item[1] for item in y_score_train_sec]
				y_score_test = [item[1] for item in y_score_test]
				y_score_oot = [item[1] for item in y_score_oot]
				base_train_pre[nom_proba] = y_score_train
				base_test_pre[nom_proba] = y_score_test
				base_oot_pre[nom_proba] = y_score_oot
				#########################################################
				# Métricas precisión, mapeo y concentraciones: Totales
				#########################################################
				_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango_estricta( base_train_pre.query(f'cli_sec_riesgo_{sector}==1'), nom_proba, target[0], 'f_analisis' )
				_, enrango_test, ordena_test, prct_desorden_test   = metricas_ordena_rango_estricta( base_test_pre,  nom_proba, target[0], 'f_analisis' )
				_, enrango_oot, ordena_oot, prct_desorden_oot     = metricas_ordena_rango_estricta( base_oot_pre,   nom_proba, target[0], 'f_analisis' )
				_,_,_,ordena_train_q, ordena_test_q, ordena_oot_q = metricas_ordena_q_estricta(base_train_pre.query(f'cli_sec_riesgo_{sector}==1'), base_test_pre, base_oot_pre, q, nom_proba, target[0], 'f_analisis')
				auc_train = metrics.roc_auc_score( y_train_sec, y_score_train_sec )
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
	if flg_poblacion :
		resultados = {'paramters': paramters,
						  'auc_tr': auc_train_l, 'auc_te': auc_test_l, 'auc_o': auc_oot_l,
						  'enrango_tr': enrango_train_l, 'enrango_te': enrango_test_l, 'enrango_o': enrango_oot_l,
						  'ordena_tr': ordena_train_l, 'ordena_te': ordena_test_l, 'ordena_o': ordena_oot_l,
						  'auc_tr_e': auc_train_l_e, 'auc_te_e': auc_test_l_e, 'auc_o_e': auc_oot_l_e,
						  'enrango_tr_e': enrango_train_l_e, 'enrango_te_e': enrango_test_l_e, 'enrango_o_e': enrango_oot_l_e,
						  'ordena_tr_e': ordena_train_l_e, 'ordena_te_e': ordena_test_l_e, 'ordena_o_e': ordena_oot_l_e,
						  'modelo' : typemodel_l
						  }
		res_df = pd.DataFrame.from_dict(resultados)
		res_df['auc_dif'] = round(abs(res_df['auc_tr'] - res_df['auc_te']), 4)
		res_df['ar_tr'] = 2*res_df['auc_tr'] - 1
		res_df['ar_te'] = 2*res_df['auc_te'] - 1
		res_df['ar_o']  = 2*res_df['auc_o']  - 1
		res_df['ar_tr_e'] = 2*res_df['auc_tr_e'] - 1
		res_df['ar_te_e'] = 2*res_df['auc_te_e'] - 1
		res_df['ar_o_e']  = 2*res_df['auc_o_e']  - 1
		res_df['enrango'] = res_df['enrango_tr'] * res_df['enrango_te']
		res_df['ordena'] = res_df['ordena_tr'] * res_df['ordena_te']
		res_df['auc_dif_e'] = round(abs(res_df['auc_tr_e'] - res_df['auc_te_e']), 4)
		res_df['enrango_e'] = res_df['enrango_tr_e'] * res_df['enrango_te_e']
		res_df['ordena_e'] = res_df['ordena_tr_e'] * res_df['ordena_te_e']
		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif', 'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o',
					'ar_tr_e','ar_te_e','ar_o_e', 'auc_dif_e', 'enrango_e', 'enrango_tr_e', 'enrango_te_e', 'enrango_o_e', 'ordena_e','ordena_tr_e', 'ordena_te_e', 'ordena_o_e']
	else :
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


# sargumed: Esta función realiza el mismo proceso que train_nueva_era_segint pero con validación cruzada
def train_nueva_era_segint_w_cv(base_final, target, n_iter, seed, feats_train, parametros, muestras_en_intervalo, n_folds = 5, escala_c = True, q = 20):
	from sklearn.model_selection import KFold
	random.seed(seed)
	target = ['def_default_12m']
	list_ID = base_final.loc[base_final.train_test_e1.isin([0,1])].llave_nombre.unique()

	#########################################################
	# Hiperparámetros Algoritmos
	#########################################################
	learning_ra = parametros['learning_ra']
	estimadores = parametros['estimadores']
	profundidad = parametros['profundidad']
	min_data_le = parametros['min_data_le']
	max_bins___ = parametros['max_bins___']
	n_leaves___ = parametros['n_leaves___']

	params_lgbm = {
		'learning_rate'    : learning_ra,
		'max_bin'          : max_bins___,
		'n_estimators'     : estimadores,
		'num_leaves'       : n_leaves___,
		'max_depth'        : profundidad,
		'min_data_in_leaf' : min_data_le
	}
	param_hgb = {
			'loss'               : ['auto'],
			'learning_rate'      : learning_ra,
			'max_iter'           : estimadores,
			'max_leaf_nodes'     : n_leaves___,
			'max_depth'          : profundidad,
			'min_samples_leaf'   : min_data_le,
			'l2_regularization'  : [float(x) for x in np.linspace(start = 0, stop = 0.5, num = muestras_en_intervalo)],
			'max_bins'           : max_bins___,
			'random_state'       : [seed],
			'tol'                : [1e-7]
			}
	dict_hpparams = {'lgbm':params_lgbm, 'hgb': param_hgb}
	#########################################################
	# Resultados a medir: totales
	#########################################################
	semilla = []; paramters = [];
	auc_train_l = [];     auc_test_l = [];     auc_oot_l = [];
	enrango_train_l = []; enrango_test_l = []; enrango_oot_l = [];
	ordena_train_l = [];  ordena_test_l = [];  ordena_oot_l = [];
	ordena_train_q_l = [];  ordena_test_q_l = [];  ordena_oot_q_l = []; 
	prct_desorden_train_l = []; prct_desorden_test_l = []; prct_desorden_oot_l = [];
	typemodel_l = []; k_fold_l = []

	nom_proba = 'proba'
	#########################################################
	# Iteraciones
	#########################################################
	for i in range(n_iter):
		if i % 10 == 0:
			print('******************************************************************** ')
			print('******************************************************************** ')
			print('Iteración: ' + str(i))
			print('******************************************************************** ')
			print('******************************************************************** ')
		for key in dict_hpparams:
			try: 
				if key == 'lgbm':
					semilla.append(seed)
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)
					model1 = lgb.LGBMClassifier(
					learning_rate = hyperparameters['learning_rate'],
					max_bin = hyperparameters['max_bin'],
					n_estimators = hyperparameters['n_estimators'],
					num_leaves = hyperparameters['num_leaves'],
					max_depth = hyperparameters['max_depth'],
					min_data_in_leaf = hyperparameters['min_data_in_leaf'],
					n_jobs = -3,
					#silent = True,
					verbose=-1,
					seed = seed
					)
					typemodel = 'Lighgbm'
				else :
					semilla.append(seed)
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)
					model1 = HistGradientBoostingClassifier(
					loss = hyperparameters['loss'],
					learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],
					max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],
					max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],
					#validation_fraction = hyperparameters['validation_fraction'],
					#n_jobs = -3,
					random_state = seed
					)
					typemodel = 'HistGradientBoosting'
				
				#########################################################
				# Cross Validation
				#########################################################
				kf = KFold(n_splits = n_folds)
				kf.get_n_splits(list_ID)
				auc_train_cv = []; auc_test_cv = []; auc_oot_cv = [];

				enrango_train_cv = []; enrango_test_cv = []; enrango_oot_cv = [];

				ordena_train_cv = []; ordena_test_cv = []; ordena_oot_cv = [];

				ordena_train_cv_q = []; ordena_test_cv_q = []; ordena_oot_cv_q = [];

				prct_desorden_train_cv = []; prct_desorden_test_cv = []; prct_desorden_oot_cv = [];

				for ID_train, ID_test in kf.split(list_ID):
					base_train_pre = base_final.loc[base_final.llave_nombre.isin(list_ID[ID_train])]
					base_test_pre = base_final.loc[base_final.llave_nombre.isin(list_ID[ID_test])]
					base_oot_pre   = base_final.query('train_test_e2 == 2').copy()                

					X_train = base_train_pre[feats_train]
					y_train = base_train_pre[target]

					X_test = base_test_pre[feats_train]
					y_test = base_test_pre[target]

					X_oot = base_oot_pre[feats_train]
					y_oot = base_oot_pre[target]
					print("Base Train %s Base Test %s Base OOT %s" % (base_train_pre.shape, base_test_pre.shape, base_oot_pre.shape))

					#########################################################
					# Entrenamiento 
					#########################################################
					model1.fit(X_train, np.ravel(y_train))

					#########################################################
					########### REPORTE Parciales de cada modelo ############
					#########################################################
					y_score_train  =  model1.predict_proba(X_train)
					y_score_test   =  model1.predict_proba(X_test)
					y_score_oot    =  model1.predict_proba(X_oot)
					y_score_train  =  [item[1] for item in y_score_train]
					y_score_test   =  [item[1] for item in y_score_test]
					y_score_oot    =  [item[1] for item in y_score_oot]

					base_train_pre[nom_proba] = y_score_train
					base_test_pre[nom_proba] = y_score_test
					base_oot_pre[nom_proba]  = y_score_oot

					#########################################################
					# Métricas precisión, mapeo y concentraciones: Totales
					#########################################################
					_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango(base_train_pre, nom_proba, target[0], 'f_analisis')
					_, enrango_test, ordena_test, prct_desorden_test = metricas_ordena_rango(base_test_pre, nom_proba, target[0], 'f_analisis')
					_, enrango_oot, ordena_oot, prct_desorden_oot = metricas_ordena_rango(base_oot_pre, nom_proba, target[0], 'f_analisis')

					_,_,_,ordena_train_q, ordena_test_q, ordena_oot_q = metricas_ordena_q(base_train_pre, base_test_pre, base_oot_pre, q, nom_proba, target[0], 'f_analisis')

					auc_train = metrics.roc_auc_score( y_train, y_score_train )
					auc_test  = metrics.roc_auc_score( y_test,  y_score_test )
					auc_oot   = metrics.roc_auc_score( y_oot,   y_score_oot )

					# AUC
					auc_train_cv.append(auc_train)
					auc_test_cv.append(auc_test)
					auc_oot_cv.append(auc_oot)
					# Escalas en rango
					enrango_train_cv.append(enrango_train)
					enrango_test_cv.append(enrango_test)
					enrango_oot_cv.append(enrango_oot)
					# Escalas ordenadas
					ordena_train_cv.append(ordena_train)
					ordena_test_cv.append(ordena_test)
					ordena_oot_cv.append(ordena_oot)
					# Escalas ordenadas q
					ordena_train_cv_q.append(ordena_train_q)
					ordena_test_cv_q.append(ordena_test_q)
					ordena_oot_cv_q.append(ordena_oot_q)
					# Porcentaje de desorden C
					prct_desorden_train_cv.append(prct_desorden_train)
					prct_desorden_test_cv.append( prct_desorden_test)
					prct_desorden_oot_cv.append(  prct_desorden_oot)


				# Promedio AUC
				auc_train_l.append(np.mean(auc_train_cv))
				auc_test_l.append(np.mean(auc_test_cv))
				auc_oot_l.append(np.mean(auc_oot_cv))

				# Promedio Escalas en rango
				enrango_train_l.append(np.mean(enrango_train_cv))
				enrango_test_l.append(np.mean(enrango_test_cv))
				enrango_oot_l.append(np.mean(enrango_oot_cv))

				# Promedio Escalas ordenadas
				ordena_train_l.append(np.mean(ordena_train_cv))
				ordena_test_l.append( np.mean(ordena_test_cv))
				ordena_oot_l.append(  np.mean(ordena_oot_cv))

				# Promedio Escalas ordenadas q
				ordena_train_q_l.append(np.mean(ordena_train_cv_q))
				ordena_test_q_l.append(np.mean(ordena_test_cv_q))
				ordena_oot_q_l.append(np.mean(ordena_oot_cv_q))

				# Promedio Porcentaje de desorden C
				prct_desorden_train_l.append(np.mean(prct_desorden_train_cv))
				prct_desorden_test_l.append(np.mean(prct_desorden_test_cv))
				prct_desorden_oot_l.append(np.mean(prct_desorden_oot_cv))

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


########################################################################################################################
# Esta función genera el modelo preliminar de alguna iteración.
# Autor: mialopez, morodrig y mpaz.
########################################################################################################################
def fit_model_group(parametros, bse_train_pre_ ,bse_test_pre_, bse_oot_pre_, target, feats_train, agrupacion, algoritmo):

	#res_group_df = pd.DataFrame()
	#res_df       = pd.DataFrame()

	#%%######################################################
	# Entrenamiento modelo
	#########################################################
	# Bases

	if (agrupacion == 'Comercio' or agrupacion == 'Infraestructura' or agrupacion == 'Manurfactura' or \
		agrupacion == 'RR_Naturales' or agrupacion == 'Servicios' or agrupacion == 'Edificaciones' or agrupacion == 'Agropecuario'):
		bse_train_pre_[bse_train_pre_['sector']==agrupacion]
		bse_test_pre_[bse_test_pre_['sector']==agrupacion]
		bse_oot_pre_[bse_oot_pre_['sector']==agrupacion]

		base_train_pre___ = bse_train_pre_.copy()
		base_test_pre___  = bse_test_pre_.copy()
		base_oot_pre___   = bse_oot_pre_.copy()

	else:

		base_train_pre___ = bse_train_pre_.copy()
		base_test_pre___  = bse_test_pre_.copy()
		base_oot_pre___   = bse_oot_pre_.copy()


	if algoritmo == 'lgbm' :
		 model1 = lgb.LGBMClassifier(
		 learning_rate = parametros['learning_rate'],
		 max_bin = parametros['max_bin'],
		 n_estimators = parametros['n_estimators'],
		 num_leaves = parametros['num_leaves'],
		 max_depth = parametros['max_depth'],
		 min_data_in_leaf = parametros['min_data_in_leaf'],
		 n_jobs = -3,
		 #silent = True,
		 verbose = -1,
		 seed = 42
		 )
		 model1.fit(base_train_pre___[feats_train], np.ravel(base_train_pre___[target]))

	elif algoritmo == 'hgb' :
		 model1 = HistGradientBoostingClassifier(
		 loss = parametros['loss'],
		 learning_rate = parametros['learning_rate'],
		 max_iter = parametros['max_iter'],
		 max_leaf_nodes = parametros['max_leaf_nodes'],
		 max_depth = parametros['max_depth'],
		 min_samples_leaf = parametros['min_samples_leaf'],
		 l2_regularization = parametros['l2_regularization'],
		 max_bins = parametros['max_bins'],
		 tol = parametros['tol'],
		 random_state = 42
		 )
		 model1.fit(base_train_pre___[feats_train], np.ravel(base_train_pre___[target]))

	return model1


########################################################################################################################
# fit_model_group CV
########################################################################################################################
def fit_model_group_cv(parametros, bse_train_test_pre_, bse_oot_pre_, target, feats_train, agrupacion, algoritmo):

	#res_group_df = pd.DataFrame()
	#res_df       = pd.DataFrame()

	#%%######################################################
	# Entrenamiento modelo
	#########################################################
	# Bases

	if (agrupacion == 'Comercio' or agrupacion == 'Infraestructura' or agrupacion == 'Manurfactura' or \
		agrupacion == 'RR_Naturales' or agrupacion == 'Servicios' or agrupacion == 'Edificaciones' or agrupacion == 'Agropecuario'):
		bse_train_test_pre_[bse_train_test_pre_['sector']==agrupacion]
		bse_oot_pre_[bse_oot_pre_['sector']==agrupacion]

		base_train_test_pre___ = bse_train_test_pre_.copy()
		base_oot_pre___   = bse_oot_pre_.copy()

	else:

		base_train_test_pre___ = bse_train_test_pre_.copy()
		base_oot_pre___   = bse_oot_pre_.copy()


	if algoritmo == 'lgbm' :
		 model1 = lgb.LGBMClassifier(
		 learning_rate = parametros['learning_rate'],
		 max_bin = parametros['max_bin'],
		 n_estimators = parametros['n_estimators'],
		 num_leaves = parametros['num_leaves'],
		 max_depth = parametros['max_depth'],
		 min_data_in_leaf = parametros['min_data_in_leaf'],
		 n_jobs = -3,
		 #silent = True,
		 verbose = -1,
		 seed = 42
		 )
		 model1.fit(base_train_test_pre___[feats_train], np.ravel(base_train_test_pre___[target]))

	elif algoritmo == 'hgb' :
		 model1 = HistGradientBoostingClassifier(
		 loss = parametros['loss'],
		 learning_rate = parametros['learning_rate'],
		 max_iter = parametros['max_iter'],
		 max_leaf_nodes = parametros['max_leaf_nodes'],
		 max_depth = parametros['max_depth'],
		 min_samples_leaf = parametros['min_samples_leaf'],
		 l2_regularization = parametros['l2_regularization'],
		 max_bins = parametros['max_bins'],
		 tol = parametros['tol']
		 )
		 model1.fit(base_train_test_pre___[feats_train], np.ravel(base_train_test_pre___[target]))

	return model1



########################################################################################################################
# Estas funciones permiten ejecutar el proceso de rescate de variables. Retorna la siguientes salidas :
# (1) numero de escalas ordenadas
# (2) numero de escalas en rango
# (3) tabla resumen con las escalas ordenas y en rango de todas las variables testeadas
# El rescate consiste en comparar las escalas mapeadas y ordenadas, entre la prediccion de un modelo que contenga
# la variable pura y la prediccion del mismo modelo pero fijando la variable en cero, es decir modificar la tabla que entra a
# ser calificada.
# Esto se hace iterativamente por cada una de las variables candidatas a ser rescatadas (se excluyen las que ya estan definidas
# en la bitacora).
# Para ello, es crea una primera funcion que calcula las metricas de desempeño de una prediccion dada y una segunda, que itera sobre
# cada candidata calculando las metricas de desempeño y realizando la comparacion.
#
# Nota aclaratoria: Este proceso no trata de crear un nuevo modelo con la alteracion de la variable, sino de calificar con un modelo
# ya construido (obtenido de la ronda de hiperparametros con todas las variables), una tabla que tenga la alteracion de la variable;
# la nueva calificacion arroja una distribucion de C's con la que se calcula las metricas de mapeo y el ordenamiento, y estas se comparan
# con las metricas de referencia, es decir las obtenidas por el modelo de la ronda de hiperparametros.
#
# Autor: morodrig.
# Modificado: wimunera 20220301.
########################################################################################################################

def metricas_propias(model1, feats_train, datos, nom_proba, var_default, var_conteo):

	# Genera la calificacion de un modelo, con un conjunto de datos deseado
	y_score_train  =  model1.predict_proba(datos[feats_train])
	y_score_train  =  [item[1] for item in y_score_train] # La calificacion la genera como un vector, por tanto se extrae el segundo elemento

	# Se asigna la calificacion a una columna de la tabla para usar la siguiente funcion
	datos[nom_proba] = y_score_train

	# Funcion que calcula las metricas de mapeo y ordenamiento de una calificacion especifica. Al revisar esta funcion (metricas_ordena_rango) se
	# puede notar que el primer paso trata de convertir a escala de C, la "probabilidad" arrojada en la calificacion del modelo
	c_entr, en_rango, ordena = metricas_ordena_rango(datos , nom_proba, var_default, var_conteo )

	return en_rango, ordena


def rescate_variables( datos, clf, feats_train, feats_candidatas, nom_proba='proba', var_default='def_default_12m', var_conteo='f_analisis' ):

	# Calcula las metricas de referencia correspondientes a la calificacion del modelo con todas las variables puras (sin alteraciones)
	b_mapea, b_ordena = metricas_propias(clf, feats_train, datos, nom_proba, var_default, var_conteo)

	# Se definen variables necesarias para esta funcion
	cont = 1
	scores={c:[] for c in feats_candidatas}

	# Se itera entre las variables candidatas a ser rescatadas. Para cada una, se fijara la variable en cero para todos los registros y se calculan las metricas para saber
	# cuanto contribuye la variable.
	# Nota: La estrategia de alteracion de la variable, fue discutida ampliamente entre usar el valor fijo de cero o reemplazar aleatoriamente. Finalmente se opto por la primera opcion.
	for c in feats_candidatas:

		X1 = datos.copy(deep=True) # Crea copia de la tabla de datos para luego alterarla y pasarla como parametro de la funcion que calcula las metricas
		X1[c] = 0 # Aplica la estrategia de alteracion de valor fijo cero solo a la variable candidata "c" de analisis.
		mapea_scores, ordena_scores = metricas_propias(clf, feats_train, X1, nom_proba, var_default, var_conteo) # Calcula las metricas correspondientes a la calificacion del modelo con la variable "c" alterada

		scores[c].append(b_mapea  - mapea_scores)  # Compara resultados entre referencia y nueva calificacion
		scores[c].append(b_ordena - ordena_scores) # Si la diferencia es positiva es porque el modelo base es mejor que el modelo danhado, por tanto disponer de la variable pura
												   # contribuye y se deberia dejar dentro del listado de variables definitivo.

		# Actualiza avance
		print( 'Finaliza analisis variable '+ str(cont) + "/" + str(len(feats_candidatas)) + ": " + c )
		cont += 1

	# Prepara la tabla final luego de iterar sobre todas las candidatas
	metrics = pd.DataFrame.from_dict(scores, orient='index')
	metrics.reset_index(inplace=True)
	metrics = metrics.rename(columns={'index': 'Variable',
								0: 'Mapeo_tr',
								1: 'Ordenamiento_tr'}
							)

	return b_mapea, b_ordena, metrics



def evaluar_modelo(paramters, algoritmo, base_train_pre, base_test_pre, base_oot_pre, target, feats_train, escala_c=True, q=19):

	#########################################################
	# Preparación Datos: Totales
	#########################################################
	_, X_train, y_train, _, X_test, y_test, _, X_oot, y_oot = prepara_bases(base_train_pre, base_test_pre, base_oot_pre, target, feats_train)

	model1 = fit_model_group(paramters, base_train_pre, base_test_pre, base_oot_pre, target, 
							feats_train, agrupacion='NA', algoritmo=algoritmo)

	nom_proba = 'proba'

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
	_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango( base_train_pre, nom_proba, target[0], 'f_analisis' )
	_, enrango_test, ordena_test, prct_desorden_test   = metricas_ordena_rango( base_test_pre,  nom_proba, target[0], 'f_analisis' )
	_, enrango_oot, ordena_oot, prct_desorden_oot     = metricas_ordena_rango( base_oot_pre,   nom_proba, target[0], 'f_analisis' )

	_,_,_,ordena_train_q, ordena_test_q, ordena_oot_q = metricas_ordena_q(base_train_pre, base_test_pre, base_oot_pre, q, nom_proba, target[0], 'f_analisis')


	auc_train = metrics.roc_auc_score( y_train, y_score_train )
	auc_test  = metrics.roc_auc_score( y_test,  y_score_test )
	auc_oot   = metrics.roc_auc_score( y_oot,   y_score_oot )

	resultados = {    'parameters': str(paramters),
					  'auc_tr': auc_train, 'auc_te': auc_test, 'auc_o': auc_oot,
					  'enrango_tr': enrango_train, 'enrango_te': enrango_test, 'enrango_o': enrango_oot,
					  'ordena_tr': ordena_train, 'ordena_te': ordena_test, 'ordena_o': ordena_oot,
					  'prct_desorden_tr': prct_desorden_train, 'prct_desorden_te': prct_desorden_test, 'prct_desorden_oot': prct_desorden_oot,
					  'ordena_tr_q': ordena_train_q, 'ordena_te_q': ordena_test_q, 'ordena_o_q': ordena_oot_q,
					  'modelo' : algoritmo
					  }
	res_df = pd.DataFrame(resultados, index = [0])

	res_df['auc_dif'] = round(abs(res_df['auc_tr'] - res_df['auc_te']), 4)
	res_df['ar_tr'] = 2*res_df['auc_tr'] - 1
	res_df['ar_te'] = 2*res_df['auc_te'] - 1
	res_df['ar_o']  = 2*res_df['auc_o']  - 1
	res_df['enrango'] = res_df['enrango_tr'] * res_df['enrango_te']
	res_df['ordena'] = res_df['ordena_tr'] * res_df['ordena_te']
	res_df['ordena_q'] = res_df['ordena_tr_q'] * res_df['ordena_te_q']

	columnas = ['parameters', 'modelo',
				'ar_tr','ar_te','ar_o', 'auc_dif',
				'enrango','enrango_tr', 'enrango_te', 'enrango_o', 
				'ordena', 'ordena_tr', 'ordena_te', 'ordena_o', 
				'prct_desorden_tr', 'prct_desorden_te', 'prct_desorden_oot']
	
	if escala_c == False : 
		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif',
					'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o',
					'prct_desorden_tr','prct_desorden_te', 'prct_desorden_oot',
					'ordena_q','ordena_tr_q', 'ordena_te_q', 'ordena_o_q' ]
	else:
		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif',
					'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o']

	return res_df[columnas]



########################################################################################################################
### sargumed: 2022-05-16 Se crea una función para revisar el ordenamiento de las variables en un modelo 
## Revisando tanto la pdp promedio como la individuales de cada registro
########################################################################################################################
def ordenamiento_riesgo( df_train, feats_total, modelo, feats_ordena_riesgo ):
	import numpy as np
	from pdpbox import pdp
	from scipy import stats

	result_ordenamiento = pd.DataFrame()

	for f in feats_total:
		pdp_feat_df = pdp.pdp_isolate(model=modelo, dataset=df_train, model_features=feats_total, feature=f)

		df = pd.DataFrame.from_dict({'valor':[x for x in pdp_feat_df.feature_grids],
									 'pdp':[x for x in pdp_feat_df.pdp],
									 'percentile_info':[x for x in pdp_feat_df.percentile_info]})
		df['variable'] = pdp_feat_df.feature
		df['signo'] = np.where(df['pdp'].diff() > 0, "Crece", np.where(df['pdp'].diff() < 0, "Decrece", np.where(df['pdp'].diff() == 0, "No cambia", np.NaN)))

		df_ice = pdp_feat_df.ice_lines.copy()
		df_ice['slope'] = df_ice.apply(lambda row : stats.linregress(df_ice.columns, y=row*100)[0].round(4), axis=1)

		result_ordenamiento =  result_ordenamiento.append(pd.DataFrame.from_dict({'feats': [f], 
														 'crece': [df.loc[df['signo'] == 'Crece', 'signo'].count()],
														 'decrece': [df.loc[df['signo'] == 'Decrece', 'signo'].count()], 
														 'no_cambia': [df.loc[df['signo'] == 'No cambia', 'signo'].count()], 
														 'pendiente': stats.linregress(df['valor'], df['pdp']*100 )[0].round(3),
														 'min_pdp': [df.pdp.min()], 
														 'max_pdp': [df.pdp.max()],
														 'cli_pendiente_positiva': [df_ice.loc[df_ice['slope'] > 0, 'slope'].count()], 
														 'cli_pendiente_negativa': df_ice.loc[df_ice['slope'] < 0, 'slope'].count(),
														 'riesgo_esperado': [feats_ordena_riesgo.get(f, None)]}))

	return result_ordenamiento


########################################################################################################################
### sargumed: 2022-05-16 Se crea una función para revisar el ordenamiento de las variables en varios modelos
## Revisando solo la pdp promedio 
########################################################################################################################
def ordenamiento_riesgo_models( df_train, feats_total, models, feats_ordena_riesgo, super_feats_ordena_riesgo, feats_bitacora, apariciones_incorrectas, decision_indet_acida = False):
	import numpy as np
	import pandas as pd
	from pdpbox import pdp
	from scipy import stats
	
	result_ordenamiento = pd.DataFrame()
	for key in models:
		print('Analizando '+ key + '...')
		for f in feats_total:
			pdp_feat_df = pdp.pdp_isolate(model=models[key], dataset=df_train, model_features=feats_total, feature=f)

			df = pd.DataFrame.from_dict({'valor':[x for x in pdp_feat_df.feature_grids],
										 'pdp':[x for x in pdp_feat_df.pdp]})
			df['variable'] = pdp_feat_df.feature
			df['signo'] = np.where(df['pdp'].diff() > 0, "Crece", np.where(df['pdp'].diff() < 0, "Decrece", np.where(df['pdp'].diff() == 0, "No cambia", np.NaN)))

			result_ordenamiento =  result_ordenamiento.append(pd.DataFrame.from_dict({'feats': [f],
																					  'Model': [key],
																					  'crece': [df.loc[df['signo'] == 'Crece', 'signo'].count()],
																					  'decrece': [df.loc[df['signo'] == 'Decrece', 'signo'].count()],
																					  'no_cambia': [df.loc[df['signo'] == 'No cambia', 'signo'].count()], 
																					  'pendiente': [stats.linregress(df['valor'], df['pdp']*100 )[0].round(3)],
																					  'riesgo_esperado': [feats_ordena_riesgo.get(f, None)]
																					 } ))
	print('Finalizado')

	## Definicion variable en bitacora
	result_ordenamiento["feats_bitacora"] = np.where(result_ordenamiento["feats"].isin(feats_bitacora), "Si", "No")
	feats_supervars = list(super_feats_ordena_riesgo.keys())
	result_ordenamiento["feats_supervars"] = np.where(result_ordenamiento["feats"].isin(feats_supervars), "Si", "No")
	result_ordenamiento[['riesgo_esperado']] = result_ordenamiento[['riesgo_esperado']].fillna(value='Indeterminado')

	if (decision_indet_acida == True):
		## Definicion variable analisis - Acido
		result_ordenamiento['analisis'] = np.where( (result_ordenamiento['riesgo_esperado'] == 'AumentaU0'), "Revisar manual",
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento['crece'] > 0) & (result_ordenamiento['decrece'] == 0), "Indeterminado_Crece",
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento['crece'] == 0) & (result_ordenamiento['decrece'] > 0), "Indeterminado_Decrece", 
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento['crece'] > 0) & (result_ordenamiento['decrece'] > 0), "Indeterminado_Crece_Decrece", 
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Aumenta') & (result_ordenamiento['crece'] > 0) & (result_ordenamiento['decrece'] == 0), "Correcto", 
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Disminuye') & (result_ordenamiento['decrece'] > 0) & (result_ordenamiento['crece'] == 0), "Correcto", 
										np.where( (result_ordenamiento['decrece'] == 0) & (result_ordenamiento['crece'] == 0) , "Indiferente",  
										'Incorrecto')))))))

		## Tabla resumen por variable
		result_ordenamiento_by_feat = pd.pivot_table(result_ordenamiento, values='Model', index=['feats','feats_bitacora','feats_supervars','riesgo_esperado'], columns=['analisis'], aggfunc='count', fill_value=0).reset_index()
		
		if( 'Correcto' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Correcto'] = 0
		if( 'Incorrecto' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Incorrecto'] = 0
		if( 'Indiferente' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Indiferente'] = 0
		if( 'Revisar manual' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Revisar manual'] = 0

		
		result_ordenamiento_by_feat['decision'] = np.where( result_ordenamiento_by_feat['Revisar manual'] > 0, 'Revisar manual',
												np.where( result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado', 'Indeterminado',
												np.where( result_ordenamiento_by_feat['Correcto'] >= (len(models)/2 + 1) , 'Aceptar', 
												np.where( result_ordenamiento_by_feat['Incorrecto'] >= apariciones_incorrectas , 'No aceptar', 'Indiferente' ))))
		
		## Crear columnas si no se tienen
		if( 'Indeterminado_Crece' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Indeterminado_Crece'] = 0
		if( 'Indeterminado_Decrece' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Indeterminado_Decrece'] = 0
		if( 'Indeterminado_Crece_Decrece' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Indeterminado_Crece_Decrece'] = 0

		## Decision sobre indeterminado - Acido
		result_ordenamiento_by_feat['decision_indeterminado'] = np.where( (result_ordenamiento_by_feat['riesgo_esperado'] != 'Indeterminado') , '',
																np.where( (result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento_by_feat['Indeterminado_Crece'] > 0) & (result_ordenamiento_by_feat['Indeterminado_Decrece'] > 0 ), 'No aceptar',
																np.where( (result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento_by_feat['Indeterminado_Crece_Decrece'] > 0), 'No aceptar',
																np.where( (result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento_by_feat['Indeterminado_Crece'] >= (len(models)/2 + 1)) & (result_ordenamiento_by_feat['Indeterminado_Decrece'] == 0), 'Aceptar_Aumenta',
																np.where( (result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento_by_feat['Indeterminado_Crece'] == 0) & (result_ordenamiento_by_feat['Indeterminado_Decrece'] >= (len(models)/2 + 1)), 'Aceptar_Disminuye',
																'Indiferente')))))
	
	else:
		## Definicion variable analisis - No acido
		result_ordenamiento['analisis'] = np.where( (result_ordenamiento['riesgo_esperado'] == 'AumentaU0'), "Revisar manual",
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento['crece'] > 0) & (result_ordenamiento['decrece'] == 0), "Indeterminado_Crece",
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento['crece'] == 0) & (result_ordenamiento['decrece'] > 0), "Indeterminado_Decrece", 
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento['crece'] > 0) & (result_ordenamiento['decrece'] > 0), "Indeterminado_Crece_Decrece", 
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Aumenta') & (result_ordenamiento['crece'] > 0) & (result_ordenamiento['decrece'] == 0), "Correcto", 
										np.where( (result_ordenamiento['riesgo_esperado'] == 'Disminuye') & (result_ordenamiento['decrece'] > 0) & (result_ordenamiento['crece'] == 0), "Correcto", 
										np.where( (result_ordenamiento['decrece'] == 0) & (result_ordenamiento['crece'] == 0) , "Indiferente",  
										'Incorrecto')))))))

		## Tabla resumen por variable
		result_ordenamiento_by_feat = pd.pivot_table(result_ordenamiento, values='Model', index=['feats','feats_bitacora','feats_supervars','riesgo_esperado'], columns=['analisis'], aggfunc='count', fill_value=0).reset_index()
		
		if( 'Correcto' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Correcto'] = 0
		if( 'Incorrecto' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Incorrecto'] = 0
		if( 'Indiferente' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Indiferente'] = 0
		if( 'Revisar manual' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Revisar manual'] = 0

		
		result_ordenamiento_by_feat['decision'] = np.where( result_ordenamiento_by_feat['Revisar manual'] > 0, 'Revisar manual',
												np.where( result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado', 'Indeterminado',
												np.where( result_ordenamiento_by_feat['Correcto'] >= (len(models)/2 + 1) , 'Aceptar', 
												np.where( result_ordenamiento_by_feat['Incorrecto'] >= apariciones_incorrectas , 'No aceptar', 'Indiferente' ))))
		
		## Crear columnas si no se tienen
		if( 'Indeterminado_Crece' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Indeterminado_Crece'] = 0
		if( 'Indeterminado_Decrece' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Indeterminado_Decrece'] = 0
		if( 'Indeterminado_Crece_Decrece' not in result_ordenamiento_by_feat.columns ) : result_ordenamiento_by_feat['Indeterminado_Crece_Decrece'] = 0

		## Decision sobre indeterminado - No Acido
		result_ordenamiento_by_feat['decision_indeterminado'] = np.where( (result_ordenamiento_by_feat['riesgo_esperado'] != 'Indeterminado') , '',
																np.where( (result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento_by_feat['Indeterminado_Crece'] >= apariciones_incorrectas) & (result_ordenamiento_by_feat['Indeterminado_Decrece'] >= apariciones_incorrectas), 'No aceptar',
																np.where( (result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento_by_feat['Indeterminado_Crece_Decrece'] > apariciones_incorrectas), 'No aceptar',
																np.where( (result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento_by_feat['Indeterminado_Crece'] >= (len(models)/2 + 1)) & (result_ordenamiento_by_feat['Indeterminado_Decrece'] <= apariciones_incorrectas), 'Aceptar_Aumenta',
																np.where( (result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado') & (result_ordenamiento_by_feat['Indeterminado_Crece'] <= apariciones_incorrectas) & (result_ordenamiento_by_feat['Indeterminado_Decrece'] >= (len(models)/2 + 1)), 'Aceptar_Disminuye',
																'Indiferente')))))

	## Unificar en la misma columna de decision
	result_ordenamiento_by_feat.loc[ result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado' , "decision"] = result_ordenamiento_by_feat.loc[ result_ordenamiento_by_feat['riesgo_esperado'] == 'Indeterminado', "decision_indeterminado"]

	## Ordenar tabla
	result_ordenamiento_by_feat = result_ordenamiento_by_feat.reindex(columns = ['feats','feats_bitacora','feats_supervars','riesgo_esperado','Correcto','Incorrecto','Indiferente','Revisar manual','Indeterminado_Crece','Indeterminado_Decrece','Indeterminado_Crece_Decrece', 'decision'])
	result_ordenamiento_by_feat = result_ordenamiento_by_feat.sort_values(by = ['decision', 'riesgo_esperado'])

	return result_ordenamiento, result_ordenamiento_by_feat


########################################################################################################################
# Función para categorizar variables, de tal forma que ordenen de acuerdo al riesgo esperado
########################################################################################################################
def categorical_feats(df, feat_dict):
	from optbinning import OptimalBinning
	for key in feat_dict:
		x = df[key].values
		y = df["def_default_12m"]
		if feat_dict[key] == "Aumenta":
			riesgo = "ascending"
		elif feat_dict[key] == "Disminuye":
			riesgo = "descending"
		else:
			riesgo = "auto_asc_desc"
		optb = OptimalBinning(name=key, dtype="numerical", monotonic_trend = riesgo)
		optb.fit(x, y)
		x_transform_indices = optb.transform(x, metric="indices")
		df["cat_" + key] = x_transform_indices
		x_transform_bins = optb.transform(x, metric="bins")
		df["bins_" + key] = x_transform_bins
	return df.copy()


########################################################################################################################
### sargumed: Esta función permite ejecutar una iteración y devuelve lo siguiente:
# ronda: Dataframe con los modelos entrenados y sus métricas
# models: Mejores modelos en ordenamiento test y diferencia en auc(train, test)
# result_ordenamiento, result_ordenamiento_by_feat: Resultado del ordenamiento de las variables en los mejores modelos
########################################################################################################################
def get_iteration(base_final, feats_total, feats_ordena_riesgo, super_feats_ordena_riesgo, feats_bitacora, n_iteraciones = 500, n_top_models = 10, apariciones_incorrectas_ = 3, escala_c_ = True, q_ = 20, decision_indet_acida_ = False):
	base_train_pre = base_final.query('train_test_e2 == 1').copy()
	base_test_pre  = base_final.query('train_test_e2 == 0').copy()
	base_oot_pre   = base_final.query('train_test_e2 == 2').copy()

	print('Train')
	print(base_train_pre.shape)
	print('Test')
	print(base_test_pre.shape)
	print('OOT')
	print(base_oot_pre.shape)

	feats_ids = ['num_doc','tipo_doc','llave_nombre','f_analisis']

	# Variable respuesta
	target = ['def_default_12m']
	y_train = base_train_pre[target]
	print('Variable respuesta: ' + str(y_train.shape))

	muestras_en_intervalo = 50
	seed = 42
	folds = 3
	n_iter = n_iteraciones

	learning_ra = list(set([float(x) for x in np.linspace(start = 0.001, stop = 0.8, num = muestras_en_intervalo)]))
	estimadores = list(set([int(x) for x in np.linspace(10, 200, num = muestras_en_intervalo)]))
	profundidad = list(set([int(x) for x in np.linspace(1, 20, num = muestras_en_intervalo)]))
	min_data_le = [int(x) for x in np.linspace(start = 0.01*base_train_pre.shape[0], stop = 0.5*base_train_pre.shape[0], num = muestras_en_intervalo)]
	max_bins___ = list(set([int(x) for x in np.linspace(4, 200, num = muestras_en_intervalo)]))
	n_leaves___ = list(set([int(x) for x in np.linspace(2, 200 , num = muestras_en_intervalo)]))

	parametros = {'learning_ra': learning_ra,
					'estimadores': estimadores, 
					'profundidad': profundidad, 
					'min_data_le': min_data_le, 
					'max_bins___': max_bins___, 
					'n_leaves___': n_leaves___ 
					}
	print("\n")
	print("Entrenando modelos: \n")
	time_1 = time.time()
	ronda = train_nueva_era_segint(base_train_pre, base_test_pre, base_oot_pre, target, \
									n_iter, seed, feats_train = feats_total, parametros = parametros, \
									muestras_en_intervalo = muestras_en_intervalo , escala_c = escala_c_, q = q_) 
	time_2 = time.time()
	print('Tiempo total ejecución entrenamiento: ', str((time_2 - time_1)/60))
	print("\n")
	print("Seleccionando los mejores modelos: \n")

	if escala_c_ :
		modelos = metricas_top_models( base_train_pre[feats_total], np.ravel(y_train), base_train_pre, base_test_pre, base_oot_pre, feats_total, ronda, num_top_models = n_top_models )
	else:
		modelos = metricas_top_models_q( base_train_pre[feats_total], np.ravel(y_train), base_train_pre, base_test_pre, base_oot_pre, feats_total, ronda, num_top_models = n_top_models, ronda_completa = False, var1= ["auc_dif", 0.05, True], var2 = ["ordena_te_q", 14, False], q = 20 )

	models = {}
	i = 0
	for tv, model in modelos:
		models["m"+str(i)] = model 
		i+=1
	print("\n")
	print("Revisando el ordenamiento de las variables: \n")
	time_1 = time.time()
	result_ordenamiento, result_ordenamiento_by_feat = ordenamiento_riesgo_models( base_train_pre, feats_total, models, feats_ordena_riesgo, super_feats_ordena_riesgo, feats_bitacora, apariciones_incorrectas=apariciones_incorrectas_, decision_indet_acida = decision_indet_acida_)
	time_2 = time.time()
	print('Tiempo total ejecución ordenamiento: ', str((time_2 - time_1)/60))

	return ronda, models, result_ordenamiento, result_ordenamiento_by_feat

########################################################################################################################
### andremej, jaimolin, mebernal: Esta función permite ejecutar una iteración y devuelve lo siguiente:
# ronda: Dataframe con los modelos entrenados y sus métricas
# models: Mejores modelos en ordenamiento test y diferencia en auc(train, test)
# result_ordenamiento, result_ordenamiento_by_feat: Resultado del ordenamiento de las variables en los mejores modelos
# tiene como diferencia con la anterior la ejecución por defecto en escala q, y el llamado de la función train_nueva_era_segint_full_sector
# la cual genera los modelos realizando entrenamiento con todos los sectores y evaluando las metricas solo en el sector específico
########################################################################################################################

def get_iteration_full_sector(sector, base_final, feats_total, feats_ordena_riesgo, super_feats_ordena_riesgo, feats_bitacora, n_iteraciones = 500, n_top_models = 10, apariciones_incorrectas_ = 3, escala_c_ = False, q_ = 10, decision_indet_acida_ = False):
	base_train_pre = base_final.query('train_test_e2 == 1').copy()
	base_test_pre  = base_final.query('train_test_e2 == 0').copy()
	base_oot_pre   = base_final.query('train_test_e2 == 2').copy()

	# Base completa sin OOT
	base_final_aux = base_final.query('train_test_e2 != 2') 

	print('Train')
	print(base_train_pre.shape)
	print('Test')
	print(base_test_pre.shape)
	print('OOT')
	print(base_oot_pre.shape)

	feats_ids = ['num_doc','tipo_doc','llave_nombre','f_analisis']

	# Variable respuesta
	target = ['def_default_12m']
	y_train = base_train_pre[target]
	print('Variable respuesta: ' + str(y_train.shape))

	muestras_en_intervalo = 50
	seed = 42
	folds = 3
	n_iter = n_iteraciones

	learning_ra = list(set([float(x) for x in np.linspace(start = 0.001, stop = 0.8, num = muestras_en_intervalo)]))
	estimadores = list(set([int(x) for x in np.linspace(10, 200, num = muestras_en_intervalo)]))
	profundidad = list(set([int(x) for x in np.linspace(1, 20, num = muestras_en_intervalo)]))
	min_data_le = [int(x) for x in np.linspace(start = 0.01*base_train_pre.shape[0], stop = 0.5*base_train_pre.shape[0], num = muestras_en_intervalo)]
	max_bins___ = list(set([int(x) for x in np.linspace(4, 200, num = muestras_en_intervalo)]))
	n_leaves___ = list(set([int(x) for x in np.linspace(2, 200 , num = muestras_en_intervalo)]))

	parametros = {'learning_ra': learning_ra,
					'estimadores': estimadores, 
					'profundidad': profundidad, 
					'min_data_le': min_data_le, 
					'max_bins___': max_bins___, 
					'n_leaves___': n_leaves___ 
					}
	print("\n")
	print("Entrenando modelos: \n")
	time_1 = time.time()
	ronda = train_nueva_era_segint_full_sector(base_train_pre, base_test_pre, base_oot_pre, sector, target, \
									n_iter, seed, feats_train = feats_total, parametros = parametros, \
									muestras_en_intervalo = muestras_en_intervalo , escala_c = escala_c_, q = q_) 
	time_2 = time.time()
	print('Tiempo total ejecución entrenamiento: ', str((time_2 - time_1)/60))
	print("\n")
	print("Seleccionando los mejores modelos: \n")

	if escala_c_ :
		modelos = metricas_top_models( base_train_pre[feats_total], np.ravel(y_train), base_train_pre.query('cli_sec_riesgo_{} == 1'.format(sector)), base_test_pre, base_oot_pre, feats_total, ronda, num_top_models = n_top_models )
	else:
		modelos = metricas_top_models_q( base_train_pre[feats_total], np.ravel(y_train), base_train_pre.query('cli_sec_riesgo_{} == 1'.format(sector)), base_test_pre, base_oot_pre, feats_total, ronda, num_top_models = n_top_models, ronda_completa = True, var1= ["auc_dif", 0.05, True], var2 = ["ordena_te_q", 8, False], q = 10, estricto = True)

	models = {}
	i = 0
	for tv, model in modelos:
		models["m"+str(i)] = model 
		i+=1
	print("\n")
	print("Revisando el ordenamiento de las variables: \n")
	time_1 = time.time()
	result_ordenamiento, result_ordenamiento_by_feat = ordenamiento_riesgo_models( base_final_aux, feats_total, models, feats_ordena_riesgo, super_feats_ordena_riesgo, feats_bitacora, apariciones_incorrectas=apariciones_incorrectas_, decision_indet_acida = decision_indet_acida_)
	time_2 = time.time()
	print('Tiempo total ejecución ordenamiento: ', str((time_2 - time_1)/60))

	return ronda, models, result_ordenamiento, result_ordenamiento_by_feat


########################################################################################################################
### sargumed/mpaz/wimunera: Esta función permite ejecutar las estrategias de iteración
########################################################################################################################
def iteration(sector, base_final_cat, feats_total, feats_ordena_riesgo, super_feats_ordena_riesgo, feats_bitacora, n_iteraciones_ = 500, n_top_models_ = 10, apariciones_incorrectas__ = 3, estrategia_ = 1,  escala_c__ = True, q__ = 20, decision_indet_acida__ = False):

	### Inicializacion
	import sys, os
	counter = 0
	num_iteracion = 0
	num_iteracion_anterior = 0
	basepath = os.path.dirname(sys.path[0])
	counter_cat = 0
	cat_superfeats_ordena_riesgo = super_feats_ordena_riesgo.copy()
	for key in super_feats_ordena_riesgo:
		new_key = "cat_" + key
		cat_superfeats_ordena_riesgo[new_key] = cat_superfeats_ordena_riesgo.pop(key)

	### Bucle que itera hasta que se cumpla el criterio de parada
	while True:
		if counter == 0:
			print("\n")
			print("\033[1m Estás en la Iteración No. {} \033[0m".format(str(num_iteracion)))
			print("\n")

			### Realizar iteracion
			ronda_, models_, result_ordenamiento_, result_ordenamiento_feat_ = get_iteration(base_final_cat, feats_total, {**feats_ordena_riesgo, **cat_superfeats_ordena_riesgo}, super_feats_ordena_riesgo, feats_bitacora, n_iteraciones = n_iteraciones_, n_top_models = n_top_models_, apariciones_incorrectas_ = apariciones_incorrectas__, escala_c_ = escala_c__, q_ = q__, decision_indet_acida_ = decision_indet_acida__)

			### Crear directorio y guardar resultados
			path = os.path.join(basepath + '\\02-Resultados\\' + str(estrategia_)+'-Estrategia', str(num_iteracion)+'-Iteracion')
			if not os.path.exists(path):
				os.makedirs(path)
			# Se guarda el resultado de la ronda    
			ronda_.to_excel(path + '\\rb_{}_modelos_iteracion_{}.xlsx'.format(sector.lower(), num_iteracion))
			# Se guarda el resultado del ordenamiento por variable y modelos
			result_ordenamiento_.sort_values(by="feats").to_excel(path + '\\rb_{}_ordenamiento_modelos_iteracion_{}.xlsx'.format(sector.lower(), num_iteracion), index=False)
			# Se guarda el resultado del ordenamiento por variable y su decisión 
			result_ordenamiento_feat_.to_excel(path + '\\rb_{}_ordenamiento_feats_iteracion_{}.xlsx'.format(sector.lower(), num_iteracion), index=False)

			### Variables que quedaron en No aceptar:
			feats_no_aceptar = result_ordenamiento_feat_.loc[((result_ordenamiento_feat_['decision'].isin(['No aceptar'])) ),"feats"].tolist()

			### Actualizar feats_total segun estrategias
			if(estrategia_ == 1):
				feats_drop_ = feats_no_aceptar.copy()
				if(len(feats_drop_) > 0):
					counter = 0 
					num_iteracion_anterior = num_iteracion
					num_iteracion += 1
					# Ahora se arma un nuevo feats_total descartanto las variables que quedaron en feats_drop_
					feats_total = [i for i in feats_total if i not in feats_drop_]
				else: 
					print("\033[1m Hemos finalizado. No hay más variables en no aceptar. \033[0m")
					break
					
			elif(estrategia_ == 2):

				### Listas de variables no aceptar, segun supervars o no supervars
				# No aceptar y no son supervars
				feats_drop_ = [ x for x in feats_no_aceptar if x not in (list(super_feats_ordena_riesgo.keys()) + list(cat_superfeats_ordena_riesgo.keys()) ) ]
				# No aceptar y son supervars
				superfeats_no_aceptar_ = [ x for x in feats_no_aceptar if x in super_feats_ordena_riesgo.keys() ] + [x for x in feats_no_aceptar if x in cat_superfeats_ordena_riesgo.keys() ]

				### Criterio de parada si no aceptar son supervars
				if(len((superfeats_no_aceptar_ + feats_drop_))>0):
					check =  all(item in cat_superfeats_ordena_riesgo.keys() for item in (superfeats_no_aceptar_ + feats_drop_) )
					if check is True:
						print("\033[1m Todas las variables que quedaron en no aceptar están categorizadas, no podemos continuar iterando. \033[0m")
						break

				if( (len(feats_drop_) > 0) & (counter_cat == 0 ) ): 
					counter = 0 
					num_iteracion_anterior = num_iteracion
					num_iteracion += 1
					# Ahora se arma un nuevo feats_total descartanto las variables que quedaron en feats_drop_
					feats_total = [i for i in feats_total if i not in feats_drop_]
				elif( (len(feats_drop_) == 0) & (len(superfeats_no_aceptar_) > 0) & (counter_cat == 0 ) ):
					print("\033[1m En este punto(No. Iteración {}), solo supervars quedan en no aceptar. Ahora, vamos a categorizar y seguir iterando \033[0m".format(str(num_iteracion)) )
					counter = 0
					num_iteracion_anterior = num_iteracion
					num_iteracion += 1
					# Dado que en "No aceptar" quedan solo supervars, éstas se categorizan
					superfeats_no_aceptar_cat_ = ['cat_' + x for x in superfeats_no_aceptar_ if x in super_feats_ordena_riesgo.keys() ] + [x for x in superfeats_no_aceptar_ if x in cat_superfeats_ordena_riesgo.keys()]
					# Ahora se arma un nuevo feats_total descartando las supervars que quedaron en "No aceptar" y dejandolas categorizadas
					feats_total = [x for x in feats_total if x not in superfeats_no_aceptar_ ] + superfeats_no_aceptar_cat_
					counter_cat = 1
				elif( ( (len(feats_drop_) > 0) | (len(superfeats_no_aceptar_) > 0)) & (counter_cat == 1) ):
					# Cuando entre a este punto, ya hay supervars categorizadas
					counter = 0
					num_iteracion_anterior = num_iteracion
					num_iteracion += 1
					superfeats_no_aceptar_cat_ = ['cat_' + x for x in superfeats_no_aceptar_ if x in super_feats_ordena_riesgo.keys() ] + [x for x in superfeats_no_aceptar_ if x in cat_superfeats_ordena_riesgo.keys()]
					feats_total = [x for x in feats_total if x not in (feats_drop_ + superfeats_no_aceptar_) ] + superfeats_no_aceptar_cat_
				else:
					print("\033[1m Hemos finalizado el proceso en la iteración No. {} \033[0m".format(str(num_iteracion)))
					break
			else: 
				print("\033[1m Hemos finalizado el proceso en la iteración No. {} \033[0m".format(str(num_iteracion)))
				break
		else:
			print("\033[1m Hemos finalizado el proceso en la iteración No. {} \033[0m".format(str(num_iteracion)))
			break
	return ronda_, result_ordenamiento_, result_ordenamiento_feat_

########################################################################################################################
### andremej/jaimolin/mebernal: Esta función permite ejecutar las estrategias de iteración con las condiciones específicas
# de los modelos de pocos datos (entrenamiento con base completa, metricas del sector, Q = 10, entre otros)
########################################################################################################################
def iteration_full_sector(sector, base_final_cat, feats_total, feats_ordena_riesgo, super_feats_ordena_riesgo, feats_bitacora, n_iteraciones_ = 500, n_top_models_ = 10, apariciones_incorrectas__ = 3, estrategia_ = 1,  escala_c__ = False, q__ = 10, decision_indet_acida__ = False):

	### Inicializacion
	import sys, os
	counter = 0
	num_iteracion = 0
	num_iteracion_anterior = 0
	basepath = os.path.dirname(sys.path[0])
	counter_cat = 0
	cat_superfeats_ordena_riesgo = super_feats_ordena_riesgo.copy()
	for key in super_feats_ordena_riesgo:
		new_key = "cat_" + key
		cat_superfeats_ordena_riesgo[new_key] = cat_superfeats_ordena_riesgo.pop(key)

	### Bucle que itera hasta que se cumpla el criterio de parada
	while True:
		if counter == 0:
			print("\n")
			print("\033[1m Estás en la Iteración No. {} \033[0m".format(str(num_iteracion)))
			print("\n")

			### Realizar iteracion
			ronda_, models_, result_ordenamiento_, result_ordenamiento_feat_ = get_iteration_full_sector(sector, base_final_cat, feats_total, {**feats_ordena_riesgo, **cat_superfeats_ordena_riesgo}, super_feats_ordena_riesgo, feats_bitacora, n_iteraciones = n_iteraciones_, n_top_models = n_top_models_, apariciones_incorrectas_ = apariciones_incorrectas__, escala_c_ = escala_c__, q_ = q__, decision_indet_acida_ = decision_indet_acida__)

			### Crear directorio y guardar resultados
			path = os.path.join(basepath + '\\02-Resultados\\' + str(estrategia_)+'-Estrategia', str(num_iteracion)+'-Iteracion')
			if not os.path.exists(path):
				os.makedirs(path)
			# Se guarda el resultado de la ronda    
			ronda_.to_excel(path + '\\rb_{}_modelos_iteracion_{}.xlsx'.format(sector.lower(), num_iteracion))
			# Se guarda el resultado del ordenamiento por variable y modelos
			result_ordenamiento_.sort_values(by="feats").to_excel(path + '\\rb_{}_ordenamiento_modelos_iteracion_{}.xlsx'.format(sector.lower(), num_iteracion), index=False)
			# Se guarda el resultado del ordenamiento por variable y su decisión 
			result_ordenamiento_feat_.to_excel(path + '\\rb_{}_ordenamiento_feats_iteracion_{}.xlsx'.format(sector.lower(), num_iteracion), index=False)

			### Variables que quedaron en No aceptar:
			feats_no_aceptar = result_ordenamiento_feat_.loc[((result_ordenamiento_feat_['decision'].isin(['No aceptar'])) ),"feats"].tolist()

			### Actualizar feats_total segun estrategias
			if(estrategia_ == 1):
				feats_drop_ = feats_no_aceptar.copy()
				if(len(feats_drop_) > 0):
					counter = 0 
					num_iteracion_anterior = num_iteracion
					num_iteracion += 1
					# Ahora se arma un nuevo feats_total descartanto las variables que quedaron en feats_drop_
					feats_total = [i for i in feats_total if i not in feats_drop_]
				else: 
					print("\033[1m Hemos finalizado. No hay más variables en no aceptar. \033[0m")
					break
					
			elif(estrategia_ == 2):

				### Listas de variables no aceptar, segun supervars o no supervars
				# No aceptar y no son supervars
				feats_drop_ = [ x for x in feats_no_aceptar if x not in (list(super_feats_ordena_riesgo.keys()) + list(cat_superfeats_ordena_riesgo.keys()) ) ]
				# No aceptar y son supervars
				superfeats_no_aceptar_ = [ x for x in feats_no_aceptar if x in super_feats_ordena_riesgo.keys() ] + [x for x in feats_no_aceptar if x in cat_superfeats_ordena_riesgo.keys() ]

				### Criterio de parada si no aceptar son supervars
				if(len((superfeats_no_aceptar_ + feats_drop_))>0):
					check =  all(item in cat_superfeats_ordena_riesgo.keys() for item in (superfeats_no_aceptar_ + feats_drop_) )
					if check is True:
						print("\033[1m Todas las variables que quedaron en no aceptar están categorizadas, no podemos continuar iterando. \033[0m")
						break

				if( (len(feats_drop_) > 0) & (counter_cat == 0 ) ): 
					counter = 0 
					num_iteracion_anterior = num_iteracion
					num_iteracion += 1
					# Ahora se arma un nuevo feats_total descartanto las variables que quedaron en feats_drop_
					feats_total = [i for i in feats_total if i not in feats_drop_]
				elif( (len(feats_drop_) == 0) & (len(superfeats_no_aceptar_) > 0) & (counter_cat == 0 ) ):
					print("\033[1m En este punto(No. Iteración {}), solo supervars quedan en no aceptar. Ahora, vamos a categorizar y seguir iterando \033[0m".format(str(num_iteracion)) )
					counter = 0
					num_iteracion_anterior = num_iteracion
					num_iteracion += 1
					# Dado que en "No aceptar" quedan solo supervars, éstas se categorizan
					superfeats_no_aceptar_cat_ = ['cat_' + x for x in superfeats_no_aceptar_ if x in super_feats_ordena_riesgo.keys() ] + [x for x in superfeats_no_aceptar_ if x in cat_superfeats_ordena_riesgo.keys()]
					# Ahora se arma un nuevo feats_total descartando las supervars que quedaron en "No aceptar" y dejandolas categorizadas
					feats_total = [x for x in feats_total if x not in superfeats_no_aceptar_ ] + superfeats_no_aceptar_cat_
					counter_cat = 1
				elif( ( (len(feats_drop_) > 0) | (len(superfeats_no_aceptar_) > 0)) & (counter_cat == 1) ):
					# Cuando entre a este punto, ya hay supervars categorizadas
					counter = 0
					num_iteracion_anterior = num_iteracion
					num_iteracion += 1
					superfeats_no_aceptar_cat_ = ['cat_' + x for x in superfeats_no_aceptar_ if x in super_feats_ordena_riesgo.keys() ] + [x for x in superfeats_no_aceptar_ if x in cat_superfeats_ordena_riesgo.keys()]
					feats_total = [x for x in feats_total if x not in (feats_drop_ + superfeats_no_aceptar_) ] + superfeats_no_aceptar_cat_
				else:
					print("\033[1m Hemos finalizado el proceso en la iteración No. {} \033[0m".format(str(num_iteracion)))
					break
			else: 
				print("\033[1m Hemos finalizado el proceso en la iteración No. {} \033[0m".format(str(num_iteracion)))
				break
		else:
			print("\033[1m Hemos finalizado el proceso en la iteración No. {} \033[0m".format(str(num_iteracion)))
			break
	return ronda_, result_ordenamiento_, result_ordenamiento_feat_

########################################################################################################################
#mialopez 17/06/2022 Graficas de resumen para la iteraciones de seleción de variables. 
########################################################################################################################
def resumen_iteraciones(ruta,sector):
	import os
	import pandas as pd
	files = folders = 0
	for _, dirnames, filenames in os.walk(ruta):
		# ^ this idiom means "we won't be using this value"
		folders += len(dirnames) 
	feats_iteracion = pd.DataFrame()
	models_iterations = pd.DataFrame()
	
	for i in range(folders):
		#iteraciones
		archivo = ruta+"\\" + str(i) +'-Iteracion\\rb_{}_ordenamiento_feats_iteracion_'.format(sector.lower())+ str(i) +".xlsx"
		iter_p = pd.read_excel(archivo, sheet_name='Sheet1')
		iter_p['iteracion'] = (i)
		feats_iteracion = pd.concat([feats_iteracion, iter_p])
		
		#modelos
		archivo = ruta+"\\" + str(i) +'-Iteracion\\rb_{}_modelos_iteracion_'.format(sector.lower())+ str(i) +".xlsx"
		iter_p = pd.read_excel(archivo, sheet_name='Sheet1')
		iter_p['iteracion'] = (i)
		models_iterations = pd.concat([models_iterations, iter_p]) 
	#Fabrica de graficas
	
	#lineas por decision 
	feats_iteracion.groupby(['iteracion','decision'])['feats'].count().unstack().plot(kind='line', stacked=True,figsize=(17, 4))
	
	#boxplot
	date_mod_1=models_iterations[['iteracion','ar_tr']]
	date_mod_1.rename(columns={'ar_tr':'auc'}, inplace=True)
	date_mod_1['tipo_auc'] = 'AUC TRAIN'
	date_mod_2=models_iterations[['iteracion','ar_te']]
	date_mod_2.rename(columns={'ar_te':'auc'}, inplace=True)
	date_mod_2['tipo_auc'] = 'AUC TEST'
	date_mod_final = pd.concat([date_mod_1,date_mod_2])

	plt.figure(figsize=(17,6))
	ax = sns.boxplot(x="iteracion", y="auc", hue="tipo_auc",data=date_mod_final)
	
	# Matriz de variables
	
	cols = []
	for i in range(folders):
		cols.append((folders-1)-i)
	aux = feats_iteracion[['feats','decision','iteracion']]
	aux = aux.pivot(index='feats',columns='iteracion',values='decision').sort_values(by = cols,ascending=True)
	aux.to_excel(ruta + '\\rb_{}_mapa_feats_iteraciones_.xlsx'.format(sector.lower()),index=True)
	aux = aux.replace({"Aceptar":1,
					   "Aceptar_Aumenta":2,
					   "Aceptar_Disminuye":3,
					   "Indiferente":4,
					   "Revisar manual":5,
					   "No aceptar":6})
	cmap = sns.color_palette("Set2",6)
	plt.figure(figsize = (18,9))
	ax = sns.heatmap( aux, cmap=cmap, cbar_kws={'label': 'Decisión'}, linewidths=.1)
	colorbar = ax.collections[0].colorbar
	colorbar.set_ticks([1,2,3,4,5,6])
	colorbar.set_ticklabels(['Aceptar','Aceptar_Aumenta','Aceptar_Disminuye','Indiferente','Revisar manual','No aceptar'])
	plt.show()
	
	
	# doble eje
	import numpy as np 
	import matplotlib.pyplot as plte 

	fig, ax1 = plte.subplots() 

	color = 'tab:orange'
	ax1.set_ylabel('Ordena test y cantidad de variables', color = color) 
	#ax1 = fig.add_subplot(111)
	models_iterations.groupby(['iteracion'])['ordena_te'].quantile(.75).plot(kind='line',figsize=(17, 4),label='ordena_test',color = color, marker = 'o')
	p = pd.DataFrame(models_iterations.groupby(['iteracion'])[['iteracion','ordena_te']].quantile(.75))
	x = range(folders)
	y = p['ordena_te'].tolist()
	for i,j in zip(x,y):
		ax1.annotate(str(round(j)),xy=(i+.05,j))
		ax1.tick_params(axis ='y', labelcolor = color) 

	feats_iteracion.groupby(['iteracion']).count()['feats'].plot(kind='line',color = 'black', marker = 'o')
	p = pd.DataFrame(feats_iteracion.groupby(['iteracion']).count()['feats'])
	y = p['feats'].tolist()
	for i,j in zip(x,y):
		ax1.annotate(str(round(j)),xy=(i+.05,j))
		ax1.tick_params(axis ='y', labelcolor = color) 

	ax2 = ax1.twinx() 

	color = 'tab:cyan'
	ax2.set_ylabel('ar test', color = color) 
	#ax2 = fig.add_subplot(111)
	models_iterations.groupby(['iteracion'])['ar_te'].quantile(.75).plot(kind='line',figsize=(17, 4),label='ar_test',color = color, marker = 'o')
	plte.ylim([0.3, .9])
	p = pd.DataFrame(models_iterations.groupby(['iteracion'])[['iteracion','ar_te']].quantile(.75))
	x = range(folders)
	y = p['ar_te'].tolist()
	for i,j in zip(x,y):
		ax2.annotate(str(round(j,3)),xy=(i+.05,j))
		ax2.tick_params(axis ='y', labelcolor = color) 

	fig.suptitle('Relación de la disminución de variables con el AUC y el ordenamiento', fontweight ="bold") 

	plte.show()
 

########################################################################################################################
# Ajusta un polinomio de un grado dado al predict_proba del modelo dado
########################################################################################################################
def ajustePD(df_train, df_test, df_oot, model, feats_total, q, deg_pol):
	from sklearn.metrics import mean_squared_error
	resultados  = pd.DataFrame(columns = ['random_state','rmse'])

	prob_v = 'prob'
	# Train
	y_score_train  =  model.predict_proba(df_train[feats_total])
	y_score_train  =  [item[1] for item in y_score_train]
	df_train[prob_v] = y_score_train
	# Test
	y_score_test  =  model.predict_proba(df_test[feats_total])
	y_score_test  =  [item[1] for item in y_score_test] 
	df_test[prob_v] = y_score_test
	# OOT
	y_score_oot  =  model.predict_proba(df_oot[feats_total])
	y_score_oot  =  [item[1] for item in y_score_oot] 
	df_oot[prob_v] = y_score_oot

	df_train['grupo'], bins = pd.qcut(df_train[prob_v], q, retbins=True)
	df_test['grupo'] = pd.cut(df_test[prob_v], bins)
	df_oot['grupo'] = pd.cut(df_oot[prob_v], bins)

	df_tr = df_train.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	df_te = df_test.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	df_oo = df_oot.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	
	# Se ajusta el polinomio de grado deg_pol
	z = np.polyfit(df_tr[prob_v], df_tr['def_default_12m'], deg_pol)
	predict = np.poly1d(z)
	df_tr['pol'] = predict(df_tr[prob_v])
	df_te['pol'] = predict(df_te[prob_v])
	df_oo['pol'] = predict(df_oo[prob_v])

	# TDO vs PD
	rmse_1 = [ round(np.sqrt(mean_squared_error(df_tr.def_default_12m, df_tr.prob)), 4), round(np.sqrt(mean_squared_error(df_te.def_default_12m, df_te.prob)), 4), round(np.sqrt(mean_squared_error(df_oo.def_default_12m, df_oo.prob)), 4) ]
	# TDO vs Polinomio
	rmse_2 = [ round(np.sqrt(mean_squared_error(df_tr.def_default_12m, df_tr.pol)), 4), round(np.sqrt(mean_squared_error(df_te.def_default_12m, df_te.pol)), 4), round(np.sqrt(mean_squared_error(df_oo.def_default_12m, df_oo.pol)), 4) ]

	rmse = pd.DataFrame({"TDO_vs_Pred_Proba": rmse_1, "TDO_vs_PD_Poli": rmse_2}, index =["TRAIN", "TEST", "OOT"])

	fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))
	fig.suptitle('Mapeo PD a TDO')
	axes[0].set_title('Train');	axes[1].set_title('Test'); axes[2].set_title('OOT')

	sns.lineplot(ax=axes[0], x=df_tr['index'], y=df_tr[prob_v])
	sns.lineplot(ax=axes[0], x=df_tr['index'], y=df_tr['def_default_12m'])
	sns.lineplot(ax=axes[0], x=df_tr['index'], y=df_tr['pol'])
	axes[0].legend(['Probabilidad', 'TDO', 'Polinomio'])

	sns.lineplot(ax=axes[1], x=df_te['index'], y=df_te[prob_v])
	sns.lineplot(ax=axes[1], x=df_te['index'], y=df_te['def_default_12m'])
	sns.lineplot(ax=axes[1], x=df_te['index'], y=df_te['pol'])
	axes[1].legend(['Probabilidad', 'TDO', 'Polinomio'])

	sns.lineplot(ax=axes[2], x=df_oo['index'], y=df_oo[prob_v])
	sns.lineplot(ax=axes[2], x=df_oo['index'], y=df_oo['def_default_12m'])
	sns.lineplot(ax=axes[2], x=df_oo['index'], y=df_oo['pol'])
	axes[2].legend(['Probabilidad', 'TDO', 'Polinomio'])

	axes[0].grid( True )
	axes[1].grid( True )
	axes[2].grid( True )
	plt.show()

	print("\033[1m Tabla resumen RMSE:\n \033[0m")
	print(rmse)

	return predict,z, rmse


####
# Ajuste PD CV
####
def ajustePD_cv(df_train_test, df_oot, model, feats_total, q, deg_pol):
	from sklearn.metrics import mean_squared_error
	resultados  = pd.DataFrame(columns = ['random_state','rmse'])

	prob_v = 'prob'
	# Train
	y_score_train_test  =  model.predict_proba(df_train_test[feats_total])
	y_score_train_test  =  [item[1] for item in y_score_train_test]
	df_train_test[prob_v] = y_score_train_test

	# OOT
	y_score_oot  =  model.predict_proba(df_oot[feats_total])
	y_score_oot  =  [item[1] for item in y_score_oot] 
	df_oot[prob_v] = y_score_oot

	df_train_test['grupo'], bins = pd.qcut(df_train_test[prob_v], q, retbins=True)
	df_oot['grupo'] = pd.cut(df_oot[prob_v], bins)

	df_trte = df_train_test.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	df_oo = df_oot.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	
	# Se ajusta el polinomio de grado deg_pol
	z = np.polyfit(df_trte[prob_v], df_trte['def_default_12m'], deg_pol)
	predict = np.poly1d(z)
	df_trte['pol'] = predict(df_trte[prob_v])
	df_oo['pol'] = predict(df_oo[prob_v])

	# TDO vs PD
	rmse_1 = [ round(np.sqrt(mean_squared_error(df_trte.def_default_12m, df_trte.prob)), 4), round(np.sqrt(mean_squared_error(df_oo.def_default_12m, df_oo.prob)), 4) ]
	# TDO vs Polinomio
	rmse_2 = [ round(np.sqrt(mean_squared_error(df_trte.def_default_12m, df_trte.pol)), 4), round(np.sqrt(mean_squared_error(df_oo.def_default_12m, df_oo.pol)), 4) ]

	rmse = pd.DataFrame({"TDO_vs_Pred_Proba": rmse_1, "TDO_vs_PD_Poli": rmse_2}, index =["TRAIN_TEST", "OOT"])

	fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
	fig.suptitle('Mapeo PD a TDO')
	axes[0].set_title('Train_Test');axes[1].set_title('OOT')

	sns.lineplot(ax=axes[0], x=df_trte['index'], y=df_trte[prob_v])
	sns.lineplot(ax=axes[0], x=df_trte['index'], y=df_trte['def_default_12m'])
	sns.lineplot(ax=axes[0], x=df_trte['index'], y=df_trte['pol'])
	axes[0].legend(['Probabilidad', 'TDO', 'Polinomio'])


	sns.lineplot(ax=axes[1], x=df_oo['index'], y=df_oo[prob_v])
	sns.lineplot(ax=axes[1], x=df_oo['index'], y=df_oo['def_default_12m'])
	sns.lineplot(ax=axes[1], x=df_oo['index'], y=df_oo['pol'])
	axes[1].legend(['Probabilidad', 'TDO', 'Polinomio'])

	axes[0].grid( True )
	axes[1].grid( True )
	plt.show()

	print("\033[1m Tabla resumen RMSE:\n \033[0m")
	print(rmse)

	return predict,z, rmse


########################################################################################################################
#mpaz: Función ajuste de la PD por medio una función logit
########################################################################################################################
def ajusteLogit(df_train, df_test, df_oot, model, feats_total, q, deg_pol):
	from math import log, exp
	from sklearn.metrics import mean_squared_error
	resultados  = pd.DataFrame(columns = ['random_state','rmse'])

	prob_v = 'prob'
	# Train
	y_score_train  =  model.predict_proba(df_train[feats_total])
	y_score_train  =  [item[1] for item in y_score_train]
	df_train[prob_v] = y_score_train

	# Test
	y_score_test  =  model.predict_proba(df_test[feats_total])
	y_score_test  =  [item[1] for item in y_score_test] 
	df_test[prob_v] = y_score_test

	# OOT
	y_score_oot  =  model.predict_proba(df_oot[feats_total])
	y_score_oot  =  [item[1] for item in y_score_oot] 
	df_oot[prob_v] = y_score_oot


	df_train['grupo'], bins = pd.qcut(df_train[prob_v], q, retbins=True)
	df_test['grupo'] = pd.cut(df_test[prob_v], bins)
	df_oot['grupo'] = pd.cut(df_oot[prob_v], bins)

	df_tr = df_train.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	df_te = df_test.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	df_oo = df_oot.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()

	# Se calcula el logit
	df_tr['logit'] = df_tr['def_default_12m'].apply(lambda x: np.log((x + 1e-7)/(1 - x)))
	df_te['logit'] = df_te['def_default_12m'].apply(lambda x: np.log((x + 1e-7)/(1 - x)))
	df_oo['logit'] = df_oo['def_default_12m'].apply(lambda x: np.log((x + 1e-7)/(1 - x)))


	# Se ajusta el polinomio de grado deg_pol
	z = np.polyfit(df_tr[prob_v], df_tr['logit'], deg_pol)
	predict = np.poly1d(z)
	df_tr['pol'] = predict(df_tr[prob_v])
	df_te['pol'] = predict(df_te[prob_v])
	df_oo['pol'] = predict(df_oo[prob_v])

	#Se encuentra el inverso del logaritmo
	df_tr['logit_pred'] = df_tr['pol'].apply(lambda x: np.exp(x)/(1+np.exp(x)))
	df_te['logit_pred'] = df_te['pol'].apply(lambda x: np.exp(x)/(1+np.exp(x)))
	df_oo['logit_pred'] = df_oo['pol'].apply(lambda x: np.exp(x)/(1+np.exp(x)))

	# TDO vs PD
	rmse_1 = [ round(np.sqrt(mean_squared_error(df_tr.def_default_12m, df_tr.prob)), 4), round(np.sqrt(mean_squared_error(df_te.def_default_12m, df_te.prob)), 4), round(np.sqrt(mean_squared_error(df_oo.def_default_12m, df_oo.prob)), 4) ]
	# TDO vs logit
	rmse_2 = [ round(np.sqrt(mean_squared_error(df_tr.def_default_12m, df_tr.logit_pred)), 4), round(np.sqrt(mean_squared_error(df_te.def_default_12m, df_te.logit_pred)), 4), round(np.sqrt(mean_squared_error(df_oo.def_default_12m, df_oo.logit_pred)), 4) ]

	rmse = pd.DataFrame({"TDO_vs_Pred_Proba": rmse_1, "TDO_vs_PD_Logit": rmse_2}, index =["TRAIN", "TEST", "OOT"])

	fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))
	fig.suptitle('Mapeo PD a TDO')
	axes[0].set_title('Train');	axes[1].set_title('Test'); axes[2].set_title('OOT')

	sns.lineplot(ax=axes[0], x=df_tr['index'], y=df_tr[prob_v])
	sns.lineplot(ax=axes[0], x=df_tr['index'], y=df_tr['def_default_12m'])
	sns.lineplot(ax=axes[0], x=df_tr['index'], y=df_tr['logit_pred'])
	axes[0].legend(['Probabilidad', 'TDO', 'Logit'])

	sns.lineplot(ax=axes[1], x=df_te['index'], y=df_te[prob_v])
	sns.lineplot(ax=axes[1], x=df_te['index'], y=df_te['def_default_12m'])
	sns.lineplot(ax=axes[1], x=df_te['index'], y=df_te['logit_pred'])
	axes[1].legend(['Probabilidad', 'TDO', 'Logit'])

	sns.lineplot(ax=axes[2], x=df_oo['index'], y=df_oo[prob_v])
	sns.lineplot(ax=axes[2], x=df_oo['index'], y=df_oo['def_default_12m'])
	sns.lineplot(ax=axes[2], x=df_oo['index'], y=df_oo['logit_pred'])
	axes[2].legend(['Probabilidad', 'TDO', 'Logit'])

	axes[0].grid( True )
	axes[1].grid( True )
	axes[2].grid( True )
	plt.show()

	print("\033[1m Tabla resumen RMSE:\n \033[0m")
	print(rmse)

	return predict,z, rmse



def ajusteLogitCV(df_train_test, df_oot, model, feats_total, q, deg_pol):
	from sklearn.metrics import mean_squared_error
	resultados  = pd.DataFrame(columns = ['random_state','rmse'])

	prob_v = 'prob'
	# Train
	y_score_train_test  =  model.predict_proba(df_train_test[feats_total])
	y_score_train_test  =  [item[1] for item in y_score_train_test]
	df_train_test[prob_v] = y_score_train_test

	# OOT
	y_score_oot  =  model.predict_proba(df_oot[feats_total])
	y_score_oot  =  [item[1] for item in y_score_oot] 
	df_oot[prob_v] = y_score_oot


	df_train_test['grupo'], bins = pd.qcut(df_train_test[prob_v], q, retbins=True)
	df_oot['grupo'] = pd.cut(df_oot[prob_v], bins)

	df_trte = df_train_test.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	df_oo = df_oot.groupby('grupo').agg({'num_doc':'count', 'def_default_12m':'mean', prob_v:'mean'}).reset_index().reset_index()
	
	# Se calcula el logit
	df_trte['logit'] = df_trte['def_default_12m'].apply(lambda x: np.log((x + 1e-7)/(1 - x)))
	df_oo['logit'] = df_oo['def_default_12m'].apply(lambda x: np.log((x + 1e-7)/(1 - x)))


	# Se ajusta el polinomio de grado deg_pol
	z = np.polyfit(df_trte[prob_v], df_trte['logit'], deg_pol)
	predict = np.poly1d(z)
	df_trte['pol'] = predict(df_trte[prob_v])
	df_oo['pol'] = predict(df_oo[prob_v])

	#Se encuentra el inverso del logaritmo
	df_trte['logit_pred'] = df_trte['pol'].apply(lambda x: np.exp(x)/(1+np.exp(x)))
	df_oo['logit_pred'] = df_oo['pol'].apply(lambda x: np.exp(x)/(1+np.exp(x)))

	# TDO vs PD
	rmse_1 = [ round(np.sqrt(mean_squared_error(df_trte.def_default_12m, df_trte.prob)), 4), round(np.sqrt(mean_squared_error(df_oo.def_default_12m, df_oo.prob)), 4) ]
	# TDO vs logit
	rmse_2 = [ round(np.sqrt(mean_squared_error(df_trte.def_default_12m, df_trte.logit_pred)), 4), round(np.sqrt(mean_squared_error(df_oo.def_default_12m, df_oo.logit_pred)), 4) ]

	rmse = pd.DataFrame({"TDO_vs_Pred_Proba": rmse_1, "TDO_vs_PD_Logit": rmse_2}, index =["TRAIN_TEST", "OOT"])

	fig, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
	fig.suptitle('Mapeo PD a TDO')
	axes[0].set_title('Train');	axes[1].set_title('OOT')

	sns.lineplot(ax=axes[0], x=df_trte['index'], y=df_trte[prob_v])
	sns.lineplot(ax=axes[0], x=df_trte['index'], y=df_trte['def_default_12m'])
	sns.lineplot(ax=axes[0], x=df_trte['index'], y=df_trte['logit_pred'])
	axes[0].legend(['Probabilidad', 'TDO', 'Logit'])

	sns.lineplot(ax=axes[1], x=df_oo['index'], y=df_oo[prob_v])
	sns.lineplot(ax=axes[1], x=df_oo['index'], y=df_oo['def_default_12m'])
	sns.lineplot(ax=axes[1], x=df_oo['index'], y=df_oo['logit_pred'])
	axes[1].legend(['Probabilidad', 'TDO', 'Logit'])

	axes[0].grid( True )
	axes[1].grid( True )
	plt.show()

	print("\033[1m Tabla resumen RMSE:\n \033[0m")
	print(rmse)

	return predict,z, rmse


########################################################################################################################
# Adaptaciones de funciones para el modelo multisectorial
########################################################################################################################
# Esta función realiza el entrenamiento de n_iter x 2 (lightgbm y HistGradientBoosting). Retorna tres métricas:
# (1) número de escalas ordenadas
# (2) número de escalas en rango
# (3) auc
# Estas metricas se calculan para cada muestra: entrenamiento, prueba y fuera de tiempo, ademas de calcular por medio de una 
# indicadora metricas para subpoblaciones en este caso sectores
#
# funcion de entrenamiento
# Autor: morodrig.
# funciones metricas q
# Autor: wilson munera
#
# Actualización: mialopez
# Se retorna de la función los valores de bins para poder ser utilizados en otras aplicaciones
########################################################################################################################
def metricas_ordena_q_sector( base_train_pre_, base_test_pre_, base_oot_pre_, q__, nom_proba, var_default, var_conteo ):

	## Train
	base_train_pre_['q'] = pd.qcut(base_train_pre_[nom_proba], q__, labels = False)
	base_train_pre_['q_range'], bins = pd.qcut(base_train_pre_[nom_proba], q__, retbins=True )
	cat = base_train_pre_[['q','q_range']].drop_duplicates()

	q_train = base_train_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_train.columns = ['q','cantidad','default']
	q_train['tdo'] = round( q_train['default'] / q_train['cantidad']*100,2)
	q_train['prct'] = round( q_train['cantidad'] / sum(q_train['cantidad'])*100,2)

	q_train.sort_values(by='q', inplace=True)
	max_val_train = q_train['tdo'].expanding(1).max()
	ini_val_train = pd.Series(-1)
	q_train['before_max_tdo'] = ini_val_train.append(max_val_train).reset_index(drop=True).drop(labels=len(q_train))
	q_train['ordena'] = (q_train['tdo'] > q_train['before_max_tdo']).astype(int)
	q_train = pd.merge(q_train, cat, how = 'left', on = 'q')
	ordena_train_q = sum(q_train['ordena'])

	## Test
	base_test_pre_['q'] = pd.cut(base_test_pre_[nom_proba], bins, labels = False)
	base_test_pre_['q_range'] = pd.cut(base_test_pre_[nom_proba], bins)

	q_test = base_test_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_test.columns = ['q','cantidad','default']
	q_test['tdo'] = round( q_test['default'] / q_test['cantidad']*100,2)
	q_test['prct'] = round( q_test['cantidad'] / sum(q_test['cantidad'])*100,2)

	q_test.sort_values(by='q', inplace=True)
	max_val_test = q_test['tdo'].expanding(1).max()
	ini_val_test = pd.Series(-1)
	q_test['before_max_tdo'] = ini_val_test.append(max_val_test).reset_index(drop=True).drop(labels=len(q_test))
	q_test['ordena'] = (q_test['tdo'] > q_test['before_max_tdo']).astype(int)
	q_test = pd.merge(q_test, cat, how = 'left', on = 'q')
	ordena_test_q = sum(q_test['ordena'])

	## OOT
	base_oot_pre_['q'] = pd.cut(base_oot_pre_[nom_proba], bins, labels = False)
	base_oot_pre_['q_range'] = pd.cut(base_oot_pre_[nom_proba], bins)

	q_oot = base_oot_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_oot.columns = ['q','cantidad','default']
	q_oot['tdo'] = round( q_oot['default'] / q_oot['cantidad']*100,2)
	q_oot['prct'] = round( q_oot['cantidad'] / sum(q_oot['cantidad'])*100,2)

	q_oot.sort_values(by='q', inplace=True)
	max_val_oot = q_oot['tdo'].expanding(1).max()
	ini_val_oot = pd.Series(-1)
	q_oot['before_max_tdo'] = ini_val_oot.append(max_val_oot).reset_index(drop=True).drop(labels=len(q_oot))
	q_oot['ordena'] = (q_oot['tdo'] > q_oot['before_max_tdo']).astype(int)
	q_oot = pd.merge(q_oot, cat, how = 'left', on = 'q')
	ordena_oot_q = sum(q_oot['ordena'])

	return q_train, q_test, q_oot, ordena_train_q, ordena_test_q, ordena_oot_q,bins

########################################################################################################################
# Adaptaciones de funciones para el modelo multisectorial
########################################################################################################################
# Esta función realiza el entrenamiento de n_iter x 2 (lightgbm y HistGradientBoosting). Retorna tres métricas:
# (1) número de escalas ordenadas
# (2) número de escalas en rango
# (3) auc
# Estas metricas se calculan para cada muestra: entrenamiento, prueba y fuera de tiempo, ademas de calcular por medio de una 
# indicadora metricas para subpoblaciones en este caso sectores
#
# funcion de entrenamiento
# Autor: morodrig.
# funciones metricas q
# Autor: wilson munera
#
# Actualización: mialopez
# Recibe los valores de bin para aplicarlos en poblaciónes determinadas.
########################################################################################################################
def metricas_ordena_sq_sector( base_train_complete ,base_train_pre_, base_test_pre_, base_oot_pre_, bins, nom_proba, var_default, var_conteo ):

	## Train
	base_train_pre_['q'] = pd.cut(base_train_pre_[nom_proba], bins, labels = False)
	base_train_pre_['q_range'] = pd.cut(base_train_pre_[nom_proba], bins)
	cat = base_train_complete[['q','q_range']].drop_duplicates()

	q_train = base_train_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_train.columns = ['q','cantidad','default']
	q_train['tdo'] = round( q_train['default'] / q_train['cantidad']*100,2)
	q_train['prct'] = round( q_train['cantidad'] / sum(q_train['cantidad'])*100,2)

	q_train.sort_values(by='q', inplace=True)
	max_val_train = q_train['tdo'].expanding(1).max()
	ini_val_train = pd.Series(-1)
	q_train['before_max_tdo'] = ini_val_train.append(max_val_train).reset_index(drop=True).drop(labels=len(q_train))
	q_train['ordena'] = (q_train['tdo'] > q_train['before_max_tdo']).astype(int)
	q_train = pd.merge(q_train, cat, how = 'left', on = 'q')
	ordena_train_q = sum(q_train['ordena'])

	## Test
	base_test_pre_['q'] = pd.cut(base_test_pre_[nom_proba], bins, labels = False)
	base_test_pre_['q_range'] = pd.cut(base_test_pre_[nom_proba], bins)

	q_test = base_test_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_test.columns = ['q','cantidad','default']
	q_test['tdo'] = round( q_test['default'] / q_test['cantidad']*100,2)
	q_test['prct'] = round( q_test['cantidad'] / sum(q_test['cantidad'])*100,2)

	q_test.sort_values(by='q', inplace=True)
	max_val_test = q_test['tdo'].expanding(1).max()
	ini_val_test = pd.Series(-1)
	q_test['before_max_tdo'] = ini_val_test.append(max_val_test).reset_index(drop=True).drop(labels=len(q_test))
	q_test['ordena'] = (q_test['tdo'] > q_test['before_max_tdo']).astype(int)
	q_test = pd.merge(q_test, cat, how = 'left', on = 'q')
	ordena_test_q = sum(q_test['ordena'])

	## OOT
	base_oot_pre_['q'] = pd.cut(base_oot_pre_[nom_proba], bins, labels = False)
	base_oot_pre_['q_range'] = pd.cut(base_oot_pre_[nom_proba], bins)

	q_oot = base_oot_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_oot.columns = ['q','cantidad','default']
	q_oot['tdo'] = round( q_oot['default'] / q_oot['cantidad']*100,2)
	q_oot['prct'] = round( q_oot['cantidad'] / sum(q_oot['cantidad'])*100,2)

	q_oot.sort_values(by='q', inplace=True)
	max_val_oot = q_oot['tdo'].expanding(1).max()
	ini_val_oot = pd.Series(-1)
	q_oot['before_max_tdo'] = ini_val_oot.append(max_val_oot).reset_index(drop=True).drop(labels=len(q_oot))
	q_oot['ordena'] = (q_oot['tdo'] > q_oot['before_max_tdo']).astype(int)
	q_oot = pd.merge(q_oot, cat, how = 'left', on = 'q')
	ordena_oot_q = sum(q_oot['ordena'])

	return q_train, q_test, q_oot, ordena_train_q, ordena_test_q, ordena_oot_q

def train_nueva_era_segint_sector(base_train_pre_ ,base_test_pre_, base_oot_pre_, target, n_iter, seed, feats_train, parametros, muestras_en_intervalo, seg_interes = '' , escala_c = True, q = 20):
	#########################################################
	# Se fija la semilla
	random.seed(seed)
	#########################################################
	# Preparación Datos: Totales
	#########################################################
	base_train_pre, X_train, y_train, base_test_pre, X_test, y_test, base_oot_pre, X_oot, y_oot = prepara_bases(base_train_pre_, base_test_pre_, base_oot_pre_, target, feats_train)

	#########################################################
	# Hiperparámetros Algoritmos
	#########################################################
	learning_ra = parametros['learning_ra']
	estimadores = parametros['estimadores']
	profundidad = parametros['profundidad']
	min_data_le = parametros['min_data_le']
	max_bins___ = parametros['max_bins___']
	n_leaves___ = parametros['n_leaves___']

	params_lgbm = {
		'learning_rate'    : learning_ra,
		'max_bin'          : max_bins___,
		'n_estimators'     : estimadores,
		'num_leaves'       : n_leaves___,
		'max_depth'        : profundidad,
		'min_data_in_leaf' : min_data_le,
	}

	param_rf = {
		'n_estimators'     : estimadores,
		'max_features'     : ['auto', 'sqrt', 'log2'],
		'max_depth'        : profundidad,
		'criterion'        : ['gini', 'entropy'],
		'min_samples_leaf' : min_data_le
	}

	param_hgb = {
			'loss'               : ['auto'],
			'learning_rate'      : learning_ra,
			'max_iter'           : estimadores,
			'max_leaf_nodes'     : n_leaves___,
			'max_depth'          : profundidad,
			'min_samples_leaf'   : min_data_le,
			'l2_regularization'  : [float(x) for x in np.linspace(start = 0, stop = 0.5, num = muestras_en_intervalo)],
			'max_bins'           : max_bins___,
			'random_state'       : [seed],
			'tol'                : [1e-7]
			}

	dict_hpparams = {'lgbm':params_lgbm, 'hgb': param_hgb}

	#########################################################
	# Resultados a medir: totales
	#########################################################
	semilla = []; paramters = [];
	auc_train_l = [];     auc_test_l = [];     auc_oot_l = [];
	enrango_train_l = []; enrango_test_l = []; enrango_oot_l = [];
	ordena_train_l = [];  ordena_test_l = [];  ordena_oot_l = [];
	ordena_train_q_l = [];  ordena_test_q_l = [];  ordena_oot_q_l = []; 
	prct_desorden_train_l = []; prct_desorden_test_l = []; prct_desorden_oot_l = [];
	typemodel_l = []; sector = []; modelo_num = []

	# Se define punto de corte
	cuto = 0.0859
	nom_proba = 'proba'

	#########################################################
	# Iteraciones
	#########################################################
	for i in range(n_iter):
		if i % 10 == 0:
			print('******************************************************************** ')
			print('******************************************************************** ')
			print('Iteración: ' + str(i))
			print('******************************************************************** ')
			print('******************************************************************** ')
		for key in dict_hpparams:
			try:
				if key == 'lgbm':
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)

					model1 = lgb.LGBMClassifier(
					learning_rate = hyperparameters['learning_rate'],
					max_bin = hyperparameters['max_bin'],
					n_estimators = hyperparameters['n_estimators'],
					num_leaves = hyperparameters['num_leaves'],
					max_depth = hyperparameters['max_depth'],
					min_data_in_leaf = hyperparameters['min_data_in_leaf'],
					n_jobs = -3,
					#silent = True,
					verbose=-1,
					seed = seed
					)

					typemodel = 'Lighgbm'

				elif  key == 'param_rf':
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)

					model1 = RandomForestClassifier(
					n_estimators = hyperparameters['n_estimators'],
					max_features = hyperparameters['max_features'],
					criterion = hyperparameters['criterion'],
					min_samples_split = hyperparameters['min_samples_split'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					n_jobs = -3,
					)

					typemodel = 'RandomForest'

				else : ##'hgb
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)

					model1 = HistGradientBoostingClassifier(
					loss = hyperparameters['loss'],
					learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],
					max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],
					max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],
					#validation_fraction = hyperparameters['validation_fraction'],
					#n_jobs = -3,
					random_state = seed
					)

					typemodel = 'HistGradientBoosting'

				#########################################################
				# Entrenamiento
				#########################################################
				model1.fit(X_train, np.ravel(y_train))#, verbose=False)
				#print("modelo ok")

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
				_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango( base_train_pre, nom_proba, target[0], 'f_analisis' )
				_, enrango_test, ordena_test, prct_desorden_test   = metricas_ordena_rango( base_test_pre,  nom_proba, target[0], 'f_analisis' )
				_, enrango_oot, ordena_oot, prct_desorden_oot     = metricas_ordena_rango( base_oot_pre,   nom_proba, target[0], 'f_analisis' )

				_,_,_, ordena_train_q, ordena_test_q, ordena_oot_q, bins = metricas_ordena_q_sector(base_train_pre, base_test_pre, base_oot_pre, q, nom_proba, target[0], 'f_analisis')
				auc_train = metrics.roc_auc_score( y_train, y_score_train )
				auc_test  = metrics.roc_auc_score( y_test,  y_score_test )
				auc_oot   = metrics.roc_auc_score( y_oot,   y_score_oot )

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

				typemodel_l.append(typemodel)
				paramters.append(hyperparameters)

				# sector 
				sector.append("Generico") 
				modelo_num.append(i)
				sectores = ['Agropecuario','Comercio','Edificaciones','Infraestructura','Manufactura','Servicios','RR_Naturales']
				for s in sectores: 
					# Separación de la población por sector
					base_train_pre_sec=base_train_pre[base_train_pre['sector'] == s]
					base_test_pre_sec=base_test_pre[base_test_pre['sector'] == s]
					base_oot_pre_sec=base_oot_pre[base_oot_pre['sector'] == s]
					
					base_train_pre_s, X_train_s, y_train_s, base_test_pre_s, X_test_s, y_test_s, base_oot_pre_s, X_oot_s, y_oot_s = prepara_bases(base_train_pre_sec, base_test_pre_sec, base_oot_pre_sec, target, feats_train)

					# Cifras
					_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango( base_train_pre_sec, nom_proba, target[0], 'f_analisis' )
					_, enrango_test, ordena_test, prct_desorden_test   = metricas_ordena_rango( base_test_pre_sec,  nom_proba, target[0], 'f_analisis' )
					_, enrango_oot, ordena_oot, prct_desorden_oot     = metricas_ordena_rango( base_oot_pre_sec,   nom_proba, target[0], 'f_analisis' )

					_,_,_, ordena_train_q, ordena_test_q, ordena_oot_q = metricas_ordena_sq_sector(base_train_pre,base_train_pre_sec, base_test_pre_sec, base_oot_pre_sec, bins, nom_proba, target[0], 'f_analisis')

					auc_train = metrics.roc_auc_score( y_train_s, base_train_pre_sec[nom_proba] )
					auc_test  = metrics.roc_auc_score( y_test_s,  base_test_pre_sec[nom_proba] )
					auc_oot   = metrics.roc_auc_score( y_oot_s,   base_oot_pre_sec[nom_proba] )

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

					typemodel_l.append(typemodel)
					paramters.append(hyperparameters)
					sector.append(s) 
					modelo_num.append(i)

			except:
				pass
	
	resultados = {'parameters': paramters,
					  'auc_tr': auc_train_l, 'auc_te': auc_test_l, 'auc_o': auc_oot_l,
					  'enrango_tr': enrango_train_l, 'enrango_te': enrango_test_l, 'enrango_o': enrango_oot_l,
					  'ordena_tr': ordena_train_l, 'ordena_te': ordena_test_l, 'ordena_o': ordena_oot_l,
					  'prct_desorden_tr': prct_desorden_train_l, 'prct_desorden_te': prct_desorden_test_l, 'prct_desorden_o': prct_desorden_oot_l,
					  'ordena_tr_q': ordena_train_q_l, 'ordena_te_q': ordena_test_q_l, 'ordena_o_q': ordena_oot_q_l,
					  'modelo' : typemodel_l, 'sector' : sector, 'modelo_num': modelo_num
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
					'ordena_q','ordena_tr_q', 'ordena_te_q', 'ordena_o_q','sector','modelo_num' ]
	else:
		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif',
					'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o','sector']

	return res_df[columnas]


def mapeo_escala_sector(sectores, modelo, df_train, df_test, df_oot, features, q_ = 20):
	for s in sectores:
		if s == 'Generico':
			y_score_train  =  modelo.predict_proba(df_train[features])
			y_score_test   =  modelo.predict_proba(df_test[features])
			y_score_oot    =  modelo.predict_proba(df_oot[features])

			y_score_train  =  [item[1] for item in y_score_train]
			y_score_test   =  [item[1] for item in y_score_test]
			y_score_oot    =  [item[1] for item in y_score_oot]

			nom_proba = 'proba'
			df_train[nom_proba] = y_score_train
			df_test[nom_proba]  = y_score_test
			df_oot[nom_proba]   = y_score_oot

			mtr, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango( df_train, nom_proba, 'def_default_12m', 'f_analisis' )
			mte, enrango_test_ee, ordena_test_ee,_   = metricas_ordena_rango( df_test,  nom_proba, 'def_default_12m', 'f_analisis' )
			mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango( df_oot,   nom_proba, 'def_default_12m', 'f_analisis' )

			mtr['rango'] = mtr['c'] == mtr['c_obs']
			mte['rango'] = mte['c'] == mte['c_obs']
			mo['rango']  = mo[ 'c'] == mo[ 'c_obs']

			mtr, mte, mo, ordena_train_ee, ordena_test_ee, ordena_oot_ee, bins = metricas_ordena_q_sector( df_train, df_test, df_oot, q_, nom_proba, 'def_default_12m', 'f_analisis' )

			cols_1 = ['q', 'q_range']
			cols_n = ['tdo','prct','ordena','cantidad']

			mtr = mtr[cols_1+cols_n]
			mte = mte[cols_1+cols_n]
			mo  = mo[ cols_1+cols_n]

			mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
			mte.columns = cols_1 + [x+'_te' for x in cols_n]
			mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

			df_consolidado = mtr.merge(mte, how='outer', on = ['q','q_range']).merge(mo, how='outer', on = ['q','q_range'])
			mapeo_q = df_consolidado
			mapeo_q['sector'] = s

		else:
			base_train_pre_sec=df_train[df_train['sector'] == s]
			base_test_pre_sec=df_test[df_test['sector'] == s]
			base_oot_pre_sec=df_test[df_test['sector'] == s]

			y_score_train  =  modelo.predict_proba(df_train[features])
			y_score_test   =  modelo.predict_proba(df_test[features])
			y_score_oot    =  modelo.predict_proba(df_oot[features])

			y_score_train  =  [item[1] for item in y_score_train]
			y_score_test   =  [item[1] for item in y_score_test]
			y_score_oot    =  [item[1] for item in y_score_oot]

			nom_proba = 'proba'
			df_train[nom_proba] = y_score_train
			df_test[nom_proba]  = y_score_test
			df_oot[nom_proba]   = y_score_oot

			mtr, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango( df_train, nom_proba, 'def_default_12m', 'f_analisis' )
			mte, enrango_test_ee, ordena_test_ee,_   = metricas_ordena_rango( df_test,  nom_proba, 'def_default_12m', 'f_analisis' )
			mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango( df_oot,   nom_proba, 'def_default_12m', 'f_analisis' )

			mtr['rango'] = mtr['c'] == mtr['c_obs']
			mte['rango'] = mte['c'] == mte['c_obs']
			mo['rango']  = mo[ 'c'] == mo[ 'c_obs']

			cols_1 = ['c']
			cols_n = ['tdo','prct','rango','ordena','cantidad']

			mtr = mtr[cols_1+cols_n]
			mte = mte[cols_1+cols_n]
			mo  = mo[ cols_1+cols_n]

			mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
			mte.columns = cols_1 + [x+'_te' for x in cols_n]
			mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

			mtr, mte, mo, ordena_train_ee, ordena_test_ee, ordena_oot_ee = metricas_ordena_sq_sector( base_train_pre_sec,base_train_pre_sec, base_test_pre_sec, base_oot_pre_sec, bins, nom_proba, 'def_default_12m', 'f_analisis' )

			cols_1 = ['q', 'q_range']
			cols_n = ['tdo','prct','ordena','cantidad']

			mtr = mtr[cols_1+cols_n]
			mte = mte[cols_1+cols_n]
			mo  = mo[ cols_1+cols_n]

			mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
			mte.columns = cols_1 + [x+'_te' for x in cols_n]
			mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

			df_consolidado = mtr.merge(mte, how='outer', on = ['q','q_range']).merge(mo, how='outer', on = ['q','q_range'])
			df_consolidado['sector'] = s
			mapeo_q = pd.concat([mapeo_q, df_consolidado])
	return mapeo_q

def train_nueva_era_segint_w_cv_sector(base_final, target, n_iter, seed, feats_train, parametros, muestras_en_intervalo, n_folds = 5, escala_c = True, q = 20):
	from sklearn.model_selection import KFold
	random.seed(seed)
	target = ['def_default_12m']
	list_ID = base_final.loc[base_final.train_test_e1.isin([0,1])].llave_nombre.unique()

	#########################################################
	# Hiperparámetros Algoritmos
	#########################################################
	learning_ra = parametros['learning_ra']
	estimadores = parametros['estimadores']
	profundidad = parametros['profundidad']
	min_data_le = parametros['min_data_le']
	max_bins___ = parametros['max_bins___']
	n_leaves___ = parametros['n_leaves___']

	params_lgbm = {
		'learning_rate'    : learning_ra,
		'max_bin'          : max_bins___,
		'n_estimators'     : estimadores,
		'num_leaves'       : n_leaves___,
		'max_depth'        : profundidad,
		'min_data_in_leaf' : min_data_le
	}
	param_hgb = {
			'loss'               : ['auto'],
			'learning_rate'      : learning_ra,
			'max_iter'           : estimadores,
			'max_leaf_nodes'     : n_leaves___,
			'max_depth'          : profundidad,
			'min_samples_leaf'   : min_data_le,
			'l2_regularization'  : [float(x) for x in np.linspace(start = 0, stop = 0.5, num = muestras_en_intervalo)],
			'max_bins'           : max_bins___,
			'random_state'       : [seed],
			'tol'                : [1e-7]
			}
	dict_hpparams = {'lgbm':params_lgbm, 'hgb': param_hgb}
	#########################################################
	# Resultados a medir: totales
	#########################################################
	semilla = []; paramters = [];
	auc_train_l = [];     auc_test_l = [];     auc_oot_l = [];
	enrango_train_l = []; enrango_test_l = []; enrango_oot_l = [];
	ordena_train_l = [];  ordena_test_l = [];  ordena_oot_l = [];
	ordena_train_q_l = [];  ordena_test_q_l = [];  ordena_oot_q_l = []; 
	prct_desorden_train_l = []; prct_desorden_test_l = []; prct_desorden_oot_l = [];
	typemodel_l = []; k_fold_l = []; sector = []; modelo_num = []

	nom_proba = 'proba'
	#########################################################
	# Iteraciones
	#########################################################
	for i in range(n_iter):
		if i % 10 == 0:
			print('******************************************************************** ')
			print('******************************************************************** ')
			print('Iteración: ' + str(i))
			print('******************************************************************** ')
			print('******************************************************************** ')
		for key in dict_hpparams:
			try: 
				if key == 'lgbm':
					semilla.append(seed)
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)
					model1 = lgb.LGBMClassifier(
					learning_rate = hyperparameters['learning_rate'],
					max_bin = hyperparameters['max_bin'],
					n_estimators = hyperparameters['n_estimators'],
					num_leaves = hyperparameters['num_leaves'],
					max_depth = hyperparameters['max_depth'],
					min_data_in_leaf = hyperparameters['min_data_in_leaf'],
					n_jobs = -3,
					#silent = True,
					verbose=-1,
					seed = seed
					)
					typemodel = 'Lighgbm'
				else :
					semilla.append(seed)
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)
					model1 = HistGradientBoostingClassifier(
					loss = hyperparameters['loss'],
					learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],
					max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],
					max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],
					#validation_fraction = hyperparameters['validation_fraction'],
					#n_jobs = -3,
					random_state = seed
					)
					typemodel = 'HistGradientBoosting'
				
				#########################################################
				# Cross Validation
				#########################################################
				kf = KFold(n_splits = n_folds)
				kf.get_n_splits(list_ID)
				auc_train_cv = []; auc_test_cv = []; auc_oot_cv = [];
				enrango_train_cv = []; enrango_test_cv = []; enrango_oot_cv = [];
				ordena_train_cv = []; ordena_test_cv = []; ordena_oot_cv = [];
				ordena_train_cv_q = []; ordena_test_cv_q = []; ordena_oot_cv_q = [];
				prct_desorden_train_cv = []; prct_desorden_test_cv = []; prct_desorden_oot_cv = [];

				for ID_train, ID_test in kf.split(list_ID):
					base_train_pre = base_final.loc[base_final.llave_nombre.isin(list_ID[ID_train])]
					base_test_pre = base_final.loc[base_final.llave_nombre.isin(list_ID[ID_test])]
					base_oot_pre   = base_final.query('train_test_e2 == 2').copy()                

					X_train = base_train_pre[feats_train]
					y_train = base_train_pre[target]

					X_test = base_test_pre[feats_train]
					y_test = base_test_pre[target]

					X_oot = base_oot_pre[feats_train]
					y_oot = base_oot_pre[target]
					print("Sector: Generico")
					print("Base Train %s Base Test %s Base OOT %s" % (base_train_pre.shape, base_test_pre.shape, base_oot_pre.shape))

					#########################################################
					# Entrenamiento 
					#########################################################
					model1.fit(X_train, np.ravel(y_train))

					#########################################################
					########### REPORTE Parciales de cada modelo ############
					#########################################################
					y_score_train  =  model1.predict_proba(X_train)
					y_score_test   =  model1.predict_proba(X_test)
					y_score_oot    =  model1.predict_proba(X_oot)
					y_score_train  =  [item[1] for item in y_score_train]
					y_score_test   =  [item[1] for item in y_score_test]
					y_score_oot    =  [item[1] for item in y_score_oot]

					base_train_pre[nom_proba] = y_score_train
					base_test_pre[nom_proba] = y_score_test
					base_oot_pre[nom_proba]  = y_score_oot

					#########################################################
					# Métricas precisión, mapeo y concentraciones: Totales
					#########################################################
					_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango_estricta(base_train_pre, nom_proba, target[0], 'f_analisis')
					_, enrango_test, ordena_test, prct_desorden_test = metricas_ordena_rango_estricta(base_test_pre, nom_proba, target[0], 'f_analisis')
					_, enrango_oot, ordena_oot, prct_desorden_oot = metricas_ordena_rango_estricta(base_oot_pre, nom_proba, target[0], 'f_analisis')

					_,_,_, ordena_train_q, ordena_test_q, ordena_oot_q, bins = metricas_ordena_q_sector(base_train_pre, base_test_pre, base_oot_pre, q, nom_proba, target[0], 'f_analisis')

					auc_train = metrics.roc_auc_score( y_train, y_score_train )
					auc_test  = metrics.roc_auc_score( y_test,  y_score_test )
					auc_oot   = metrics.roc_auc_score( y_oot,   y_score_oot )

					# AUC
					auc_train_cv.append(auc_train)
					auc_test_cv.append(auc_test)
					auc_oot_cv.append(auc_oot)
					# Escalas en rango
					enrango_train_cv.append(enrango_train)
					enrango_test_cv.append(enrango_test)
					enrango_oot_cv.append(enrango_oot)
					# Escalas ordenadas
					ordena_train_cv.append(ordena_train)
					ordena_test_cv.append(ordena_test)
					ordena_oot_cv.append(ordena_oot)
					# Escalas ordenadas q
					ordena_train_cv_q.append(ordena_train_q)
					ordena_test_cv_q.append(ordena_test_q)
					ordena_oot_cv_q.append(ordena_oot_q)
					# Porcentaje de desorden C
					prct_desorden_train_cv.append(prct_desorden_train)
					prct_desorden_test_cv.append( prct_desorden_test)
					prct_desorden_oot_cv.append(  prct_desorden_oot)


				# Promedio AUC
				auc_train_l.append(np.mean(auc_train_cv))
				auc_test_l.append(np.mean(auc_test_cv))
				auc_oot_l.append(np.mean(auc_oot_cv))

				# Promedio Escalas en rango
				enrango_train_l.append(np.mean(enrango_train_cv))
				enrango_test_l.append(np.mean(enrango_test_cv))
				enrango_oot_l.append(np.mean(enrango_oot_cv))

				# Promedio Escalas ordenadas
				ordena_train_l.append(np.mean(ordena_train_cv))
				ordena_test_l.append( np.mean(ordena_test_cv))
				ordena_oot_l.append(  np.mean(ordena_oot_cv))

				# Promedio Escalas ordenadas q
				ordena_train_q_l.append(np.mean(ordena_train_cv_q))
				ordena_test_q_l.append(np.mean(ordena_test_cv_q))
				ordena_oot_q_l.append(np.mean(ordena_oot_cv_q))

				# Promedio Porcentaje de desorden C
				prct_desorden_train_l.append(np.mean(prct_desorden_train_cv))
				prct_desorden_test_l.append(np.mean(prct_desorden_test_cv))
				prct_desorden_oot_l.append(np.mean(prct_desorden_oot_cv))

				typemodel_l.append(typemodel)
				paramters.append(hyperparameters)
				# sector 
				sector.append("Generico") 
				modelo_num.append(i)


				sectores = ['Agropecuario','Comercio','Edificaciones','Infraestructura','Manufactura','Servicios','RR_Naturales']
				for s in sectores:

					kf.get_n_splits(list_ID)
					auc_train_cv = []; auc_test_cv = []; auc_oot_cv = [];
					enrango_train_cv = []; enrango_test_cv = []; enrango_oot_cv = [];
					ordena_train_cv = []; ordena_test_cv = []; ordena_oot_cv = [];
					ordena_train_cv_q = []; ordena_test_cv_q = []; ordena_oot_cv_q = [];
					prct_desorden_train_cv = []; prct_desorden_test_cv = []; prct_desorden_oot_cv = [];
					
					base_final_sec = base_final[base_final['sector'] == s]

					for ID_train, ID_test in kf.split(list_ID):
						base_train_pre_sec = base_final_sec.loc[base_final_sec.llave_nombre.isin(list_ID[ID_train])]
						base_test_pre_sec = base_final_sec.loc[base_final_sec.llave_nombre.isin(list_ID[ID_test])]
						base_oot_pre_sec   = base_final_sec.query('train_test_e2 == 2').copy()                

						X_train = base_train_pre_sec[feats_train]
						y_train = base_train_pre_sec[target]

						X_test = base_test_pre_sec[feats_train]
						y_test = base_test_pre_sec[target]

						X_oot = base_oot_pre_sec[feats_train]
						y_oot = base_oot_pre_sec[target]
						print("Sector: " + s)
						print("Base Train %s Base Test %s Base OOT %s" % (base_train_pre_sec.shape, base_test_pre_sec.shape, base_oot_pre_sec.shape))

						y_score_train  =  model1.predict_proba(X_train)
						y_score_test   =  model1.predict_proba(X_test)
						y_score_oot    =  model1.predict_proba(X_oot)
						y_score_train  =  [item[1] for item in y_score_train]
						y_score_test   =  [item[1] for item in y_score_test]
						y_score_oot    =  [item[1] for item in y_score_oot]

						base_train_pre_sec[nom_proba] = y_score_train
						base_test_pre_sec[nom_proba] = y_score_test
						base_oot_pre_sec[nom_proba]  = y_score_oot

						#########################################################
						# Métricas precisión, mapeo y concentraciones: Totales
						#########################################################
						_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango_estricta(base_train_pre_sec, nom_proba, target[0], 'f_analisis')
						_, enrango_test, ordena_test, prct_desorden_test = metricas_ordena_rango_estricta(base_test_pre_sec, nom_proba, target[0], 'f_analisis')
						_, enrango_oot, ordena_oot, prct_desorden_oot = metricas_ordena_rango_estricta(base_oot_pre_sec, nom_proba, target[0], 'f_analisis')

						_,_,_, ordena_train_q, ordena_test_q, ordena_oot_q = metricas_ordena_sq_sector(base_train_pre,base_train_pre_sec, base_test_pre_sec, base_oot_pre_sec, bins, nom_proba, target[0], 'f_analisis')

						auc_train = metrics.roc_auc_score( y_train, y_score_train )
						auc_test  = metrics.roc_auc_score( y_test,  y_score_test )
						auc_oot   = metrics.roc_auc_score( y_oot,   y_score_oot )

						# AUC
						auc_train_cv.append(auc_train)
						auc_test_cv.append(auc_test)
						auc_oot_cv.append(auc_oot)
						# Escalas en rango
						enrango_train_cv.append(enrango_train)
						enrango_test_cv.append(enrango_test)
						enrango_oot_cv.append(enrango_oot)
						# Escalas ordenadas
						ordena_train_cv.append(ordena_train)
						ordena_test_cv.append(ordena_test)
						ordena_oot_cv.append(ordena_oot)
						# Escalas ordenadas q
						ordena_train_cv_q.append(ordena_train_q)
						ordena_test_cv_q.append(ordena_test_q)
						ordena_oot_cv_q.append(ordena_oot_q)
						# Porcentaje de desorden C
						prct_desorden_train_cv.append(prct_desorden_train)
						prct_desorden_test_cv.append( prct_desorden_test)
						prct_desorden_oot_cv.append(  prct_desorden_oot)


					# Promedio AUC
					auc_train_l.append(np.mean(auc_train_cv))
					auc_test_l.append(np.mean(auc_test_cv))
					auc_oot_l.append(np.mean(auc_oot_cv))

					# Promedio Escalas en rango
					enrango_train_l.append(np.mean(enrango_train_cv))
					enrango_test_l.append(np.mean(enrango_test_cv))
					enrango_oot_l.append(np.mean(enrango_oot_cv))

					# Promedio Escalas ordenadas
					ordena_train_l.append(np.mean(ordena_train_cv))
					ordena_test_l.append( np.mean(ordena_test_cv))
					ordena_oot_l.append(  np.mean(ordena_oot_cv))

					# Promedio Escalas ordenadas q
					ordena_train_q_l.append(np.mean(ordena_train_cv_q))
					ordena_test_q_l.append(np.mean(ordena_test_cv_q))
					ordena_oot_q_l.append(np.mean(ordena_oot_cv_q))

					# Promedio Porcentaje de desorden C
					prct_desorden_train_l.append(np.mean(prct_desorden_train_cv))
					prct_desorden_test_l.append(np.mean(prct_desorden_test_cv))
					prct_desorden_oot_l.append(np.mean(prct_desorden_oot_cv))

					typemodel_l.append(typemodel)
					paramters.append(hyperparameters)
					# sector 
					sector.append(s) 
					modelo_num.append(i)
				
			except:
				pass

	resultados = {'parameters': paramters,
				  'auc_tr': auc_train_l, 'auc_te': auc_test_l, 'auc_o': auc_oot_l,
				  'enrango_tr': enrango_train_l, 'enrango_te': enrango_test_l, 'enrango_o': enrango_oot_l,
				  'ordena_tr': ordena_train_l, 'ordena_te': ordena_test_l, 'ordena_o': ordena_oot_l,
				  'prct_desorden_tr': prct_desorden_train_l, 'prct_desorden_te': prct_desorden_test_l, 'prct_desorden_o': prct_desorden_oot_l,
				  'ordena_tr_q': ordena_train_q_l, 'ordena_te_q': ordena_test_q_l, 'ordena_o_q': ordena_oot_q_l,
				  'modelo' : typemodel_l, 'sector' : sector, 'modelo_num': modelo_num
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
					'ordena_q','ordena_tr_q', 'ordena_te_q', 'ordena_o_q','sector','modelo_num' ]
	else:
		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif',
					'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o']

	return res_df[columnas]

def mapeo_escala_cv_sector(sectores, modelo, df_train_test, df_oot, features, q_ = 20):
	
	y_score_train_test  =  modelo.predict_proba(df_train_test[features]) 
	y_score_oot    =  modelo.predict_proba(df_oot[features])

	y_score_train_test  =  [item[1] for item in y_score_train_test]
	y_score_oot    =  [item[1] for item in y_score_oot]

	nom_proba = 'proba'
	df_train_test[nom_proba] = y_score_train_test
	df_oot[nom_proba]   = y_score_oot

	for s in sectores:
		if s == 'Generico':	

			mtrte, mo, ordena_train_test_ee, ordena_oot_ee, bins = metricas_ordena_q_cv_sector( df_train_test, df_oot, q_, nom_proba, 'def_default_12m', 'f_analisis' )

			cols_1 = ['q', 'q_range']
			cols_n = ['tdo','prct','ordena','cantidad']

			mtrte = mtrte[cols_1+cols_n]
			mo  = mo[ cols_1+cols_n]

			mtrte.columns = cols_1 + [x+'_trte' for x in cols_n]
			mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

			df_consolidado = mtrte.merge(mo, how='outer', on = ['q','q_range'])
			mapeo_q = df_consolidado
			mapeo_q['sector'] = s
		else:
			base_train_pre_sec=df_train_test[df_train_test['sector'] == s]
			base_oot_pre_sec=df_oot[df_oot['sector'] == s]

			mtrte, mo, ordena_train_test_ee, ordena_oot_ee = metricas_ordena_sq_cv_sector( base_train_pre_sec, base_oot_pre_sec, bins, nom_proba, 'def_default_12m', 'f_analisis' )

			cols_1 = ['q', 'q_range']
			cols_n = ['tdo','prct','ordena','cantidad']

			mtrte = mtrte[cols_1+cols_n]
			mo  = mo[ cols_1+cols_n]

			mtrte.columns = cols_1 + [x+'_trte' for x in cols_n]
			mo.columns  = cols_1 + [x+'_o'  for x in cols_n]

			df_consolidado = mtrte.merge(mo, how='outer', on = ['q','q_range'])
			df_consolidado['sector'] = s
			mapeo_q = pd.concat([mapeo_q, df_consolidado])
	return mapeo_q


def metricas_ordena_q_cv_sector( base_train_test_pre_, base_oot_pre_, q__, nom_proba, var_default, var_conteo ):
	
	## Train-test
	base_train_test_pre_['q'] = pd.qcut(base_train_test_pre_[nom_proba], q__, labels = False, duplicates='drop')
	base_train_test_pre_['q_range'], bins = pd.qcut(base_train_test_pre_[nom_proba], q__, retbins=True, duplicates='drop' )
	cat = base_train_test_pre_[['q','q_range']].drop_duplicates()

	q_train_test = base_train_test_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_train_test.columns = ['q','cantidad','default']
	q_train_test['tdo'] = round( q_train_test['default'] / q_train_test['cantidad']*100,2)
	q_train_test['prct'] = round( q_train_test['cantidad'] / sum(q_train_test['cantidad'])*100,2)

	q_train_test.sort_values(by='q', inplace=True)
	max_val = q_train_test['tdo'].expanding(1).max()
	ini_val = pd.Series(-1)
	q_train_test['before_max_tdo'] = ini_val.append(max_val).reset_index(drop=True).drop(labels=len(q_train_test))
	q_train_test['ordena'] = (q_train_test['tdo'] > q_train_test['before_max_tdo']).astype(int)
	q_train_test = pd.merge(q_train_test, cat, how = 'left', on = 'q')
	ordena_train_test_q = sum(q_train_test['ordena'])

	## OOT
	base_oot_pre_['q'] = pd.cut(base_oot_pre_[nom_proba], bins, labels = False)
	base_oot_pre_['q_range'] = pd.cut(base_oot_pre_[nom_proba], bins)

	q_oot = base_oot_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_oot.columns = ['q','cantidad','default']
	q_oot['tdo'] = round( q_oot['default'] / q_oot['cantidad']*100,2)
	q_oot['prct'] = round( q_oot['cantidad'] / sum(q_oot['cantidad'])*100,2)

	q_oot.sort_values(by='q', inplace=True)
	max_val = q_oot['tdo'].expanding(1).max()
	ini_val = pd.Series(-1)
	q_oot['before_max_tdo'] = ini_val.append(max_val).reset_index(drop=True).drop(labels=len(q_oot))
	q_oot['ordena'] = (q_oot['tdo'] > q_oot['before_max_tdo']).astype(int)
	q_oot = pd.merge(q_oot, cat, how = 'left', on = 'q')
	ordena_oot_q = sum(q_oot['ordena'])

	return q_train_test, q_oot, ordena_train_test_q, ordena_oot_q, bins

def metricas_ordena_sq_cv_sector( base_train_test_pre_, base_oot_pre_, bins, nom_proba, var_default, var_conteo ):
	
	## Train-test
	base_train_test_pre_['q'] = pd.cut(base_train_test_pre_[nom_proba],bins, labels = False, duplicates='drop')
	base_train_test_pre_['q_range'] = pd.cut(base_train_test_pre_[nom_proba], bins, duplicates='drop')
	cat = base_train_test_pre_[['q','q_range']].drop_duplicates()

	q_train_test = base_train_test_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_train_test.columns = ['q','cantidad','default']
	q_train_test['tdo'] = round( q_train_test['default'] / q_train_test['cantidad']*100,2)
	q_train_test['prct'] = round( q_train_test['cantidad'] / sum(q_train_test['cantidad'])*100,2)

	q_train_test.sort_values(by='q', inplace=True)
	max_val = q_train_test['tdo'].expanding(1).max()
	ini_val = pd.Series(-1)
	q_train_test['before_max_tdo'] = ini_val.append(max_val).reset_index(drop=True).drop(labels=len(q_train_test))
	q_train_test['ordena'] = (q_train_test['tdo'] > q_train_test['before_max_tdo']).astype(int)
	q_train_test = pd.merge(q_train_test, cat, how = 'left', on = 'q')
	ordena_train_test_q = sum(q_train_test['ordena'])

	## OOT
	base_oot_pre_['q'] = pd.cut(base_oot_pre_[nom_proba], bins, labels = False)
	base_oot_pre_['q_range'] = pd.cut(base_oot_pre_[nom_proba], bins)
	
	q_oot = base_oot_pre_.groupby(['q']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	q_oot.columns = ['q','cantidad','default']
	q_oot['tdo'] = round( q_oot['default'] / q_oot['cantidad']*100,2)
	q_oot['prct'] = round( q_oot['cantidad'] / sum(q_oot['cantidad'])*100,2)

	q_oot.sort_values(by='q', inplace=True)
	max_val = q_oot['tdo'].expanding(1).max()
	ini_val = pd.Series(-1)
	q_oot['before_max_tdo'] = ini_val.append(max_val).reset_index(drop=True).drop(labels=len(q_oot))
	q_oot['ordena'] = (q_oot['tdo'] > q_oot['before_max_tdo']).astype(int)
	q_oot = pd.merge(q_oot, cat, how = 'left', on = 'q')
	ordena_oot_q = sum(q_oot['ordena'])

	return q_train_test, q_oot, ordena_train_test_q, ordena_oot_q

def metricas_ordena_rango_estricta( base, nom_proba, var_default, var_conteo ):
	base['c'] = base[nom_proba].apply( escala_pj )
	c_entr = base.groupby(['c']).agg({var_conteo:'count', var_default:'sum'}).reset_index()
	c_entr.columns = ['c','cantidad','default']
	c_entr['tdo'] = round( c_entr['default'] / c_entr['cantidad']*100,2)
	c_entr['prct'] = round( c_entr['cantidad'] / sum(c_entr['cantidad'])*100,2)
	c_entr['c_obs'] = c_entr['tdo'].apply( lambda x: escala_pj(x/100) )

	en_rango = sum(c_entr['c']==c_entr['c_obs'])

	c_entr.sort_values(by='c', inplace=True)
	max_val = c_entr['tdo'].expanding(1).max()
	ini_val = pd.Series(-1)
	c_entr['before_max_tdo'] = ini_val.append(max_val).reset_index(drop=True).drop(labels=len(c_entr))
	c_entr['ordena'] = (c_entr['tdo'] > c_entr['before_max_tdo']).astype(int)
	ordena = sum(c_entr['ordena'])

	prct_desorden = sum(c_entr[c_entr['ordena']==0]['prct'])

	return c_entr, en_rango, ordena, prct_desorden

########################################################################################################################
# Esta funcion genera la transformacion de q a C: retorna el ordenamiento y el mapeo
# Autor: wiareval
########################################################################################################################
def mapeo_q_a_c_estricta(base_final, feats_total_, var_partition, model, escala_maestra, zpredict = None, li_p=0, ls_p=1):
	"""Funcion que permite mapear de Q calibrada a C
	Input:
		base_final: dataframe pandas, base completa con la que se entrena el modelo
		feats_final: lista, Lista con la variables finales que se utilizan para entrenar el modelo
		var_partition: string, variable por la que se particiona Train, Test y OOT
		model: model sklearn, modelo entrenado
		zpredict: vector de transformacion sklearn, vector de calibracion.
		escala_maestra: dataframe pandas con escala maestra, cols: C, l_inf,l_sup
	Output:
		df_consolidado: dataframe, devuelve las escalas mapeadas a C.
	"""
	base_final_ = base_final.copy()
	base_final_['prob_q'] = [item[1] for item in model.predict_proba(base_final_[feats_total_])]

	if zpredict is not None:
		y_0 = [ i if (i >= li_p and i <= ls_p) else li_p if i < li_p else ls_p for i in base_final_['prob_q'] ]
		yp =  zpredict(y_0)		
		base_final_['prob_aj'] = yp
		mtr, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango_estricta( base_final_[base_final_[var_partition]==1], 'prob_aj', 'def_default_12m', 'f_analisis' )
		mte, enrango_test_ee, ordena_test_ee,_   = metricas_ordena_rango_estricta( base_final_[base_final_[var_partition]==0],  'prob_aj', 'def_default_12m', 'f_analisis' )
		mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango_estricta( base_final_[base_final_[var_partition]==2],   'prob_aj', 'def_default_12m', 'f_analisis' )
	else:
		mtr, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango_estricta( base_final_[base_final_[var_partition]==1], 'prob_q', 'def_default_12m', 'f_analisis' )
		mte, enrango_test_ee, ordena_test_ee,_   = metricas_ordena_rango_estricta( base_final_[base_final_[var_partition]==0],  'prob_q', 'def_default_12m', 'f_analisis' )
		mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango_estricta( base_final_[base_final_[var_partition]==2],   'prob_q', 'def_default_12m', 'f_analisis' )
	
	mtr['rango'] = mtr['c'] == mtr['c_obs']
	mte['rango'] = mte['c'] == mte['c_obs']
	mo['rango']  = mo[ 'c'] == mo[ 'c_obs']
	cols_1 = ['c']
	cols_n = ['tdo','prct','rango','ordena','cantidad']
	mtr = mtr[cols_1+cols_n]
	mte = mte[cols_1+cols_n]
	mo  = mo[ cols_1+cols_n]

	mtr.columns = cols_1 + [x+'_tr' for x in cols_n]
	mte.columns = cols_1 + [x+'_te' for x in cols_n]
	mo.columns  = cols_1 + [x+'_o'  for x in cols_n]
	
	df_consolidado = escala_maestra.merge(mtr, how='outer').merge(mte, how='outer').merge(mo, how='outer')
	return df_consolidado

########################################################################################################################
# Busqueda de hiperparametros con CV estratificado
# Autor: mialopez
# Description Búsqueda de parámetros por CrossValidation para el modelo financiero multisector, 
# basado en la propuesta de train_nueva_era_segint_w_cv_sector cambia el tipo de CV por estratificado y genera
#  los k-folds de acuerdo a la variable sector, seguidamente guardo los k-folds de acuerdo al index 
# lo que genera que un cliente puede quedar en train y test, debido a que los registros aparecen únicamente 
# cuando cambian de estados financieros no implica un problema para la modelación.
########################################################################################################################

def train_nueva_era_segint_w_cvsk_sector(base_final_tr_te, base_oot, target, n_iter, seed, feats_train, parametros, muestras_en_intervalo, n_folds = 5, escala_c = True, q = 20):
	from sklearn.model_selection import StratifiedKFold
	target = ['def_default_12m']
	target_skfold = 'sector'
	random.seed(seed)

	#########################################################
	# Hiperparámetros Algoritmos
	#########################################################
	learning_ra = parametros['learning_ra']
	estimadores = parametros['estimadores']
	profundidad = parametros['profundidad']
	min_data_le = parametros['min_data_le']
	max_bins___ = parametros['max_bins___']
	n_leaves___ = parametros['n_leaves___']

	params_lgbm = {
		'learning_rate'    : learning_ra,
		'max_bin'          : max_bins___,
		'n_estimators'     : estimadores,
		'num_leaves'       : n_leaves___,
		'max_depth'        : profundidad,
		'min_data_in_leaf' : min_data_le
	}
	param_hgb = {
			'loss'               : ['auto'],
			'learning_rate'      : learning_ra,
			'max_iter'           : estimadores,
			'max_leaf_nodes'     : n_leaves___,
			'max_depth'          : profundidad,
			'min_samples_leaf'   : min_data_le,
			'l2_regularization'  : [float(x) for x in np.linspace(start = 0, stop = 0.5, num = muestras_en_intervalo)],
			'max_bins'           : max_bins___,
			'random_state'       : [seed],
			'tol'                : [1e-7]
			}
	dict_hpparams = {'lgbm':params_lgbm, 'hgb': param_hgb}
	#########################################################
	# Resultados a medir: totales
	#########################################################
	semilla = []; paramters = [];
	auc_train_l = [];     auc_test_l = [];     auc_oot_l = [];
	enrango_train_l = []; enrango_test_l = []; enrango_oot_l = [];
	ordena_train_l = [];  ordena_test_l = [];  ordena_oot_l = [];
	ordena_train_q_l = [];  ordena_test_q_l = [];  ordena_oot_q_l = []; 
	prct_desorden_train_l = []; prct_desorden_test_l = []; prct_desorden_oot_l = [];
	typemodel_l = []; sector = []; modelo_num = []

	nom_proba = 'proba'
	#########################################################
	# Iteraciones
	#########################################################
	for i in range(n_iter):
		if i % 10 == 0:
			print('******************************************************************** ')
			print('******************************************************************** ')
			print('Iteración: ' + str(i))
			print('******************************************************************** ')
			print('******************************************************************** ')
		for key in dict_hpparams:
			try: 
				if key == 'lgbm':
					semilla.append(seed)
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)
					model1 = lgb.LGBMClassifier(
					learning_rate = hyperparameters['learning_rate'],
					max_bin = hyperparameters['max_bin'],
					n_estimators = hyperparameters['n_estimators'],
					num_leaves = hyperparameters['num_leaves'],
					max_depth = hyperparameters['max_depth'],
					min_data_in_leaf = hyperparameters['min_data_in_leaf'],
					n_jobs = -3,
					#silent = True,
					verbose=-1,
					seed = seed
					)
					typemodel = 'Lighgbm'
				else :
					semilla.append(seed)
					hyperparameters = {k: random.sample(v, 1)[0] for k, v in dict_hpparams[key].items()}
					#paramters.append(hyperparameters)
					#print(hyperparameters)
					model1 = HistGradientBoostingClassifier(
					loss = hyperparameters['loss'],
					learning_rate = hyperparameters['learning_rate'],
					max_iter = hyperparameters['max_iter'],
					max_leaf_nodes = hyperparameters['max_leaf_nodes'],
					max_depth = hyperparameters['max_depth'],
					min_samples_leaf = hyperparameters['min_samples_leaf'],
					l2_regularization = hyperparameters['l2_regularization'],
					max_bins = hyperparameters['max_bins'],
					tol = hyperparameters['tol'],
					#validation_fraction = hyperparameters['validation_fraction'],
					#n_jobs = -3,
					random_state = seed
					)
					typemodel = 'HistGradientBoosting'
				
				#########################################################
				# Cross Validation
				#########################################################
				skf = StratifiedKFold(n_splits = n_folds,shuffle=True, random_state=seed)
				stratifies_kf = base_final_tr_te.loc[:,target_skfold]
				auc_train_cv = []; auc_test_cv = []; auc_oot_cv = [];
				enrango_train_cv = []; enrango_test_cv = []; enrango_oot_cv = [];
				ordena_train_cv = []; ordena_test_cv = []; ordena_oot_cv = [];
				ordena_train_cv_q = []; ordena_test_cv_q = []; ordena_oot_cv_q = [];
				prct_desorden_train_cv = []; prct_desorden_test_cv = []; prct_desorden_oot_cv = [];
				k_rs_fold_train = {}; k_rs_fold_test = {}

				for n,tupla in enumerate(skf.split(base_final_tr_te, stratifies_kf)):
					k_rs_fold_train[f'kf{n}_index'] = tupla[0]
					k_rs_fold_test[f'kf{n}_index'] = tupla[1]
					base_train_pre = base_final_tr_te.loc[tupla[0],:]
					base_test_pre = base_final_tr_te.loc[tupla[1],:]
					base_oot_pre   = base_oot                

					X_train = base_train_pre[feats_train]
					y_train = base_train_pre[target]

					X_test = base_test_pre[feats_train]
					y_test = base_test_pre[target]

					X_oot = base_oot_pre[feats_train]
					y_oot = base_oot_pre[target]
					print("Sector: Generico")
					print("Base Train %s Base Test %s Base OOT %s" % (base_train_pre.shape, base_test_pre.shape, base_oot_pre.shape))

					#########################################################
					# Entrenamiento 
					#########################################################
					model1.fit(X_train, np.ravel(y_train))
					#########################################################
					########### REPORTE Parciales de cada modelo ############
					#########################################################
					y_score_train  =  model1.predict_proba(X_train)
					y_score_test   =  model1.predict_proba(X_test)
					y_score_oot    =  model1.predict_proba(X_oot)
					y_score_train  =  [item[1] for item in y_score_train]
					y_score_test   =  [item[1] for item in y_score_test]
					y_score_oot    =  [item[1] for item in y_score_oot]

					base_train_pre[nom_proba] = y_score_train
					base_test_pre[nom_proba] = y_score_test
					base_oot_pre[nom_proba]  = y_score_oot

					#########################################################
					# Métricas precisión, mapeo y concentraciones: Totales
					#########################################################
					_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango_estricta(base_train_pre, nom_proba, target[0], 'f_analisis')
					_, enrango_test, ordena_test, prct_desorden_test = metricas_ordena_rango_estricta(base_test_pre, nom_proba, target[0], 'f_analisis')
					_, enrango_oot, ordena_oot, prct_desorden_oot = metricas_ordena_rango_estricta(base_oot_pre, nom_proba, target[0], 'f_analisis')

					_,_,_, ordena_train_q, ordena_test_q, ordena_oot_q, bins = metricas_ordena_q_sector(base_train_pre, base_test_pre, base_oot_pre, q, nom_proba, target[0], 'f_analisis')

					auc_train = metrics.roc_auc_score( y_train, y_score_train )
					auc_test  = metrics.roc_auc_score( y_test,  y_score_test )
					auc_oot   = metrics.roc_auc_score( y_oot,   y_score_oot )

					# AUC
					auc_train_cv.append(auc_train)
					auc_test_cv.append(auc_test)
					auc_oot_cv.append(auc_oot)
					# Escalas en rango
					enrango_train_cv.append(enrango_train)
					enrango_test_cv.append(enrango_test)
					enrango_oot_cv.append(enrango_oot)
					# Escalas ordenadas
					ordena_train_cv.append(ordena_train)
					ordena_test_cv.append(ordena_test)
					ordena_oot_cv.append(ordena_oot)
					# Escalas ordenadas q
					ordena_train_cv_q.append(ordena_train_q)
					ordena_test_cv_q.append(ordena_test_q)
					ordena_oot_cv_q.append(ordena_oot_q)
					# Porcentaje de desorden C
					prct_desorden_train_cv.append(prct_desorden_train)
					prct_desorden_test_cv.append( prct_desorden_test)
					prct_desorden_oot_cv.append(  prct_desorden_oot)


				# Promedio AUC
				auc_train_l.append(np.mean(auc_train_cv))
				auc_test_l.append(np.mean(auc_test_cv))
				auc_oot_l.append(np.mean(auc_oot_cv))

				# Promedio Escalas en rango
				enrango_train_l.append(np.mean(enrango_train_cv))
				enrango_test_l.append(np.mean(enrango_test_cv))
				enrango_oot_l.append(np.mean(enrango_oot_cv))

				# Promedio Escalas ordenadas
				ordena_train_l.append(np.mean(ordena_train_cv))
				ordena_test_l.append( np.mean(ordena_test_cv))
				ordena_oot_l.append(  np.mean(ordena_oot_cv))

				# Promedio Escalas ordenadas q
				ordena_train_q_l.append(np.mean(ordena_train_cv_q))
				ordena_test_q_l.append(np.mean(ordena_test_cv_q))
				ordena_oot_q_l.append(np.mean(ordena_oot_cv_q))

				# Promedio Porcentaje de desorden C
				prct_desorden_train_l.append(np.mean(prct_desorden_train_cv))
				prct_desorden_test_l.append(np.mean(prct_desorden_test_cv))
				prct_desorden_oot_l.append(np.mean(prct_desorden_oot_cv))

				typemodel_l.append(typemodel)
				paramters.append(hyperparameters)
				# sector 
				sector.append("Generico") 
				modelo_num.append(i)


				sectores = ['Agropecuario','Comercio','Edificaciones','Infraestructura','Manufactura','Servicios','RR_Naturales']
				print(sectores)
				for s in sectores:
					print(s)
					auc_train_cv = []; auc_test_cv = []; auc_oot_cv = [];
					enrango_train_cv = []; enrango_test_cv = []; enrango_oot_cv = [];
					ordena_train_cv = []; ordena_test_cv = []; ordena_oot_cv = [];
					ordena_train_cv_q = []; ordena_test_cv_q = []; ordena_oot_cv_q = [];
					prct_desorden_train_cv = []; prct_desorden_test_cv = []; prct_desorden_oot_cv = [];
					
					base_final_sec = base_final_tr_te[base_final_tr_te['sector'] == s]
					base_oot_sec = base_oot[base_oot['sector'] == s]

					for clave, valor in k_rs_fold_train.items():
						base_train_pre_sec = base_final_sec[base_final_sec.index.isin(k_rs_fold_train[clave].tolist())]
						base_test_pre_sec = base_final_sec[base_final_sec.index.isin(k_rs_fold_test[clave].tolist())]
						base_oot_pre_sec   = base_oot_sec               

						X_train = base_train_pre_sec[feats_train]
						y_train = base_train_pre_sec[target]

						X_test = base_test_pre_sec[feats_train]
						y_test = base_test_pre_sec[target]

						X_oot = base_oot_pre_sec[feats_train]
						y_oot = base_oot_pre_sec[target]
						print("Sector: " + s)
						print("Base Train %s Base Test %s Base OOT %s" % (base_train_pre_sec.shape, base_test_pre_sec.shape, base_oot_pre_sec.shape))

						y_score_train  =  model1.predict_proba(X_train)
						y_score_test   =  model1.predict_proba(X_test)
						y_score_oot    =  model1.predict_proba(X_oot)
						y_score_train  =  [item[1] for item in y_score_train]
						y_score_test   =  [item[1] for item in y_score_test]
						y_score_oot    =  [item[1] for item in y_score_oot]

						base_train_pre_sec[nom_proba] = y_score_train
						base_test_pre_sec[nom_proba] = y_score_test
						base_oot_pre_sec[nom_proba]  = y_score_oot

						#########################################################
						# Métricas precisión, mapeo y concentraciones: Totales
						#########################################################
						_, enrango_train, ordena_train, prct_desorden_train = metricas_ordena_rango_estricta(base_train_pre_sec, nom_proba, target[0], 'f_analisis')
						_, enrango_test, ordena_test, prct_desorden_test = metricas_ordena_rango_estricta(base_test_pre_sec, nom_proba, target[0], 'f_analisis')
						_, enrango_oot, ordena_oot, prct_desorden_oot = metricas_ordena_rango_estricta(base_oot_pre_sec, nom_proba, target[0], 'f_analisis')

						_,_,_, ordena_train_q, ordena_test_q, ordena_oot_q = metricas_ordena_sq_sector(base_train_pre,base_train_pre_sec, base_test_pre_sec, base_oot_pre_sec, bins, nom_proba, target[0], 'f_analisis')

						auc_train = metrics.roc_auc_score( y_train, y_score_train )
						auc_test  = metrics.roc_auc_score( y_test,  y_score_test )
						auc_oot   = metrics.roc_auc_score( y_oot,   y_score_oot )

						# AUC
						auc_train_cv.append(auc_train)
						auc_test_cv.append(auc_test)
						auc_oot_cv.append(auc_oot)
						# Escalas en rango
						enrango_train_cv.append(enrango_train)
						enrango_test_cv.append(enrango_test)
						enrango_oot_cv.append(enrango_oot)
						# Escalas ordenadas
						ordena_train_cv.append(ordena_train)
						ordena_test_cv.append(ordena_test)
						ordena_oot_cv.append(ordena_oot)
						# Escalas ordenadas q
						ordena_train_cv_q.append(ordena_train_q)
						ordena_test_cv_q.append(ordena_test_q)
						ordena_oot_cv_q.append(ordena_oot_q)
						# Porcentaje de desorden C
						prct_desorden_train_cv.append(prct_desorden_train)
						prct_desorden_test_cv.append( prct_desorden_test)
						prct_desorden_oot_cv.append(  prct_desorden_oot)


					# Promedio AUC
					auc_train_l.append(np.mean(auc_train_cv))
					auc_test_l.append(np.mean(auc_test_cv))
					auc_oot_l.append(np.mean(auc_oot_cv))

					# Promedio Escalas en rango
					enrango_train_l.append(np.mean(enrango_train_cv))
					enrango_test_l.append(np.mean(enrango_test_cv))
					enrango_oot_l.append(np.mean(enrango_oot_cv))

					# Promedio Escalas ordenadas
					ordena_train_l.append(np.mean(ordena_train_cv))
					ordena_test_l.append( np.mean(ordena_test_cv))
					ordena_oot_l.append(  np.mean(ordena_oot_cv))

					# Promedio Escalas ordenadas q
					ordena_train_q_l.append(np.mean(ordena_train_cv_q))
					ordena_test_q_l.append(np.mean(ordena_test_cv_q))
					ordena_oot_q_l.append(np.mean(ordena_oot_cv_q))

					# Promedio Porcentaje de desorden C
					prct_desorden_train_l.append(np.mean(prct_desorden_train_cv))
					prct_desorden_test_l.append(np.mean(prct_desorden_test_cv))
					prct_desorden_oot_l.append(np.mean(prct_desorden_oot_cv))

					typemodel_l.append(typemodel)
					paramters.append(hyperparameters)
					# sector 
					sector.append(s) 
					modelo_num.append(i)
				
			except:
				pass

	resultados = {'parameters': paramters,
				  'auc_tr': auc_train_l, 'auc_te': auc_test_l, 'auc_o': auc_oot_l,
				  'enrango_tr': enrango_train_l, 'enrango_te': enrango_test_l, 'enrango_o': enrango_oot_l,
				  'ordena_tr': ordena_train_l, 'ordena_te': ordena_test_l, 'ordena_o': ordena_oot_l,
				  'prct_desorden_tr': prct_desorden_train_l, 'prct_desorden_te': prct_desorden_test_l, 'prct_desorden_o': prct_desorden_oot_l,
				  'ordena_tr_q': ordena_train_q_l, 'ordena_te_q': ordena_test_q_l, 'ordena_o_q': ordena_oot_q_l,
				  'modelo' : typemodel_l, 'sector' : sector, 'modelo_num': modelo_num
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
					'ordena_q','ordena_tr_q', 'ordena_te_q', 'ordena_o_q','sector','modelo_num' ]
	else:
		columnas = ['parameters', 'modelo',
					'ar_tr','ar_te','ar_o', 'auc_dif',
					'enrango','enrango_tr', 'enrango_te', 'enrango_o', 'ordena','ordena_tr', 'ordena_te', 'ordena_o']

	return res_df[columnas]


def mapeo_q_a_c_cv_estricta(base_train_test, base_oot, feats_total_, model, escala_maestra, zpredict = None):
	"""Funcion que permite mapear de Q calibrada a C
	Input:
		base_final: dataframe pandas, base completa con la que se entrena el modelo
		feats_final: lista, Lista con la variables finales que se utilizan para entrenar el modelo
		var_partition: string, variable por la que se particiona Train, Test y OOT
		model: model sklearn, modelo entrenado
		zpredict: vector de transformacion sklearn, vector de calibracion.
		escala_maestra: dataframe pandas con escala maestra, cols: C, l_inf,l_sup
	Output:
		df_consolidado: dataframe, devuelve las escalas mapeadas a C.
	"""
	base_train_test['prob_q'] = [item[1] for item in model.predict_proba(base_train_test[feats_total_])]
	base_oot['prob_q'] = [item[1] for item in model.predict_proba(base_oot[feats_total_])]
	if zpredict is not None:
		base_train_test['prob_aj'] = zpredict(base_train_test['prob_q'])
		base_oot['prob_aj'] = zpredict(base_oot['prob_q'])
		mtrte, enrango_train_test_ee, ordena_train_test_ee,_ = metricas_ordena_rango_estricta( base_train_test, 'prob_aj', 'def_default_12m', 'f_analisis' )
		mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango_estricta( base_oot,   'prob_aj', 'def_default_12m', 'f_analisis' )
	else:
		mtrte, enrango_train_ee, ordena_train_ee,_ = metricas_ordena_rango_estricta( base_train_test, 'prob_q', 'def_default_12m', 'f_analisis' )
		mo, enrango_oot_ee, ordena_oot_ee,_      = metricas_ordena_rango_estricta( base_oot,   'prob_q', 'def_default_12m', 'f_analisis' )
	mtrte['rango'] = mtrte['c'] == mtrte['c_obs']
	mo['rango']  = mo[ 'c'] == mo[ 'c_obs']
	cols_1 = ['c']
	cols_n = ['tdo','prct','rango','ordena','cantidad']
	mtrte = mtrte[cols_1+cols_n]
	mo  = mo[ cols_1+cols_n]

	mtrte.columns = cols_1 + [x+'_trte' for x in cols_n]
	mo.columns  = cols_1 + [x+'_o'  for x in cols_n]
	
	df_consolidado = escala_maestra.merge(mtrte, how='outer').merge(mo, how='outer')
	return df_consolidado


def metricas_top_models_rn_imp(base_x, base_y, base_x_tr, base_x_te, base_x_o, varis, ronda_i, num_top_models = 10, ronda_completa = False, var1= ["auc_dif", 0.05, True], var2 = ["ordena_te_q", 14, False], q = 20):
	"""
	Esta función devuelve el listado de métricas y modelos que se entrenan a partir de la función 
	fm.metricas para varios modelos. Sin necesidad de copiar y pegar los parametros uno a uno.
	
	Se obtienen los primeros n_top_models en base a su ordenamiento en test y su auc_dif.
	
	Parametros:
	- ronda_i: base de datos con los modelos de una iteración (ronda)
	- num_top_models: número de modelos a comparar
	- ordena_by: metrica para usar en el top
	- ascending: ordena de forma ascendente o descendente
	
	Resultado: Lista donde cada elemento es una tupla -> [(tv_0, model_0), ..., (tv_n, model_n)]
	"""
   
	if ronda_completa : 
		ronda_i_top = ronda_i.loc[(ronda_i[var1[0]] <= var1[1]) & (ronda_i[var2[0]] >= var2[1])]
	else: 
		ronda_i_top = ronda_i.loc[ronda_i['ordena_te_q'].isin(ronda_i.sort_values('ordena_te_q', ascending = False).head(50)['ordena_te_q'].unique())]
 
	ronda_i_top = ronda_i_top.sort_values([var1[0], var2[0]], ascending=[var1[2], var2[2]])
	ronda_i_top['rn_var1'] = tuple(zip(ronda_i_top[var1[0]], ronda_i_top[var2[0]]))
	ronda_i_top['rn_var1'] = ronda_i_top[['rn_var1']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top = ronda_i_top.sort_values([var2[0], var1[0]], ascending=[var2[2], var1[2]])
	ronda_i_top['rn_var2']=tuple(zip(ronda_i_top[var2[0]], ronda_i_top[var1[0]]))
	ronda_i_top['rn_var2'] = ronda_i_top[['rn_var2']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top['rn'] = ronda_i_top['rn_var2'] * ronda_i_top['rn_var1']

	ronda_i_top = ronda_i_top.sort_values(by="rn", ascending=True).head(num_top_models)

	# Bases de entrenamiento, prueba y oot con una feature aleatoria para top models lgbm
	varis_with_random = varis.copy()
	varis_with_random.append("random_feature")
	base_x_with_random = base_x.copy()
	base_x_with_random["random_feature"] = np.random.rand(len(base_x_with_random))
	base_x_tr_with_random = base_x_tr.copy()
	base_x_tr_with_random["random_feature"] = base_x_with_random["random_feature"]
	base_x_te_with_random  = base_x_te.copy()
	base_x_te_with_random["random_feature"] = np.random.rand(len(base_x_te_with_random))
	base_x_o_with_random = base_x_o.copy()
	base_x_o_with_random["random_feature"] = np.random.rand(len(base_x_o_with_random))
	
	# Inicializacion de vector modelos y vector impotancias
	modelos = []
	importance_top_models = np.zeros(len(varis))
	for index, row in ronda_i_top.iterrows():
		# Hiper-parametros del modelo row
		print("modelo_", index, sep = "")
		hyperparameters_row = eval(str(ronda_i_top.loc[index,:]['parameters']))
		# Tipo del modelo y creacion del mismo

		if str(ronda_i_top.loc[index,:]['modelo']) == 'HistGradientBoosting':
			tipo_modelo = "hgb"
			modelx = HistGradientBoostingClassifier( loss = hyperparameters_row['loss'],learning_rate = hyperparameters_row['learning_rate'],
					max_iter = hyperparameters_row['max_iter'],max_leaf_nodes = hyperparameters_row['max_leaf_nodes'],
					max_depth = hyperparameters_row['max_depth'],min_samples_leaf = hyperparameters_row['min_samples_leaf'],
					l2_regularization = hyperparameters_row['l2_regularization'],max_bins = hyperparameters_row['max_bins'],
					tol = hyperparameters_row['tol'],random_state = 42 )
		else:
			tipo_modelo = "lgbm"
			modelx = lgb.LGBMClassifier( learning_rate = hyperparameters_row['learning_rate'], max_bin = hyperparameters_row['max_bin'],
								n_estimators = hyperparameters_row['n_estimators'], num_leaves = hyperparameters_row['num_leaves'],
								max_depth = hyperparameters_row['max_depth'], min_data_in_leaf = hyperparameters_row['min_data_in_leaf'],
								n_jobs = -3, silent = True, seed = 42 )
		
		model_row = modelx.fit(base_x, base_y)
		tv_row = mapeo_escala_estricto( model_row, base_x_tr, base_x_te, base_x_o, varis , escala_c = False, q_ = q)
		#tv_row = tv_row.fillna(0)
		
		# Se guarda la información de los modelos en una lista
		modelos.append((tv_row, model_row))
		# Cración de ranking por impórtancia
		if tipo_modelo == "lgbm":
			importance_top_models += model_row.feature_importances_
		print("\n")
	df_importance_top_models = pd.DataFrame({"feats" : varis, "all_top_importance" : importance_top_models})
	df_importance_top_models = df_importance_top_models.sort_values(by = "all_top_importance", ascending = False).reset_index(drop = True).reset_index()
	df_importance_top_models.rename(columns = {"index" : "rn_imp"}, inplace = True)
	df_importance_top_models["rn_imp"] = df_importance_top_models["rn_imp"] + 1

	# return modelos
	return df_importance_top_models

# wimunera 20221019: Funcion necesaria para evaluar_ordenamiento
def valores_importantes( df_x ):
    df__ = pd.DataFrame.from_dict({'valor':[x for x in df_x.feature_grids], 'pdp':[x for x in df_x.pdp]})
    
    df__r = df__.groupby('pdp').agg( Min = ('valor', np.min), Max = ( 'valor', np.max ) ).reset_index().sort_values('Min')
    return df__r

## wimunera 20221019: Funcion para probar ordenamiento de un conjunto de modelos
def evaluar_ordenamiento(best_models, feats_train, feats_revision, feats_ordena_riesgo, num_grid_points_, base_train_pre,  base_test_pre, base_oot_pre, target, ruta_resultados, sector): 
	
	# Inicializar variables de resultado
	from pdpbox import pdp
	df_pdp = pd.DataFrame()
	df_acum_full = pd.DataFrame()
	list_desordenados = []
	cont_models = 1
	best_models = best_models.reset_index()

	# For para recorrer todos los modelos de best_models
	for index, row in best_models.iterrows():
		
		# Informacion del modelo
		flg_pass = False ## Bandera para indicar si se suspende validacion de un modelo
		print( '\033[1m' + '\n(' + str(cont_models) + '/' + str(best_models.shape[0]) + '). ANALIZANDO MODELO ' + best_models.loc[index,:]['ronda'] + "_" + str(best_models.loc[index,:]['index']) + '\033[0m' )
		hyperparameters = eval(str(best_models.loc[index,:]['parameters']))
			
		# Tipo del modelo y ajuste 
		if str(best_models.loc[index,:]['modelo']) == 'HistGradientBoosting':
			algoritmo = 'hgb'
		else: 
			algoritmo = 'lgbm' 
			
		model_i = fit_model_group(hyperparameters, base_train_pre, base_test_pre, base_oot_pre, target, feats_train, agrupacion='NA', algoritmo=algoritmo)

		# Inicializar df
		df_acum = pd.DataFrame()
		cont = 1 
		for f in feats_revision:
			
			## Validar si variable tiene riesgo_esperado 
			if f not in feats_ordena_riesgo:
				feats_ordena_riesgo = {**feats_ordena_riesgo, **{f:'Indeterminado'}}

			print('================= ' + str(cont) + '/' + str(len(feats_train)) + '. Analizando variable ' + f + '...' )
			pdp_feat_df = pdp.pdp_isolate(model= model_i, dataset= base_train_pre, model_features= feats_train, feature=f, num_grid_points = num_grid_points_ )
			fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_feat_df, center=True, x_quantile=True, ncols=3, plot_lines=True, frac_to_plot=100, figsize=(15,6), feature_name=f )

			df_f = valores_importantes( pdp_feat_df )
			df_f['variable'] = f

			df_acum = df_acum.append( df_f )
				
			## validar ordenamiento para suspender proceso
			df_f.reset_index(drop=True, inplace=True)
			df_f['diff'] = df_f.groupby(['variable'])['pdp'].diff()
			df_f = df_f.groupby(['variable'])['diff'].describe().reset_index()
			df_f = df_f.rename({'count': 'conteo'}, axis=1)
			df_f['signo_min'] = [ 1 if x > 0 else 0 for x in df_f['min'] ]
			df_f['signo_max'] = [ 1 if x > 0 else 0 for x in df_f['max'] ]
			df_f['signos'] = df_f['signo_min'] + df_f['signo_max']
			cont += 1 
				
			if (df_f.signos[0] == 1) | ( (df_f.conteo[0] != 0) & (df_f.signos[0] == 0) & (feats_ordena_riesgo[f] == 'Aumenta') ) | ( (df_f.conteo[0] != 0) & (df_f.signos[0] == 2) & (feats_ordena_riesgo[f] == 'Disminuye') ) :
				print('Variable ' + f + ' está desordenada en el modelo ' + best_models.loc[index,:]['ronda'] + "_" + str(best_models.loc[index,:]['index']) )
				print('Se suspende validación del modelo')
				list_desordenados = list_desordenados + [ best_models.loc[index,:]['ronda'] + "_" + str(best_models.loc[index,:]['index']) ]
				flg_pass = True
				break
			
		if flg_pass:
			pass
			
		## Retomar proceso
		df_acum['hyperparameters'] = str(hyperparameters)
		df_acum['index_i'] = str(best_models.loc[index,:]['index'])
		df_acum['ronda'] = str(best_models.loc[index,:]['ronda'])
					
		df_acum_full = pd.concat([df_acum_full, df_acum])
			
		### Calcular total de variables desordenadas
		df_new = df_acum.copy()
		df_new.reset_index(drop=True, inplace=True)
		df_new['diff'] = df_new.groupby(['variable'])['pdp'].diff()
		df_new_ = df_new.groupby(['variable'])['diff'].describe().reset_index()
		df_new_['signo_min'] = [ 1 if x > 0 else 0 for x in df_new_['min'] ]
		df_new_['signo_max'] = [ 1 if x > 0 else 0 for x in df_new_['max'] ]
		df_new_['signos'] = df_new_['signo_min'] + df_new_['signo_max']
		df_new_.sort_values('signos')
		df_new_['hyperparameters'] = str(hyperparameters)
		df_new_['index_i'] = str(best_models.loc[index,:]['index'])
		df_new_['ronda'] = str(best_models.loc[index,:]['ronda'])

		df_pdp = pd.concat([df_pdp, df_new_])
			
		print('\nModelo: ' + str(hyperparameters))
		cont_models += 1
		print('Cantidad de variables desordenadas: ' + str(len(df_new_.loc[ df_new_.signos == 1,  ] ) ) )

		## Escribir resultados	
		df_acum_full.to_excel( ruta_resultados + '\\df_acum_full_' + str(num_grid_points_) + '.xlsx')
		df_pdp.to_excel(ruta_resultados + '\\df_pdp_' + str(num_grid_points_) + '.xlsx')
		with open(ruta_resultados + '\\list_desordenados_' + str(num_grid_points_) + '.txt'.format(sector.lower()), 'w') as file:
			for item in list_desordenados:
				file.write('%s\n' % item)

	return df_pdp, df_acum_full, list_desordenados


# sargumed: Función para crear intervalos de confianza de la TDO y compararlos con los rangos de la escala maestra
def intervalos(df, z=1.96, ms='tr'):
	df['li_' + ms] = df['tdo_' + ms] - z*np.sqrt((df['tdo_' + ms] * (1-df['tdo_' + ms]))/df['cantidad_' + ms])
	df['ls_' + ms] = df['tdo_' + ms] + z*np.sqrt((df['tdo_' + ms] * (1-df['tdo_' + ms]))/df['cantidad_' + ms])

	df['ampl_intersec_' + ms] = np.maximum(0, df[['ls_' + ms, 'l_sup']].min(axis=1) - df[['li_' + ms, 'l_inf']].max(axis=1) )

	df['ic_' + ms] = ( df['ampl_intersec_' + ms] > 0 ) *1

	df['prct_intersec_' + ms] = 100* ( df['ampl_intersec_' + ms] / (df['l_sup'] - df['l_inf']) )   

	df.loc[df['tdo_' + ms].isna(), ['ampl_intersec_' + ms,'ic_' + ms,'prct_intersec_' + ms]] =  float('nan')
	return df


def ic_tdo(tv_c, z=1.96): 
	rangos_escala_pj = pd.DataFrame.from_dict( {'c':['C01',	'C02',	'C03',	'C04',	'C05',	'C06',	'C07',	'C08',	'C09',	'C10',	'C11',	'C12',	'C13',	'C14',	'C15',	'C16',	'C17',	'C18',	'C19'],
												'l_inf': [0,	0.0027,	0.0038,	0.0054,	0.0077,	0.0106,	0.0153,	0.0216,	0.0255,	0.0305,	0.0357,	0.0431,	0.0518,	0.0658,	0.0859,	0.1123,	0.17,	0.23,	0.5],
												'l_sup': [0.0027,	0.0038,	0.0054,	0.0077,	0.0106,	0.0153,	0.0216,	0.0255,	0.0305,	0.0357,	0.0431,	0.0518,	0.0658,	0.0859,	0.1123,	0.17,	0.23,	0.5,	1] } )
	df = rangos_escala_pj.merge(tv_c, how='outer')
	df[['tdo_tr','tdo_te','tdo_o']] = df[['tdo_tr','tdo_te','tdo_o']]/100

	df1 = intervalos(df, z, ms='tr')
	df2 = intervalos(df1, z, ms='te')
	df3 = intervalos(df2, z, ms='o')	

	return df3

def metricas_top_models_q_cv_rn_imp(base_x, base_y, base_x_tr_te, base_x_o, varis, ronda_i, num_top_models = 10, ronda_completa = False, var1= ["auc_dif", 0.05, True], var2 = ["ordena_te_q", 14, False], q = 20):
	"""
	Esta función devuelve el listado de métricas y modelos que se entrenan a partir de la función 
	fm.metricas para varios modelos. Sin necesidad de copiar y pegar los parametros uno a uno.
	
	Se obtienen los primeros n_top_models en base a su ordenamiento en test y su auc_dif.
	
	Parametros:
	- ronda_i: base de datos con los modelos de una iteración (ronda)
	- num_top_models: número de modelos a comparar
	- ordena_by: metrica para usar en el top
	- ascending: ordena de forma ascendente o descendente
	
	Resultado: Lista donde cada elemento es una tupla -> [(tv_0, model_0), ..., (tv_n, model_n)]
	"""
   
	if ronda_completa : 
		ronda_i_top = ronda_i.loc[(ronda_i[var1[0]] <= var1[1]) & (ronda_i[var2[0]] >= var2[1])]
	else: 
		ronda_i_top = ronda_i.loc[ronda_i['ordena_te_q'].isin(ronda_i.sort_values('ordena_te_q', ascending = False).head(50)['ordena_te_q'].unique())]
 
	ronda_i_top = ronda_i_top.sort_values([var1[0], var2[0]], ascending=[var1[2], var2[2]])
	ronda_i_top['rn_var1'] = tuple(zip(ronda_i_top[var1[0]], ronda_i_top[var2[0]]))
	ronda_i_top['rn_var1'] = ronda_i_top[['rn_var1']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top = ronda_i_top.sort_values([var2[0], var1[0]], ascending=[var2[2], var1[2]])
	ronda_i_top['rn_var2']=tuple(zip(ronda_i_top[var2[0]], ronda_i_top[var1[0]]))
	ronda_i_top['rn_var2'] = ronda_i_top[['rn_var2']].apply(lambda x : pd.Series(pd.factorize(x)[0])).values + 1

	ronda_i_top['rn'] = ronda_i_top['rn_var2'] * ronda_i_top['rn_var1']

	ronda_i_top = ronda_i_top.sort_values(by="rn", ascending=True).head(num_top_models)

	# Bases de entrenamiento, prueba y oot con una feature aleatoria para top models lgbm
	#varis_with_random = varis.copy()
	#varis_with_random.append("random_feature")
	#base_x_with_random = base_x.copy()
	#base_x_with_random["random_feature"] = np.random.rand(len(base_x_with_random))
	#base_x_tr_with_random = base_x_tr.copy()
	#base_x_tr_with_random["random_feature"] = base_x_with_random["random_feature"]
	#base_x_te_with_random  = base_x_te.copy()
	#base_x_te_with_random["random_feature"] = np.random.rand(len(base_x_te_with_random))
	#base_x_o_with_random = base_x_o.copy()
	#base_x_o_with_random["random_feature"] = np.random.rand(len(base_x_o_with_random))
	
	# Inicializacion de vector modelos y vector impotancias
	modelos = []
	importance_top_models = np.zeros(len(varis))
	for index, row in ronda_i_top.iterrows():
		# Hiper-parametros del modelo row
		print("modelo_", index, sep = "")
		hyperparameters_row = eval(str(ronda_i_top.loc[index,:]['parameters']))
		# Tipo del modelo y creacion del mismo

		if str(ronda_i_top.loc[index,:]['modelo']) == 'HistGradientBoosting':
			tipo_modelo = "hgb"
			modelx = HistGradientBoostingClassifier( loss = hyperparameters_row['loss'],learning_rate = hyperparameters_row['learning_rate'],
					max_iter = hyperparameters_row['max_iter'],max_leaf_nodes = hyperparameters_row['max_leaf_nodes'],
					max_depth = hyperparameters_row['max_depth'],min_samples_leaf = hyperparameters_row['min_samples_leaf'],
					l2_regularization = hyperparameters_row['l2_regularization'],max_bins = hyperparameters_row['max_bins'],
					tol = hyperparameters_row['tol'],random_state = 42 )
		else:
			tipo_modelo = "lgbm"
			modelx= lgb.LGBMClassifier( learning_rate = hyperparameters_row['learning_rate'], max_bin = hyperparameters_row['max_bin'],
								n_estimators = hyperparameters_row['n_estimators'], num_leaves = hyperparameters_row['num_leaves'],
								max_depth = hyperparameters_row['max_depth'], min_data_in_leaf = hyperparameters_row['min_data_in_leaf'],
								n_jobs = -3, silent = True, seed = 42 )
		
		model_row = modelx.fit(base_x, base_y)
				
		# Creción del modelo y sus métricas
		print("modelo_", index, sep = "")

		tv_row = mapeo_escala_cv( model_row, base_x_tr_te, base_x_o, varis , escala_c = False, q_ = q)
		#tv_row = tv_row.fillna(0)
		
		# Se guarda la información de los modelos en una lista
		modelos.append((tv_row, model_row))
		# Cración de ranking por impórtancia
		if tipo_modelo == "lgbm":
			importance_top_models += model_row.feature_importances_
		print("\n")
	df_importance_top_models = pd.DataFrame({"feats" : varis, "all_top_importance" : importance_top_models})
	df_importance_top_models = df_importance_top_models.sort_values(by = "all_top_importance", ascending = False).reset_index(drop = True).reset_index()
	df_importance_top_models.rename(columns = {"index" : "rn_imp"}, inplace = True)
	df_importance_top_models["rn_imp"] = df_importance_top_models["rn_imp"] + 1

	# return modelos
	return df_importance_top_models
