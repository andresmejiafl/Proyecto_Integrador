def get_threshold(df):

    from pycaret.utils.generic import check_metric

    y_test = df['vo_oe_cartera_leasing']
    y_test_probs = df['prediction_score_1']

    f1_5_scores = []

    probability_thresholds = np.linspace(0, 1, num=100)

    for p in probability_thresholds:
        
        y_test_preds = []
        
        for prob in y_test_probs:
            if prob > p:
                y_test_preds.append(1)
            else:
                y_test_preds.append(0)
                
        f1_5 = fbeta_score(y_test, y_test_preds, beta=1.5)
        f1_5_scores.append(f1_5)

    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(probability_thresholds, f1_5_scores, label='f1.5')
    ax.set_xlabel('Probability Threshold')
    ax.set_ylabel('f-beta score')
    ax.legend(loc='center right')

    fbeta_max = max(f1_5_scores)
    index_fbeta = f1_5_scores.index(fbeta_max)
    th_optimo = probability_thresholds[index_fbeta]

    print("Valor maximo:"+str(fbeta_max))
    print("\n")
    print("Threshold Optimo:"+str(th_optimo))
    print("\n")

    y_test_threshold = []
    for prob in y_test_probs:
        if prob >= th_optimo:
            y_test_threshold.append(1)
        else:
            y_test_threshold.append(0)

    print(confusion_matrix(y_test,y_test_threshold))
    print("\n")

    print("F1_Score:" + str(f1_score(y_test, y_test_threshold)))
    print("F1_Score(1.5):" + str(fbeta_score(y_test, y_test_threshold,beta= 1.5)))
    print("Precision:" + str(precision_score(y_test, y_test_threshold)))
    print("Recall:" + str(recall_score(y_test, y_test_threshold)))
    print("Accuracy:" + str(accuracy_score(y_test, y_test_threshold)))
    print("ROC_AUC:" + str(roc_auc_score(y_test, y_test_threshold)))    



def optimizar_hiperparametros(df_oot):
    
    n_estimators     = [int(x) for x in np.linspace(1, 200, 10)]
    max_depth_0      = [int(x) for x in np.linspace(1, 100, 10)]
    min_samples_le_0 = [int(x) for x in np.linspace(1, 1000, 10)]  
    
    params_grid = list(product(n_estimators, max_depth_0, min_samples_le_0))
    
    max_depth_l = []
    min_samples_leaf_l = []
    n_estimators_l = []
    count = 0
    auc_tr = []; accuracy_tr = []; recall_tr = []
    precision_tr = []; f1_tr = []; fbeta_tr = []
    auc_te = []; accuracy_te = []; recall_te = []
    precision_te = []; f1_te = []; fbeta_te = []
    auc_val = []; accuracy_val = []; recall_val = []
    precision_val = []; f1_val = []; fbeta_val = []

    for i, j, k in params_grid:
            
        print("Creando modelo # " + str(count) + ".........")

        aux = (create_model('et',
                            return_train_score = True, 
                            verbose = False,
                            n_estimators = i,
                            max_depth = j,
                            min_samples_leaf = k))

        n_estimators_l.append(i)
        max_depth_l.append(j)
        min_samples_leaf_l.append(k)

        accuracy_tr.append(pull()['Accuracy']['CV-Train']['Mean'])
        auc_tr.append(pull()['AUC']['CV-Train']['Mean'])
        recall_tr.append(pull()['Recall']['CV-Train']['Mean'])
        precision_tr.append(pull()['Prec.']['CV-Train']['Mean'])
        f1_tr.append(pull()['F1']['CV-Train']['Mean'])
        fbeta_tr.append(pull()['FBetaScore']['CV-Train']['Mean'])

        accuracy_te.append(pull()['Accuracy']['CV-Val']['Mean'])
        auc_te.append(pull()['AUC']['CV-Val']['Mean'])
        recall_te.append(pull()['Recall']['CV-Val']['Mean'])
        precision_te.append(pull()['Prec.']['CV-Val']['Mean'])
        f1_te.append(pull()['F1']['CV-Val']['Mean'])
        fbeta_te.append(pull()['FBetaScore']['CV-Val']['Mean'])

        aux_finalize = finalize_model(aux)

        predict_model(aux_finalize,
                     raw_score = True, 
                     data = df_oot)

        accuracy_val.append(pull()['Accuracy'][0])
        auc_val.append(pull()['AUC'][0])
        recall_val.append(pull()['Recall'][0])
        precision_val.append(pull()['Prec.'][0])
        f1_val.append(pull()['F1'][0])
        fbeta_val.append(pull()['FBetaScore'][0])

        count = count + 1

    resultados = {'max_depth': max_depth_l,
                  'min_sample_leaf': min_samples_leaf_l,
                  'n_estimators': n_estimators_l,
                  'Accuracy_tr': accuracy_tr, 
                  'AUC_tr': auc_tr, 
                  'Recall_tr': recall_tr, 
                  'Precision_tr': precision_tr, 
                  'F1_tr': f1_tr, 
                  'FBetaScore_tr': fbeta_tr, 
                  'Accuracy_te': accuracy_te, 
                  'AUC_te': auc_te, 
                  'Recall_te': recall_te, 
                  'Precision_te': precision_te, 
                  'F1_te': f1_te,
                  'FBetaScore_te': fbeta_te,
                  'Accuracy_val': accuracy_val, 
                  'AUC_val': auc_val, 
                  'Recall_val': recall_val, 
                  'Precision_val': precision_val, 
                  'F1_val': f1_val, 
                  'FBetaScore_val': fbeta_val
                }
    
    res_df = pd.DataFrame.from_dict(resultados)
            
    return res_df.sort_values(by='AUC_te', ascending=False)