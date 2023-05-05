# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:12:16 2023

@author: 31643
"""


print('full pipeline code triggered')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functions import *


from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import scipy as sp


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


import warnings 
import sys
import os
pd.set_option('mode.chained_assignment', None)


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses



# load in datasets
df_snijders, df_torres, df_triple, df_ovi, df_step_PP, df_step_ITT_clinical,df_step_ITT_outcome, df_step_ITT_lab, df_step_ITT_vitals, df_sant, df_confalo, df_fernan_x, df_fernan_y, df_fernan_demo, df_fernan_fine = import_data()



#%% clean datasets

X_torres, Y_torres = clean_torres(df_torres)
X_snij, Y_snij = clean_snijders(df_snijders)
X_triple, Y_triple = clean_triple(df_triple)
X_ovi, Y_ovi = clean_ovi(df_ovi)
X_step, Y_step = clean_step(df_step_PP, df_step_ITT_clinical, df_step_ITT_outcome, df_step_ITT_lab, df_step_ITT_vitals)
X_sant, Y_sant = clean_sant(df_sant)
X_confalo, Y_confalo = clean_confalo(df_confalo)
X_fernan, Y_fernan = clean_fernan(df_fernan_x, df_fernan_y, df_fernan_demo, df_fernan_fine)



# add study label
Y_triple, Y_ovi, Y_step, Y_sant, Y_torres, Y_snij, Y_confalo, Y_fernan = add_study_label(Y_triple, Y_ovi, Y_step, Y_sant, Y_torres, Y_snij, Y_confalo, Y_fernan)


# merge
X = pd.concat([X_triple,X_ovi,X_step,X_sant,X_torres,X_snij, X_confalo, X_fernan]).reset_index(drop=True)
Y = pd.concat([Y_triple,Y_ovi,Y_step,Y_sant,Y_torres,Y_snij, Y_confalo, Y_fernan]).reset_index(drop=True)


# add Survival column
Y['Survival'] = np.abs(Y.Mort - 1)
Y['Mort'] = Y.Mort.astype(int)
Y['random'] = Y.random.astype(int)
Y['Survival'] = Y.Survival.astype(int)

# preprocess

# remove NaNs
X = remove_nan(X)
Y = remove_nan(Y)

print('treatment labels in observational set:')
print(Y[Y.STUDY==1].random)

# convert values with wring units
X = convert_units(X, Y)

# remove impossible values
X = remove_impossible_values(X)

assert X.shape[0] == Y.shape[0]
# add treatment variable to X table
X['random'] = Y.random


# # (optional) plot variable distributions
# plot_distributions(X, Y, plot_type='box')

# print(non_existing_variable)

#%%



# start saving print statemenyts
# original_stdout = sys.stdout # Save a reference to the original standard output
# sys.stdout = open('../overall_pipeline_results/' + save_name + ".txt", "w")


# SET HYPERPARAMS
save_directory = 'FINAL_2D/RIDGE_unpenalized_intercept_2D/'
# save_directory = 'FINALIST/treatment_var_interaction_lambda/FLEXIBLE_LASSO_TUNED/'
save_name = save_directory + '10_02_2023'

elastic_net_value = 0
# 0=Ridge, 1=LASSO

unpenalized = False
risk_modelling = False
remove_intercept = False

single_lambda = False


# metrics; etd, eqd, auc_benefit, abs_ite_lift, population_benefit
metric = 'abs_ite_lift'

if (risk_modelling | unpenalized):
    print('\n====== RISK MODELLING / UNPENALIZED REGRESSION !!! =========\n')
    assert single_lambda
    



penalty_dict = {
                0:'RIDGE',
                0.5:'ELASTICNET',
                1: 'LASSO'}

print('======= ' + penalty_dict[elastic_net_value] + ' ==========')







missingness_threshold = 0.33
imp = 'mice'
penalty='none'
C=1
B=2000

hyper_tuning=False



study_dict = {0:'Meijvis', 2:'Blum', 3:'Wittermans', 
              4:'Torres', 5:'Snijders', 6:'Confalonieri',
              7:'Fernandez', 8:'Meduri', 9:'Lloyd'}




# to be included in modelling
all_studies = [
    
    
    0, 
    2, 
    3, 
    4, 
    5,
    6,
    7,
    # 8
    ]

# to be included in LORO procedure
LORO_studies = [
    
    0, 
    2, 
    3, 
    4, 
    5,
    6,
    7,
    # 8,
    # 9
    ]


# a priori variable candidates
a_priori_selection = [
       
       
       'sex', 
       'age', 'rr', 'dbp', 
       'sbp',
        'temp','hr','spo2',
       'creat',
         'sodium',
           'urea', 
           'crp', 
       
        'glucose',
        'wbc',
       
       'random',
       
       # 
       
       
                      
                    #   'pco2', 'po2',
    #    'ph', 
       
    #    'ht', 'trombo', 'hb', 
    
    # 'albumin',
       
       
        #  'asat', 'alat', 'ld', 'bili', 
       # 'tnfa', 'il_1ra', 'mcp',
       # 'il_6', 'il_8', 'il_10', 'procal', 'trop', 'cort', 
       # 'psi', 
    #    'potassium'
       ]


Y_train_stacked = pd.DataFrame()
Y_test_stacked = pd.DataFrame()

plot_df = pd.DataFrame()

summary_df = pd.DataFrame()
left_out_studies = []
included_main_effects_per_study = []
included_interactions_per_study = []


df_weights_full = pd.DataFrame()
df_weights_full_sklearn = pd.DataFrame()


# overall print statements
print('====== Start running job name: ' + save_name + ' =========')

print('Using missingness threshold: ', missingness_threshold)
print('Bootstrap samples for forward selection: ', B)
print('Used metric: ', metric)

if single_lambda:
    df_results_total = pd.DataFrame()



# ============ START OUTER LORO LOOP ===================
# ======================================================

for left_out_study in LORO_studies:
    
    print('\n====== left out study OUTER LORO LOOP: ' + study_dict[left_out_study])
    
    # keep track
    left_out_studies.append(study_dict[left_out_study])
    
    
    include = [i for i in all_studies if i not in [left_out_study]]
    
    # isolate train set
    mask = Y.STUDY.isin(include)
    X_train = X[mask].reset_index(drop=True)
    Y_train = Y[mask].reset_index(drop=True)
    
    # isolate test set
    mask = Y.STUDY == left_out_study
    X_test = X[mask].reset_index(drop=True)
    Y_test = Y[mask].reset_index(drop=True)

    # isolate observational dataset
    mask = Y.STUDY == 1
    X_obs = X[mask].reset_index(drop=True)
    
    
    # stack train set and observational set
    X_combined = pd.concat([X_train, X_obs], axis=0)
    
    
    print('Size train data:' + ' (' + str(Y_train.STUDY.unique().shape[0]) + ' studies)')
    print(X_train.shape)
    
    print('Size left-out data:' + ' (' + str(Y_test.STUDY.unique().shape[0]) + ' studies)')
    print(X_test.shape)
    
    
    
    # STEP 1
    print('\nOUTER LORO LOOP: ' + study_dict[left_out_study] + '\n====== STEP 1: SELECT MAIN EFFECTS =========')
    

    if left_out_study in [8, 9]:
        selected_variables = a_priori_selection
    
    else:
        selected_variables = select_main_effects(
                                                a_priori_selection, 
                                                X_combined, 
                                                X_test, 
                                                missingness_threshold,
                                                )
        
    
    # select the variables with sufficient data in train and lef-out datasets
    X_train_selection = X_train[selected_variables]
    X_test_selection = X_test[selected_variables]
    X_obs_selection = X_obs[selected_variables]
    

    # keep track per left out study
    included_main_effects_per_study.append(selected_variables)
    

    #STEP 2
    print('\nOUTER LORO LOOP: ' + study_dict[left_out_study] + '\n====== STEP 2: TUNE LAMBDAS FOR FLEXIBLE PENALIZATION =========')


    if (risk_modelling | unpenalized):
        
        optimal_lambda = 0 #make unpenalized regression
    

    elif single_lambda:
        
        
        optimal_lambda, df_results = tune_single_lambda(X_train_selection, 
                                                        Y_train, 
                                                        X_obs_selection, 
                                                        imp, 
                                                        B, 
                                                        elastic_net_value,
                                                        study_dict[left_out_study],
                                                        metric,
                                                        save_directory,
                                                        missingness_threshold,
                                                        remove_intercept,
                                                        )
        
        df_results_total = pd.concat([df_results_total, df_results], axis=0)
        df_results_total.to_excel('../overall_pipeline_results/' + save_directory + 'df_lambdas.xlsx')

    

    else: 
        # optimal_pair = (0.4242151817220516, 0.01948297047624236)
        optimal_pair = tune_lambdas(
                                                        X_train_selection, 
                                                        Y_train, 
                                                        X_obs_selection, 
                                                        imp, 
                                                        B, 
                                                        elastic_net_value,
                                                        study_dict[left_out_study],
                                                        metric,
                                                        save_directory,
                                                        missingness_threshold,
                                                        )
        

        

    
    
    
    # STEP 3: FIT FLEXIBLE PENALIZATION S learner
    print('\nOUTER LORO LOOP: ' + study_dict[left_out_study] + '\n====== STEP 3: FIT Flexible penalization S learner with tuned lambdas =========')



    if left_out_study in [8, 9]:


        # FIRST IMPUTE
        print('\nFIRST IMPUTE')
        X_train_imp, _ = impute_only(X_train_selection, 
                                                        X_train_selection, 
                                                        X_obs_selection, 
                                                        imp
                                                        )
        
        assert X_train_imp.shape[0] + Y_train.shape[0]
        
        # reset index
        X_train_imp = X_train_imp.reset_index(drop=True)
        Y_train = Y_train.reset_index(drop=True)
        
        # NORMALIZE
        print('\nNORMALIZE')
        X_train_norm, _, _ = normalize_only(X_train_imp,
                                            X_train_imp,
                                            X_train_imp)

        #  ADD INTERACTIONS
        print('\nADD INTERACTIONS')
        for variable in selected_variables:
        # for variable in ['crp']:
            
            if variable == 'random':
                pass

            else:
                col = variable+'*random'
                # train set
                X_train_norm[col] = X_train_norm.random * X_train_norm[variable]

        n_features = len(selected_variables)

        # add intercepts
        X_train_norm.insert(0, 'intercept', np.ones(X_train_norm.shape[0]))
        
        model = sm.GLM(Y_train.Mort, X_train_norm, family=sm.families.Binomial())

        penalties = [optimal_pair[0]] + [optimal_pair[1]] * (2*n_features - 1)
        
        
        print(penalties)
        print(len(penalties))
        print(X_train_norm.shape)
        assert len(penalties) == X_train_norm.shape[1] 
        results = model.fit_regularized(alpha=penalties, L1_wt=elastic_net_value)

        print(len(penalties))
        print(X_train_norm.shape)
        
        # keep track of weights in each fold
        coefs = results.params
        print('MODEL WEIGHTS:')
        print(coefs)
        df_weight_plot = pd.DataFrame()
        df_weight_plot['weight'] = coefs
        df_weight_plot['var'] = X_train_norm.columns
        df_weight_plot['left_out_study'] = study_dict[left_out_study]
        df_weights_full = pd.concat([df_weights_full, df_weight_plot],axis=0)

        df_weights_full.to_csv('../overall_pipeline_results/' + save_directory+'df_weights_FULL.csv')

        plt.figure()
        sns.barplot(data=df_weights_full, x="var", y="weight", hue='left_out_study')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.legend(loc='upper center', 
                ncols=3
                )
        plt.savefig('../overall_pipeline_results/' + save_directory+'WEIGHTS_FULL.jpeg',dpi=300)


    else:

        # FIRST IMPUTE
        print('\nFIRST IMPUTE')
        X_train_imp, X_test_imp = impute_only(X_train_selection, 
                                                        X_test_selection, 
                                                        X_obs_selection, 
                                                        imp
                                                        )
        
        assert X_train_imp.shape[0] + Y_train.shape[0]
        assert X_test_imp.shape[0] + Y_test.shape[0]

        # reset index
        X_train_imp = X_train_imp.reset_index(drop=True)
        Y_train = Y_train.reset_index(drop=True)
        X_test_imp = X_test_imp.reset_index(drop=True)
        Y_test = Y_test.reset_index(drop=True)
        
        # CREATE TREATED/UNTREATED DATASETS 
        print('\nCREATE TREATED/UNTREATED DATASETS')
        X_test_treated = X_test_imp.copy()
        X_test_treated['random'] = 1

        X_test_untreated = X_test_imp.copy()
        X_test_untreated['random'] = 0
        

        # NORMALIZE
        print('\nNORMALIZE')
        X_train_norm, X_test_treated_norm, X_test_untreated_norm = normalize_only(X_train_imp,
                                                                                X_test_treated,
                                                                                X_test_untreated)
        
        
        if not risk_modelling:
            # ADD INTERACTIONS
            print('\nADD INTERACTIONS')
            for variable in selected_variables:
                
                if variable == 'random':
                    pass

                else:
                    col = variable+'*random'
                    # train set
                    X_train_norm[col] = X_train_norm.random * X_train_norm[variable]
                    
                    # test set under treatment
                    X_test_treated_norm[col] = X_test_treated_norm.random * X_test_treated_norm[variable]
                    
                    # test set under no treatment
                    X_test_untreated_norm[col] = X_test_untreated_norm.random * X_test_untreated_norm[variable]


        n_features = len(selected_variables)
        
        
        
            
        # fit a LASSO/RIDGE model with different penalization strengths
        if single_lambda:

            
            # add intercepts
            X_train_norm.insert(0, 'intercept', np.ones(X_train_norm.shape[0]))
            X_test_treated_norm.insert(0, 'intercept', np.ones(X_test_treated_norm.shape[0]))
            X_test_untreated_norm.insert(0, 'intercept', np.ones(X_test_untreated_norm.shape[0]))

            assert X_train_norm.shape[1] == X_test_treated_norm.shape[1] == X_test_untreated_norm.shape[1]
            

            
            
            model = sm.GLM(Y_train.Mort, X_train_norm, family=sm.families.Binomial())
            
            
            if risk_modelling:
                penalties = [0] + [optimal_lambda] * n_features

            else:
                penalties = [0] + [optimal_lambda] * (n_features*2 - 1)
            
            
            print(len(penalties))
            print(X_train_norm.shape)
            assert len(penalties) == X_train_norm.shape[1] == X_test_treated_norm.shape[1] == X_test_untreated_norm.shape[1]
            print('penalties:')
            print(penalties)

            results = model.fit_regularized(alpha=penalties, L1_wt=elastic_net_value)
            print('outer LOTO fit (so on 6 trials):')
            print(results.params)

            
        else:
            

            # add intercepts
            X_train_norm.insert(0, 'intercept', np.ones(X_train_norm.shape[0]))
            X_test_treated_norm.insert(0, 'intercept', np.ones(X_test_treated_norm.shape[0]))
            X_test_untreated_norm.insert(0, 'intercept', np.ones(X_test_untreated_norm.shape[0]))

            assert X_train_norm.shape[1] == X_test_treated_norm.shape[1] == X_test_untreated_norm.shape[1]

            model = sm.GLM(Y_train.Mort, X_train_norm, family=sm.families.Binomial())

            penalties = [0] + [optimal_pair[0]] * n_features + [optimal_pair[1]] * (n_features - 1)
            # penalties = [optimal_pair[0]] + [optimal_pair[1]] * (2*n_features - 1)
            
            
            print(penalties)
            print(len(penalties))
            print(X_train_norm.shape)
            assert len(penalties) == X_train_norm.shape[1] == X_test_treated_norm.shape[1] == X_test_untreated_norm.shape[1]
            results = model.fit_regularized(alpha=penalties, L1_wt=elastic_net_value)

            print(len(penalties))
            print(X_train_norm.shape)
        
        # keep track of weights in each fold
        coefs = results.params
        print('MODEL WEIGHTS:')
        print(coefs)
        df_weight_plot = pd.DataFrame()
        df_weight_plot['weight'] = coefs
        df_weight_plot['var'] = X_train_norm.columns
        df_weight_plot['left_out_study'] = study_dict[left_out_study]
        df_weights_full = pd.concat([df_weights_full, df_weight_plot],axis=0)

        df_weights_full.to_csv('../overall_pipeline_results/' + save_directory+'df_weights_FULL.csv')

        plt.figure()
        sns.barplot(data=df_weights_full, x="var", y="weight", hue='left_out_study')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.legend(loc='upper center', 
                ncols=3
                )
        plt.savefig('../overall_pipeline_results/' + save_directory+'WEIGHTS_FULL.jpeg',dpi=300)

        
        
        # calculate and add ITEs for test data
        y_pred_treated = results.predict(X_test_treated_norm)
        y_pred_untreated = results.predict(X_test_untreated_norm)

        ite = list(y_pred_untreated - y_pred_treated)
        Y_test['ite'] = ite
        
        #add to stacked Y dfs
        Y_test_stacked = pd.concat([Y_test_stacked, Y_test], axis=0)
        Y_test_stacked.to_csv('../overall_pipeline_results/' + save_directory + 'Y_test.csv')
        
        
        
        print('\nOverall c-for-benefit on left-out set:')
        score, _ = c_for_benefit(None, Y_test)
        print('overall c-for-benefit: ', score)
        
        # add to dataframe for histogram
        temp_df = pd.DataFrame()    
        temp_df['ITE'] = Y_test.ite 
        temp_df['left_out_study'] = study_dict[left_out_study] + ' (external)'+ ', c-for-benefit=' + str(score)
        plot_df = pd.concat([plot_df, temp_df], axis=0).reset_index(drop=True)
    

print('\n OUTER LORO DONE')


 
Y_test_stacked.to_csv('../overall_pipeline_results/' + save_directory + 'Y_test.csv')


print('\nOverall c-for-benefit on left out sets:')
score, _ = c_for_benefit(None, Y_test_stacked)
print(score)

etd = ETD(Y_test_stacked)
    
# add to dataframe for histogram
temp_df = pd.DataFrame()    
temp_df['ITE'] = Y_test_stacked.ite 
temp_df['left_out_study'] = 'TOTAL'+ ' (external)'+ ', c-for-benefit=' + str(score) + '\n ETD=' + str(etd)
plot_df = pd.concat([plot_df, temp_df], axis=0).reset_index(drop=True)    
    

# =========== plot Summary ITE distributions==============
sns.displot(plot_df, x="ITE", hue="left_out_study", kind="kde")
plt.ylim([0,5])
plt.xlim([-1,1])
plt.savefig('../overall_pipeline_results/'+save_name+'.jpeg',dpi=300)
    
    
# save summary df
summary_df['left_out_study'] = left_out_studies
summary_df['main_effects'] = included_main_effects_per_study
summary_df.to_excel('../overall_pipeline_results/'+save_name+'.xlsx')

# sys.stdout.close()
# sys.stdout=original_stdout


# plot weights overall

df_weights_full.to_csv('../overall_pipeline_results/' + save_directory+'df_weights_FULL.csv')

plt.figure()
sns.barplot(data=df_weights_full, x="var", y="weight", hue='left_out_study')
plt.xticks(rotation=90)
plt.tight_layout()
plt.legend(loc='upper center', 
        ncols=3
        )
plt.savefig('../overall_pipeline_results/' + save_directory+'WEIGHTS_FULL.jpeg',dpi=300)


plt.figure()
sns.barplot(data=df_weights_full, x="var", y="weight", hue='left_out_study')
plt.xticks(rotation=90)
plt.ylim(-1,1)
plt.tight_layout()
plt.legend(loc='upper center', 
        ncols=3
        )
plt.savefig('../overall_pipeline_results/' + save_directory+'WEIGHTS_FULL_ZOOM.jpeg',dpi=300)




directory = '../overall_pipeline_results/' + save_directory

study_name = 'overall'
outcome='Mort'
B=500
HTE_sub_plots_binary(Y_test_stacked, outcome, B, directory, study=study_name)

# plot total calibration figure
plot_ite_calibration(Y_test_stacked, "_Overall_Cali_plot", B, directory)

# plot per individual trial
for study in LORO_studies:
    
    print(study)

    # for specific trial
    Y_plot = Y_test_stacked[Y_test_stacked.STUDY==study]
    study_name = study_dict[study]

    if (Y_plot.ite.min() > 0) | (Y_plot.ite.max() < 0):
        print('only 1 group')
        print(Y_plot.ite)
    else:
        outcome='Mort'
        B=500
        HTE_sub_plots_binary(Y_plot, outcome, B, directory, study=study_name)

# plot ITE distribution
plt.figure()
plt.hist(Y_test_stacked.ite, bins=100)
plt.vlines(0, 0, 1000, linestyle='--', color='k')
plt.ylabel('count')
plt.xlabel('ITE')
plt.savefig('../overall_pipeline_results/' + save_directory + 'ite_distribution.jpg', dpi=300)

