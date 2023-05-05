# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:01:03 2022

@author: 777018
"""
import statsmodels.api as sm
import statsmodels.formula.api as smf
# from econml.dml import CausalForestDML
import pyreadstat
from numpy import linalg
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.utils import resample

# fit logit model for slope
families = [sm.families.Binomial(), sm.families.Poisson(), sm.families.Gamma(), sm.families.InverseGaussian()]





#%% Make HTE table per outcome


def import_data():
    
    df_snijders, _ = pyreadstat.read_sav("../datasets/Snijders_set.sav")
    df_snijders = pd.read_spss("../datasets/Snijders_set.sav")
    print('imported snijders', df_snijders.shape)
    
    # df_snijders_update = pd.read_spss("../datasets/Snijders_update.sav")
    
    # df_snijders_PP = pd.read_spss("../datasets/capisce data zonder ZIS naam.sav")
    
    # print('imported snijders update', df_snijders_update.shape)
    
    
    
    
    df_torres, _ = pyreadstat.read_sav("../datasets/Torres_set.sav")
    print('imported torres', df_torres.shape)
    
    df_triple_ovi = pd.read_excel('../data/Data TripleP + Ovidius EMC 18022022.xlsx')
    df_triple_ovi = df_triple_ovi.replace(r'^\s*$', np.nan, regex=True)
    
    # split tripleP and Ovidius
    df_triple = df_triple_ovi[df_triple_ovi.studie==0]
    print('imported triple P (Meijvis)', df_triple.shape)
    
    df_ovi = df_triple_ovi[df_triple_ovi.studie==1]
    print('imported Ovidius', df_ovi.shape)
    
    df_step_PP = pd.read_excel('../data/STEP data 18022022.xlsx')
    df_step_PP = df_step_PP.replace(r'^\s*$', np.nan, regex=True)
    print('imported STEP (Blum)', df_step_PP.shape)
    
    # import extra Blum files for extra ITT patients
    df_step_ITT_clinical = pd.read_excel('../datasets/STEP/STEP clinical and outcome ITT.xlsx',sheet_name='Clinical')
    df_step_ITT_outcome = pd.read_excel('../datasets/STEP/STEP clinical and outcome ITT.xlsx',sheet_name='Outcome')
    df_step_ITT_lab = pd.read_excel('../datasets/STEP/STEP Laboratory values all days ITT.xlsx')
    df_step_ITT_vitals = pd.read_excel('../datasets/STEP/STEP vital signs D1 ITT.xlsx')
    
    df_sant = pd.read_excel('../data/SanteonCAPdata 18022022.xlsx')
    df_sant = df_sant.replace(r'^\s*$', np.nan, regex=True)
    print('imported Santeon (Wittermans)', df_sant.shape)
    
    df_confalo = pd.read_excel('../datasets/confalo/SCAPLAST2_gluc.xlsx')
    df_confalo = df_confalo[~df_confalo['ID#'].isna()]
    print('imported Confalonieri', df_confalo.shape)
    
    df_fernan_x = pd.read_excel('../datasets/Fernandez/set1.xlsx')
    df_fernan_y = pd.read_excel('../datasets/Fernandez/set2.xlsx')
    df_fernan_demo = pd.read_excel('../datasets/Fernandez/set3.xlsx')
    df_fernan_fine = pd.read_excel('../datasets/Fernandez/fine.xlsx')
    
    print('imported Fernandez', df_fernan_y.shape)
    
                                                      
    return df_snijders, df_torres, df_triple, df_ovi, df_step_PP, df_step_ITT_clinical, df_step_ITT_outcome, df_step_ITT_lab, df_step_ITT_vitals, df_sant, df_confalo, df_fernan_x, df_fernan_y, df_fernan_demo, df_fernan_fine


def clean_fernan(df_fernan_x, df_fernan_y, df_fernan_demo, df_fernan_fine):
    
    
    
    df_fernan_x_adm = pd.DataFrame()
    age = []
    sex = []
    psi = []
    
    psi_ids = list(df_fernan_fine['Nºrand'].unique())
    
    for idx in df_fernan_x['Nºrand'].unique():
        
        # only keep earliest measurement
        snip = df_fernan_x[df_fernan_x['Nºrand']==idx]
        snip = snip[snip['Nºcontrol']==snip['Nºcontrol'].min()]
        df_fernan_x_adm = pd.concat([df_fernan_x_adm, snip],axis=0)
        
        # collect sex and age
        snip = df_fernan_demo[df_fernan_demo['Filiación_Nºrand']==idx]
        age.append(snip.Edad.max())    
        sex.append(snip.Codsex.max())
        
        if idx in psi_ids:
            psi.append(df_fernan_fine[df_fernan_fine['Nºrand'] == idx].Fine.values[0])
        else:
            psi.append(np.nan)
    
    df_fernan_x_adm['Sex'] = sex
    df_fernan_x_adm['Age'] = age
    df_fernan_x_adm['psi'] = psi
    
    # merge with outcome table
    df_fernan = df_fernan_y.merge(df_fernan_x_adm, how = 'inner', on = ['Nºrand'])
    
    
    
    
    
    
    keep_cols = [
        'Sex', 'Age', 
        
                 'TAs','TAd',
                 'FR','Tª','FC','SatO2',
                 'pCO2','pO2 *','pH', 
                 
                 'Creat','Na','K','Urea',
                 
                 
                 'Hto',
                 
                 
                 'Gluc', 'Nºleuc',
                 
                  
                  'IL-6v', 'IL-8v', 'IL-10v', 
                  'psi',
                 
                 
                 # ,'EvolMort28','EvolUCI'
                 
                 ]

    cols = [
            'sex','age',
            'sbp','dbp','rr','temp','hr','spo2',
            'pco2','po2','ph','creat','sodium','potassium','urea','ht',
            'glucose','wbc',
              'il_6','il_8','il_10',
              'psi']

    X = df_fernan[keep_cols]
    X.columns = cols
    
    Y = df_fernan[['Codmor','UCI','Random']]
    Y.columns = ['Mort','ICU','random']
    
    Y.loc[~Y.Mort.isna(), 'Mort'] = 1
    Y.loc[Y.Mort.isna(), 'Mort'] = 0
    
    
    print('Fernandez:')
    print(X.sex.sum(),'/',X.shape[0], ' Men')
    print(X.shape)
    print(Y.shape)
    
    return X,Y
    
    
def clean_torres(df_torres):
    
    keep_cols = [
        'Sexo', 'Edad', 
        # 'Fumador', 'Enolismo','Diabetes',
                 # 'Obesidad','CardIsq','Cancer',
                 
                 'FR','Temperatura','FC',
                 'GpCO201','GpO201','GpH01', 
                 
                 'BCreatinina','BSodio','BBUN','BPCR',
                 
                 
                 'HHematocrito','HPlaquetas',
                 'HHemoglobina','BAlbumina',
                 
                 'BGlucosa', 'HLeucocitos',
                 'BLDH','BBilirrubina',
                  
                  'IL61', 'IL81', 'IL101', 'PCT1', 'Cortisol1',
                  'Puntos',
                 
                 
                 # ,'EvolMort28','EvolUCI'
                 
                 ]

    cols = ['sex','age','rr','temp','hr',
            'pco2','po2','ph','creat','sodium','urea','crp','ht','trombo',
            'hb','albumin','glucose','wbc','ld','bili',
              'il_6','il_8','il_10','procal','cort','psi']

    X = df_torres[keep_cols]
    X.columns = cols
    
    Y = df_torres[['EvolMort28','UCI_Interm','EvolDiasEstanciaTotal','Grupo']]
    Y.columns = ['Mort','ICU','LOS','random']
    
    print('Torres:')
    print(X.sex.sum(),'/',X.shape[0], ' Men')
    print(X.shape)
    print(Y.shape)
    
    return X, Y


    
def clean_confalo(df_confalo):
    
    keep_cols = [
        'sex', 'age', 
        
                 'admission pH', 'admission PaCO2',
                 'Admission SaO2%',
                 'respiratory rate at admission',
                 'Reactive C-protein - 1st day mg/L',
                 'Temperature °C   admission',
                 'Blood pressure max/min admission',
                 'heart rate admission',
                 'sodium mEq/L admission',
                 'WBC  admission',
                 'creatinine serum  admission',
                 'potassium mEq/L admission',
                 '  bilirubin mg/dl  admission',
                 'platelets admission',
                 'Hb admission',
                 'ALT admission',
                 'Abumin  1st day',
                 'BUN admission', 
                 'Blood Glucose admission mg/dl',
                 'PSI score admission'
                 ]

    cols = ['sex','age',
            'ph', 'pco2', 'spo2', 'rr', 'crp', 'temp','bp', 'hr',
            'sodium', 'wbc', 'creat', 'potassium', 'bili', 'trombo',
            'hb', 'alat', 'albumin', 'urea',
            'glucose', 'psi']

    X = df_confalo[keep_cols]
    X.columns = cols
    
    # fix blood pressure variable
    X['sbp'] = X.bp.str.partition("/")[0]
    X['dbp'] = X.bp.str.partition("/")[2]
    X = X.drop(['bp'], axis=1)
    
    # fix heart rate variable
    X.loc[X.hr=='98(af)', 'hr'] = 98
    
    # fix bili variable
    X['bili'] = pd.to_numeric(X.bili, errors ='coerce').fillna(np.nan)
    
    # fix sex variable
    X.loc[X.sex=='f','sex'] = 0
    X.loc[X.sex=='m','sex'] = 1
    
    
    
    # make float types
    X = X.astype(float)
    
    
    Y = df_confalo[['in-hospital death','total lenght of hospital stay (days)','treatment group']]
    Y.columns = ['Mort','LOS','random']
    
    # import extra mortality info 
    df_mort = pd.read_excel('../datasets/confalo/SCPA study survival 3mo.xlsx', header=1)
    
    Y.loc[df_mort.iloc[:,4]<=30, 'Mort'] = 1
    Y.loc[~(df_mort.iloc[:,4]<=30), 'Mort'] = 0
    
    
    # fix treatment variable
    Y.loc[Y.random=='Hydrocortisone', 'random'] = 1
    Y.loc[Y.random=='Placebo', 'random'] = 0
    
    print('Confalonieri:')
    print(X.sex.sum(),'/',X.shape[0], ' Men')
    print(X.shape)
    print(Y.shape)
    
    return X, Y
    
def clean_snijders(df_snijders):
    
    keep_cols = [
        'geslacht', 'leeftijd', 

                 
                  'af1',
                 'diastbp','sysbp','temp1','hr1','sat',
                  'po2','pco1','ph1',
                 
                 'kreat','natrium','ureum',
                 'Ht','trombo1',
                 
                  'Hb',
                 'crp1','alb1','Gluc','leuco1',
                 # 'ASAT_0', 'ALAT_0', 'LD_0', 'bili_0',
                 
                 # 'TNFa_0', 'IL_1ra_0','MCP_0','IL6_0','IL8_0','IL10_0',
                 'PCT1', 
                 # 'TroponinT_0', 'cortisol_0',
                 'psiscore',
                 
                 
                 ]

    cols = ['sex','age','rr','dbp','sbp','temp','hr','spo2',
            'po2','pco2','ph','creat','sodium','urea','ht','trombo',
            'hb','crp','albumin','glucose','wbc',
              # 'asat','alat','ld','bili',
              # 'tnfa', 'il_1ra','mcp','il_6','il_8','il_10',
              'procal',
              # 'trop', 'cort',
              'psi'
              ]


    X = df_snijders[keep_cols]
    X.columns = cols

    Y = df_snijders[['mort30',
                     # 'icu','los',
                     'Studiemed']]
    Y.columns = ['Mort',
                 # 'ICU','LOS',
                 'random']
    
    Y['random'] = Y.random.astype(str)
    
    # fix error in ICU label
    # Y.loc[Y['ICU']==2,'ICU'] = 0
    
    # reverse label for mortality
    Y.loc[:,'Mort'] = np.abs(Y.Mort - 1)
    
    Y.loc[Y.random=='placebo','random'] = 0
    Y.loc[Y.random=='prednisolon','random'] = 1
    
    # switch male/female label (M=1, F=0)
    X['sex'] = X.sex.astype(str)
    X.loc[X.sex=='Man','sex'] = 1
    X.loc[X.sex=='vrouw','sex'] = 0
    
    print('Snijders:')
    print(X.sex.sum(),'/',X.shape[0], ' Men')
    print(X.shape)
    print(Y.shape)
    
    return X, Y
    
    
    
    
    
def clean_triple(df_triple):
    cols_triple_ovi =  [
              'Gender','Age',
              
              'Respiratoryrate_0','BPdiastolic','BPsystolic','Temperature_0','Heartrate','saturation_0',
               'PCO2_0','PO2_0','pH_0',
              
              'Creat_0','Sodium_0','Urea_0',
              'Ht_0','trombo_0',
              
               'Hb_0',
              'CRP_0','albumine_0','Glucose_0','Leukocytes_0',
              'ASAT_0', 'ALAT_0', 'LD_0', 'bili_0',
              
              'TNFa_0', 'IL_1ra_0','MCP_0','IL6_0','IL8_0','IL10_0',
              'Procalcitonin_0', 'TroponinT_0', 'cortisol_0',
              'PSI_score'
              
              # 'Diabetesmellitus','CongestiveHF_medicalhistory','CVA_medicalhistory',
              # 'alcohol', 
              # 'Renaldisease_medicalhistory','AB_athome',
              # 'xray_pleuraleffusion','COPD','Liverdisease_medicalhistory','PSI_score','Altered_mental_status',
              # 'Current_smoker','Malignancy_medicalhistory'
              
              
              ]
    
    cols = ['sex','age','rr','dbp','sbp','temp','hr','spo2',
            'pco2','po2','ph','creat','sodium','urea','ht','trombo',
            'hb','crp','albumin','glucose','wbc',
              'asat','alat','ld','bili',
              'tnfa', 'il_1ra','mcp','il_6','il_8','il_10','procal','trop', 'cort',
              'psi']


    X = df_triple[cols_triple_ovi]
    X.columns = cols
    
    Y = df_triple[['Mortality30','ICU','Lengthofstay','random']]
    Y.columns = ['Mort','ICU','LOS','random']
    
    # switch male/female label (M=1, F=0)
    X['sex'] = np.abs(X.sex - 1)
    
    print('Meijvis:')
    print(X.sex.sum(),'/',X.shape[0], ' Men')
    print(X.shape)
    print(Y.shape)
    return X, Y
    
    
def clean_ovi(df_ovi):
    
    cols_triple_ovi =  [
              'Gender','Age',
              
              'Respiratoryrate_0','BPdiastolic','BPsystolic','Temperature_0','Heartrate','saturation_0',
               'PCO2_0','PO2_0','pH_0',
              
              'Creat_0','Sodium_0','Urea_0',
              'Ht_0','trombo_0',
              
               'Hb_0',
              'CRP_0','albumine_0','Glucose_0','Leukocytes_0',
              'ASAT_0', 'ALAT_0', 'LD_0', 'bili_0',
              
              'TNFa_0', 'IL_1ra_0','MCP_0','IL6_0','IL8_0','IL10_0',
              'Procalcitonin_0', 'TroponinT_0', 'cortisol_0',
              'PSI_score'
              
              # 'Diabetesmellitus','CongestiveHF_medicalhistory','CVA_medicalhistory',
              # 'alcohol', 
              # 'Renaldisease_medicalhistory','AB_athome',
              # 'xray_pleuraleffusion','COPD','Liverdisease_medicalhistory','PSI_score','Altered_mental_status',
              # 'Current_smoker','Malignancy_medicalhistory'
              
              
              ]
    
    cols = ['sex','age','rr','dbp','sbp','temp','hr','spo2',
            'pco2','po2','ph','creat','sodium','urea','ht','trombo',
            'hb','crp','albumin','glucose','wbc',
              'asat','alat','ld','bili',
              'tnfa', 'il_1ra','mcp','il_6','il_8','il_10','procal','trop', 'cort',
              'psi']



    X = df_ovi[cols_triple_ovi]
    X.columns = cols
    
    Y = df_ovi[['Mortality30','ICU','Lengthofstay','random']]
    Y.columns = ['Mort','ICU','LOS','random']
    
    # switch male/female label (M=1, F=0)
    X['sex'] = np.abs(X.sex - 1)
    
    print('Ovidius:')
    print(X.sex.sum(),'/',X.shape[0], ' Men')
    print(X.shape)
    print(Y.shape)
    
    return X, Y
    
    
def clean_step(df_step_PP, df_step_ITT_clinical, df_step_ITT_outcome, df_step_ITT_lab, df_step_ITT_vitals):
    
    cols_step =  [
              'Gender','Age',
              
              'Respiratoryrate_0','BPdiastolic','BPsystolic','Temperature_0','Heartrate','saturation_0',
               'PCO2_0','PO2_0','pH_0',
              
              'Creat_0','Sodium_0','Urea_0',
              'Ht_0','trombo_0',
              
               'Hb_0',
              'CRP_0','albumine_0','Glucose_0','Leukocytes_0',
              'ASAT_0', 'ALAT_0', 'LD_0', 'bili_0',
              
              'TNFa_0', 'IL_1ra_0','MCP_0','IL6_0','IL8_0','IL10_0',
              'Procalcitonin', 'TroponineT', 'cortisol_0',
              'PSI_score'
              
              # 'Diabetesmellitus','CongestiveHF_medicalhistory','CVA_medicalhistory',
              # 'alcohol', 
              # 'Renaldisease_medicalhistory','AB_athome',
              # 'xray_pleuraleffusion','COPD','Liverdisease_medicalhistory','PSI_score','Altered_mental_status',
              # 'Current_smoker','Malignancy_medicalhistory'
              
              
              ]

    cols = ['sex','age','rr','dbp','sbp','temp','hr','spo2',
            'pco2','po2','ph','creat','sodium','urea','ht','trombo',
            'hb','crp','albumin','glucose','wbc',
              'asat','alat','ld','bili',
              'tnfa', 'il_1ra','mcp','il_6','il_8','il_10','procal','trop', 'cort',
              'psi']
    
    X_PP = df_step_PP[cols_step]
    X_PP.columns = cols
    
    Y_PP = df_step_PP[['Mortality30','ICU','Lengthofstay','random']]
    Y_PP.columns = ['Mort','ICU','LOS','random']
    

    # add extra ITT patients
    
    # map gender labels to 0/1 in ITT set
    df_step_ITT_clinical.loc[df_step_ITT_clinical['pat_sex']=='M', 'pat_sex'] = 0
    df_step_ITT_clinical.loc[df_step_ITT_clinical['pat_sex']=='F', 'pat_sex'] = 1
    
    # map randomization to 0/1 in ITT set
    df_step_ITT_outcome.loc[df_step_ITT_outcome['Randomization A=Pred B=Placebo']=='A', 'Randomization A=Pred B=Placebo'] = 1
    df_step_ITT_outcome.loc[df_step_ITT_outcome['Randomization A=Pred B=Placebo']=='B', 'Randomization A=Pred B=Placebo'] = 0
    
    # make 30-day mort from mort and time-to-death data 
    df_step_ITT_outcome['Death30'] = df_step_ITT_outcome.Death
    df_step_ITT_outcome.loc[df_step_ITT_outcome['timetodeath, days']> 30, 'Death30'] = 0
    
    
    # make list of patient IDs of extra ITT patients
    ids = [x for x in list(df_step_ITT_outcome['STEP-ID'].unique()) if x not in list(df_step_PP['Studynumber'].unique())]
    
    big_list = []
    
    # loop over demographics
    for var in ['pat_sex', 'Age at study entry', 'PSI score']:
        l = []
        for idx in ids:
            l.append(df_step_ITT_clinical[df_step_ITT_clinical['STEP-ID']==idx][var].values[0])
        big_list.append(l)
    
    # loop over vitals
    for var in ['Temperature','systolic', 'diastolic', 'pulse', 'respiratory frequency', 'pulsoxymeter %']:
        l = []
        for idx in ids:
            l.append(df_step_ITT_vitals[df_step_ITT_vitals['STEP-ID']==idx][var].values[0])
        big_list.append(l)
        
    # loop over labs
    for var in ['Natrium', 'Kreatinin','Procalcitonin', 'CRP', 'Albumin', 'Leukocytes', 'Urea',
            'Glucose 07h30']:
        l = []
        for idx in ids:
            l.append(df_step_ITT_lab[df_step_ITT_lab['STEP-ID']==idx][var].values[0])
        big_list.append(l)
    
    X_ITT = pd.DataFrame(big_list).T
    X_ITT.columns = ['sex','age','psi','temp','sbp','dbp','hr','rr',
                     'spo2','sodium','creat','procal','crp',
                     'albumin','wbc','urea','glucose']
    
    
    
    big_list = []
    # loop over treatment and outcomes
    for var in ['Death30', 'ICU transfer (in-hospital)','LOS (d)', 'Randomization A=Pred B=Placebo']:
        l = []
        for idx in ids:
            l.append(df_step_ITT_outcome[df_step_ITT_outcome['STEP-ID']==idx][var].values[0])
        big_list.append(l)
    
    Y_ITT = pd.DataFrame(big_list).T
    Y_ITT.columns = ['Mort','ICU','LOS','random']
    
    X = pd.concat([X_PP,X_ITT],axis=0)
    Y = pd.concat([Y_PP,Y_ITT],axis=0)
    
    # switch male/female label (M=1, F=0)
    X['sex'] = np.abs(X.sex - 1)
    
    print('Blum:')
    print(X.sex.sum(),'/',X.shape[0], ' Men')
    print(X.shape)
    print(Y.shape)
    
    return X, Y

def clean_sant(df_sant):
    
    
    cols_sant =  [
              'Gender','Age',
              
              'Respiratory_rate','Diastolic_blood_pressure','Systolic_blood_pressure','Temperature','Heart_rate','Oxygen_saturation',
               'pCO2_kPa','po2_kpa','ph',
              
              'Creatinine','Sodium','Ureum',
              'Hematocrit','Thrombocyte_count',
              
               'Hemoglobin',
              'CRP','Albumin','glucose','Leukocyte_count',
              'Aspartate_Amino_Transferase', 'Alanine_Amino_Transferase', 
              
              'Tumor_necrosis_factor_alfa', 'Interleukin_1ra','Monocyte_chemoattractant_protein_1','Interleukin_6','Interleukin_8',
              'PSI_score'
              
              # 'History_of_Diabetes','History_congestive_heart_failure','History_of_cerebrovascular_disease',
              # # 'alcohol', 
              # 'RenalChronic','ABhome',
              # 'pleuralfluid','History_of_COPD','History_of_liver_disease','PSI_score','Altered_mental_status',
              # 'Smoking_status','History_of_neoplastic_disease'
              
              
              ]
    
    X = df_sant[cols_sant]
    
    cols = ['sex','age','rr','dbp','sbp','temp','hr','spo2',
            'pco2','po2','ph','creat','sodium','urea','ht','trombo',
            'hb','crp','albumin','glucose','wbc',
              'asat','alat',
              'tnfa', 'il_1ra','mcp','il_6','il_8',
              'psi']
    
    X.columns = cols
    
    Y = df_sant[['day30_mortality','Intensive_care_unit_admission','Length_of_stay','Treatment_allocation']]
    Y.columns = ['Mort','ICU','LOS','random']
    
    # switch male/female label (M=1, F=0)
    X['sex'] = np.abs(X.sex - 1)
    
    print('Wittermans:')
    print(X.sex.sum(),'/',X.shape[0], ' Men')
    print(X.shape)
    print(Y.shape)
    
    return X, Y




def add_study_label(Y_triple, Y_ovi, Y_step, Y_sant, Y_torres, Y_snij, Y_confalo, Y_fernan):
    Y_triple.loc[:,'STUDY'] = 0
    Y_ovi.loc[:,'STUDY'] = 1 
    Y_step.loc[:,'STUDY'] = 2
    Y_sant.loc[:,'STUDY'] = 3
    Y_torres.loc[:,'STUDY'] = 4
    Y_snij.loc[:,'STUDY'] = 5
    Y_confalo.loc[:,'STUDY'] = 6
    Y_fernan.loc[:,'STUDY'] = 7
    
    
    return Y_triple, Y_ovi, Y_step, Y_sant, Y_torres, Y_snij, Y_confalo, Y_fernan



def remove_nan(df):
    
    # remove '999' missing
    df = df.replace(888, np.nan)
    df = df.replace(999, np.nan)
    df = df.replace(9999, np.nan)
    
    return df

def convert_units(X, Y):
    
    pco2_CF = 0.133322
    
    BUN_to_urea = 0.357
    
    
    glucose_CF = 0.0555
    bili_CF = 17.1
    cort_CF = 27.59
    CRP_CF = 9.524
    hb_CF = 0.6206
    creat_CF = 	88.42
    
    CF_mcp = 10
    
    
    # convert MCP Meijvis and Wittermans to pg/mL
    # X.loc[(Y.STUDY==1),'mcp'] *= 10
    # X.loc[(Y.STUDY==2),'mcp'] *= 10
    
    # convert IL-8 
    # X.loc[(Y.STUDY==0),'il_8'] *= 0.1
    # X.loc[(Y.STUDY==1),'il_8'] *= 0.1
    # X.loc[(Y.STUDY==3),'il_8'] *= 0.1
    # X.loc[(Y.STUDY==4),'il_8'] *= 0.01
    
    
    
    # Ilterleukin - 6
    # X.loc[(Y.STUDY==4),'il_6'] *= 0.1 # pg/dL to pg/mL
    
    X.loc[(Y.STUDY==5),'pco2'] *= pco2_CF 
    
    X.loc[(Y.STUDY==5),'po2'] *= pco2_CF
    

    X.loc[(Y.STUDY==3),'ht'] *= 0.01
    
    
    
    
    # Hb units in Torres mixed, first convert by factor 10
    X.loc[((Y.STUDY==4) & (X.hb>50)),'hb'] *= 0.1
    X.loc[(Y.STUDY==4),'hb'] *= hb_CF
    
    # Albumin units in Torres mixed, first convert by factor 10
    X.loc[((Y.STUDY==4) & (X.albumin<10)),'albumin'] *= 10
    
    X.loc[(Y.STUDY==4),'pco2'] *= pco2_CF 
    X.loc[(Y.STUDY==4),'po2'] *= pco2_CF
    X.loc[(Y.STUDY==4),'glucose'] *= glucose_CF 
    X.loc[(Y.STUDY==4),'bili'] *= bili_CF
    X.loc[(Y.STUDY==4),'cort'] *= cort_CF
    X.loc[(Y.STUDY==4),'crp'] *= CRP_CF
    X.loc[(Y.STUDY==4),'creat'] *= creat_CF
    X.loc[(Y.STUDY==4),'ht'] *= 0.01 
    X.loc[(Y.STUDY==4),'urea'] *= BUN_to_urea 
 
    
    # confalonieri
    
    X.loc[(Y.STUDY==6),'wbc'] *= 0.001
    X.loc[(Y.STUDY==6),'trombo'] *= 0.001
    X.loc[(Y.STUDY==6),'urea'] /= 2.1428
    X.loc[(Y.STUDY==6),'urea'] *= 0.357
    
    X.loc[(Y.STUDY==6),'creat'] *= creat_CF
    X.loc[(Y.STUDY==6),'pco2'] *= pco2_CF
    X.loc[(Y.STUDY==6),'albumin'] *= 10
    X.loc[(Y.STUDY==6),'hb'] *= hb_CF
    X.loc[(Y.STUDY==6),'bili'] *= bili_CF
    X.loc[(Y.STUDY==6),'glucose'] *= (1/18.0182)
    
    
    # Fernandez
    X.loc[(Y.STUDY==7),'pco2'] *= pco2_CF
    X.loc[(Y.STUDY==7),'po2'] *= pco2_CF
    X.loc[(Y.STUDY==7),'wbc'] *= 0.001
    
    # Hematocrit in fraction
    mask = X.ht>1
    X.loc[mask,'ht'] = np.nan
    
    return X


def remove_impossible_values(X):
    
    lower_bounds = {'age':0, 'rr':0, 'dbp':0, 'sbp':0, 'temp':30, 'hr':0, 'spo2':0, 'pco2':0, 'po2':0,
            'ph':6, 'creat':0, 'sodium':50, 'urea':0, 'ht':0, 'trombo':0, 'hb':0, 'crp':0, 'albumin':0,
            'glucose':0, 'wbc':0, 'asat':0, 'alat':0, 'ld':0, 'bili':0, 'tnfa':0, 'il_1ra':0, 'mcp':0,
            'il_6':0, 'il_8':0, 'il_10':0, 'procal':0, 'trop':0, 'cort':0, 'potassium':0, 'psi':0}
    
    upper_bounds = {'age':120, 'rr':200, 'dbp':300, 'sbp':300, 'temp':50, 'hr':300, 'spo2':100, 'pco2':20, 'po2':np.inf,
            'ph':8, 'creat':np.inf, 'sodium':np.inf, 'urea':300, 'ht':1, 'trombo':np.inf, 'hb':np.inf, 'crp':np.inf, 'albumin':np.inf,
            'glucose':np.inf, 'wbc':np.inf, 'asat':np.inf, 'alat':np.inf, 'ld':np.inf, 'bili':np.inf, 'tnfa':np.inf, 'il_1ra':np.inf, 'mcp':np.inf,
            'il_6':np.inf, 'il_8':np.inf, 'il_10':np.inf, 'procal':np.inf, 'trop':np.inf, 'cort':np.inf, 'potassium':np.inf,
            'psi':300}
    
    
    for col in X.columns:
        if col == 'sex':
            pass
        else:
            X.loc[X[col] < lower_bounds[col], col] = np.nan
            X.loc[X[col] > upper_bounds[col], col] = np.nan

    return X

def remove_outliers(X):
    
    for v in X.columns:
        m = X[v].mean()
        s = X[v].std()
        mask = X[v] > m+5*s
        X.loc[mask,v] = np.nan

    return X


def show_missingness(X):
    
    print('Missingness:')    
    print(X.isna().sum()/X.shape[0])
    
def plot_distributions(X, Y, plot_type='box'):
    
    
    
    var_dict = { 'age': 'Age', 'rr':'Resp. rate', 'dbp':'Dias. blood pressure',
                'sbp':'Syst. blood pressure', 'temp':'Temperature', 'hr':'Heart rate',
                'spo2': u'SaO\u2082', 'pco2':u'PaCO\u2082', 'po2':u'PaO\u2082',
            'ph':'pH', 'creat':'Creatinine', 'sodium':'Sodium', 'urea':'Urea',
            'ht':'Haematocrit', 'trombo':'Thrombocyte count', 'hb':'Haemoglobin',
            'crp':'CRP', 'albumin':'Albumin',
            'glucose': 'Glucose', 'wbc':'WBC count', 'asat':'ASAT',
            'alat':'ALAT', 'ld':'LD', 'bili':'Bilirubin', 'tnfa':'TNF-α', 
            'il_1ra':'IL-1 ra', 'mcp':'MCP',
            'il_6':'IL-6', 'il_8':'IL-8', 'il_10':'IL-10',
            'procal':'Procalcitonin', 'cort':'Cortisol', 'trop': 'Troponine', 'potassium': 'K', 'psi':'PSI'}

    unit_dict = { 'age': '[years]', 'rr':'[breaths/min]', 'dbp':'[mmHg]',
                'sbp':'[mmHg]', 'temp':u'[\N{DEGREE SIGN}C]', 'hr':'[bpm]',
                'spo2': '[%]', 'pco2':'[kPa]', 'po2':'[kPa]',
            'ph':'', 'creat':'[µmol/L]', 'sodium':'[mmol/L]', 'urea':'[mmol/L]',
            'ht':'[fraction]', 'trombo':'[10^9 cells/L]', 'hb':'[mmol/L]',
            'crp':'[mg/L]', 'albumin':'[g/L]',
            'glucose': '[mmol/L]', 'wbc':'[10^9 cells/L]', 'asat':'[U/L]',
            'alat':'[U/L]', 'ld':'', 'bili':'[µmol/L]', 'tnfa':'[??]', 
            'il_1ra':'[??]', 'mcp':'[??]',
            'il_6':'[??]', 'il_8':'[??]', 'il_10':'[??]',
            'procal':'', 'cort':'', 'trop': '', 'potassium': '', 'psi':'[total score]'}
    
    
    try:
        X = X.reset_index()
        Y = Y.reset_index()
    except:
        print('already reset indices')
    X_boxplot = pd.concat([X,Y],axis=1)
    
    # rename 'random' variable
    X_boxplot.loc[X_boxplot.random==0,'random'] = 'placebo'
    X_boxplot.loc[X_boxplot.random==1,'random'] = 'corticosteroids'
    
    #remove observational
    # X_boxplot = X_boxplot[X_boxplot.STUDY!=1]
    
    # change names
    
    study_dict = {0:'Meijvis', 1:'Endeman', 2:'Blum', 3:'Wittermans', 
                  4:'Torres', 5:'Snijders', 6:'Confalonieri', 7: 'Fernández-\nSerrano'}
    cols=[]
    for study in [6,5,0,1,7,2,4,3]:
        name = study_dict[study]+ '\n'+'(N=' +str(X[Y.STUDY==study].shape[0]) + ')'
        cols += [name]
        
        X_boxplot.loc[X_boxplot.STUDY==study,'STUDY'] = name
    
    
    log_cols = [
        # 'urea', 
                'asat', 'alat','tnfa', 'il_1ra', 
                'mcp',
           'il_6', 'il_8', 'il_10', 'procal', 'trop']
    
    # l = X.columns
    l = [
        'psi',
        'sex', 'age', 
          'rr', 'dbp', 'sbp', 'temp', 'hr', 'spo2', 'pco2', 'po2',
            'ph', 'creat', 'sodium', 'urea', 'ht', 'trombo', 'hb', 'crp', 'albumin',
            'glucose', 'wbc', 'asat', 'alat', 'ld', 'bili', 'tnfa', 'il_1ra', 'mcp',
            'il_6', 'il_8', 'il_10', 'procal', 'trop', 'cort', 'potassium'
           ]

    
    
    fontsize=7.5
    
    for v in l:
        print(v)
        if v == 'sex':
            pass
        else:
        
            
            
            # if log scale, then also a log plot
            
            if v in log_cols:
                plt.figure()
                X_boxplot[v+'_log'] = np.log10(X_boxplot[v])
                
                X_boxplot.replace([np.inf, -np.inf], np.nan, inplace=True)  
                
                if plot_type == 'box':
                    sns.boxplot(x="STUDY", y=v+'_log',
                            # hue="random",
                            palette="Blues",
                            # palette=["m", "g"],
                            order=cols,
                            data=X_boxplot)
                else:
                    sns.violinplot(x="STUDY", y=v+'_log',
                            # hue="random",
                            # palette=["m", "g"],
                            data=X_boxplot)
                
                y_label = var_dict[v] + ' ' + unit_dict[v]
                y_label = y_label + ' (log-scale)'
                
                plt.xlabel('')
                plt.ylabel(y_label)
                plt.xticks(fontsize = fontsize)
                plt.tight_layout()
                plt.savefig('../figures/variable_distributions/'+ plot_type+'/'+v+'_log.jpg',dpi=300)
                    
            
            # always non-log plot
            plt.figure()
            
            if plot_type == 'box':
                sns.boxplot(x="STUDY", y=v,
                        # hue="random",
                        palette="Blues",
                        # palette=["m", "g"],
                        order=cols,
                        data=X_boxplot)
            else:
                sns.violinplot(x="STUDY", y=v,
                        # hue="random",
                        # palette=["m", "g"],
                        data=X_boxplot)
                
                    
            y_label = var_dict[v] + ' ' + unit_dict[v]
            
            plt.ylabel(y_label)
            plt.xticks(fontsize = fontsize)
            plt.xlabel('')
            plt.tight_layout()
            plt.savefig('../figures/variable_distributions/'+plot_type+'/'+v+'.jpg',dpi=300)
            
            
            
            
                
                
                
def select_variables(X, threshold = 0.15):
    selection = list(X.columns[X.isna().sum()/X.shape[0] < threshold])
    X_selection = X[selection]
    
    return X_selection, selection    


def impute_only(X_train, 
                         X_test, 
                         X_obs, 
                         imp):
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.impute import KNNImputer
    import random
    
    
    # print(X_train.columns)
    # print(X_obs.columns)
    
    # save treatment variable to add in the end
    random_train = list(X_train.random)
    random_test = list(X_test.random)

    

    # drop treatment variable columns
    X_train = X_train.drop(['random'], axis=1)
    X_test = X_test.drop(['random'], axis=1)
    X_obs = X_obs.drop(['random'], axis=1)

    cols = X_train.columns

    # make combined dataset (train data and observational data)
    X_combined = pd.concat([X_train, X_obs], axis=0).reset_index(drop=True)
    
    # ===== normalize ========
    scaler = StandardScaler()
    # fit scaler
    fitted_scaler = scaler.fit(X_combined)
    
    # use fitted scaler to scale combined data (we need this for fitting imputer)
    X_combined_norm = pd.DataFrame(fitted_scaler.transform(X_combined))
    X_combined_norm.columns = cols
    
    # use fitted scaler to scale train data
    X_train_norm = pd.DataFrame(fitted_scaler.transform(X_train))
    X_train_norm.columns = cols
    
    # use fitted scaler to scale test data
    X_test_norm = pd.DataFrame(fitted_scaler.transform(X_test))
    X_test_norm.columns = cols
        
    # ==== impute =====
    
    if imp == 'mice':
        print('impute with Iterative Imputer')
        imputer = IterativeImputer(imputation_order='random',
                                    random_state=random.randint(0, 1000), 
                                    # random_state=0, 
                                    initial_strategy='median',
                                    # sample_posterior=True
                                    )
    
    elif imp == 'knn':
        print('impute with K Nearest Neighbour')
        imputer = KNNImputer(n_neighbors=5)
        
        

    
    # fit imputer
    fitted_imputer = imputer.fit(X_combined_norm)
    
    # use fitted imputer to impute train data
    X_train_imp = pd.DataFrame(fitted_imputer.transform(X_train_norm))
    X_train_imp.columns = cols
    
    # use fitted imputer to impute test data
    X_test_imp = pd.DataFrame(fitted_imputer.transform(X_test_norm))
    X_test_imp.columns = cols
    
    # use fitted scaler to scale datasets back to original representation
    X_train_inv = pd.DataFrame(scaler.inverse_transform(X_train_imp))
    X_train_inv.columns = cols
    X_test_inv = pd.DataFrame(scaler.inverse_transform(X_test_imp))
    X_test_inv.columns = cols


    # add treatment variable again
    X_train_inv['random'] = random_train
    X_test_inv['random'] = random_test 
    
    
    return X_train_inv, X_test_inv

def normalize_only(X_train, 
                         X_test_1,
                         X_test_2 
                         ):
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.impute import KNNImputer
    import random
    
    
    cols = X_train.columns
    
    # print(X_train.columns)
    # print(X_obs.columns)
    
    # ===== normalize ========
    scaler = StandardScaler()
    # fit scaler
    fitted_scaler = scaler.fit(X_train)
    
    
    # use fitted scaler to scale train data
    X_train_norm = pd.DataFrame(fitted_scaler.transform(X_train))
    X_train_norm.columns = cols
    
    # use fitted scaler to scale test data
    X_test_1_norm = pd.DataFrame(fitted_scaler.transform(X_test_1))
    X_test_1_norm.columns = cols
    X_test_2_norm = pd.DataFrame(fitted_scaler.transform(X_test_2))
    X_test_2_norm.columns = cols
        
    return X_train_norm, X_test_1_norm, X_test_2_norm

def normalize_and_impute(X_train, 
                         X_test, 
                         X_obs, 
                         imp):
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.impute import KNNImputer
    import random
    
    
    cols = X_train.columns
    
    # print(X_train.columns)
    # print(X_obs.columns)
    
    # make combined dataset (train data and observational data)
    X_combined = pd.concat([X_train, X_obs], axis=0).reset_index(drop=True)
    
    # ===== normalize ========
    scaler = StandardScaler()
    # fit scaler
    fitted_scaler = scaler.fit(X_combined)
    
    # use fitted scaler to scale combined data (we need this for fitting imputer)
    X_combined_norm = pd.DataFrame(fitted_scaler.transform(X_combined))
    X_combined_norm.columns = cols
    
    # use fitted scaler to scale train data
    X_train_norm = pd.DataFrame(fitted_scaler.transform(X_train))
    X_train_norm.columns = cols
    
    # use fitted scaler to scale test data
    X_test_norm = pd.DataFrame(fitted_scaler.transform(X_test))
    X_test_norm.columns = cols
        
    # ==== impute =====
    
    if imp == 'mice':
        print('impute with Iterative Imputer')
        imputer = IterativeImputer(imputation_order='random',
                                    random_state=random.randint(0, 1000), 
                                    # random_state=0, 
                                    initial_strategy='median',
                                    # sample_posterior=True
                                    )
    
    elif imp == 'knn':
        print('impute with K Nearest Neighbour')
        imputer = KNNImputer(n_neighbors=5)
        
        

    
    # fit imputer
    fitted_imputer = imputer.fit(X_combined_norm)
    
    # use fitted imputer to impute train data
    X_train_imp = pd.DataFrame(fitted_imputer.transform(X_train_norm))
    X_train_imp.columns = cols
    
    # use fitted imputer to impute test data
    X_test_imp = pd.DataFrame(fitted_imputer.transform(X_test_norm))
    X_test_imp.columns = cols
    
    # use fitted scaler to scale datasets back to original representation
    X_train_inv = pd.DataFrame(scaler.inverse_transform(X_train_imp))
    X_train_inv.columns = cols
    X_test_inv = pd.DataFrame(scaler.inverse_transform(X_test_imp))
    X_test_inv.columns = cols
    
    # fit new scaler, to make imputed datasets zero mean, unit variance
    scaler = StandardScaler()
    # fit scaler
    fitted_scaler = scaler.fit(X_train_inv)

    # use fitted scaler to scale train data
    X_train_return = pd.DataFrame(scaler.transform(X_train_inv))
    X_train_return.columns = cols
    X_test_return = pd.DataFrame(scaler.transform(X_test_inv))
    X_test_return.columns = cols
    

    return X_train_return, X_test_return



def save_pooled_data_for_interactions(X, Y, X_obs, studies_to_pool, left_out_study, selected_variables, missingness_threshold, imp, pooled_data_source):

    
    
    study_dict = {0:'Meijvis (N=304)', 1:'Endeman (N=201)', 2:'Blum (N=785)', 3:'Wittermans (N=401)', 
                  4:'Torres (N=120)', 5:'Snijders (N=213)', 6:'Confalonieri (N=46)',
                  7: 'Fernández-\nSerrano (N=56)'}
    
    
    # make list of effect modifiers to include as interaction terms in effect model
    effect_modifiers_to_include = []
    
    # normalize and impute
    X_imp, _ = normalize_and_impute(X, X, X_obs, imp)
        
    # X_imp, Y, _ = normalize_and_impute_zero_mean_per_study(X, Y, X, X_obs, imp)
    
    # merge X and Y
    data = pd.concat([X_imp,Y],axis=1)

        
    # initialize
    included_variables = []
    stat_df = pd.DataFrame()
    interactions = []
    p_values = []


    for interaction in selected_variables:
        print(interaction)
        
        
        # create list with studies that have enough data for this variable
        studies_with_enough_data = []
        
        for i in studies_to_pool:
            X_study = X[Y.STUDY==i]
            missingness = X_study[interaction].isna().sum()/X_study.shape[0]
            if missingness < missingness_threshold:
                studies_with_enough_data.append(i)
        
        # pool only RCTs without too much missingness
        pooled_data = data[data.STUDY.isin(studies_with_enough_data)]
        
        
        if len(studies_with_enough_data)>0: # if any study left with enough data for this variable
            
            included_variables.append(interaction) # save this variable as 'studied' interaction for end table
            
            # save pooled data for modelling in R
            
            pooled_data.to_csv(pooled_data_source + left_out_study + '_' + interaction+'.csv')
            
            

def find_interactions(X, Y, X_obs, studies_to_plot, studies_to_pool, 
                      left_out_study, selected_variables, 
                      missingness_threshold, imp, show_missingess_per_study, 
                      pooled_data_source, directory):

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import statsmodels.genmod.bayes_mixed_glm as smgb
    
    
    p_value_threshold = 0.05
    lower_x_limit = -5
    upper_x_limit = 5
    
    
    study_dict = {0:'Meijvis (N=304)', 1:'Endeman (N=201)', 2:'Blum (N=785)', 3:'Wittermans (N=401)', 
                  4:'Torres (N=120)', 5:'Snijders (N=213)', 6:'Confalonieri (N=46)',
                  7: 'Fernández-\nSerrano (N=56)'}
    
    
    # make list of effect modifiers to include as interaction terms in effect model
    effect_modifiers_to_include = []
    
    # normalize and impute
    X_imp, _ = normalize_and_impute(X, X, X_obs, imp)
        
    # X_imp, Y, _ = normalize_and_impute_zero_mean_per_study(X, Y, X, X_obs, imp)
    
    # merge X and Y
    data = pd.concat([X_imp,Y],axis=1)

        
    # initialize
    included_variables = []
    stat_df = pd.DataFrame()
    interactions = []
    p_values = []


    for interaction in selected_variables:
        print('========= ' + interaction + ' ==========')
        
        # initiatlize flags for 2 criteria to include an effect modofier as interaction term
        # criterion 1; P < 0.05 in test of pooled data
        criterion_1 = False
        
         
        # formula is set
        formula = 'Survival ~' + interaction + '*random' 
        
        
        # create list with studies that have enough data for this variable
        studies_with_enough_data = []
        
        for i in studies_to_pool:
            X_study = X[Y.STUDY==i]
            missingness = X_study[interaction].isna().sum()/X_study.shape[0]
            if missingness < missingness_threshold:
                studies_with_enough_data.append(i)
        
        # pool only RCTs without too much missingness
        raw_pooled_data = X[Y.STUDY.isin(studies_with_enough_data)]
        pooled_data = data[data.STUDY.isin(studies_with_enough_data)]
        
        
        if len(studies_with_enough_data)>0: # if any study left with enough data for this variable
            
            included_variables.append(interaction) # save this variable as 'studied' interaction for end table
            
            y_ticks = []
            
            # initialize forest plot
            plt.figure()
            
            # ==== plot pooled data, random intercepts and slopes model ============
            # y_loc = len(studies_to_plot)+2
            
            # model_data = pd.read_csv(pooled_data_source + left_out_study + '_' + interaction + '_random_intercept_slope.csv')
            
            # effect = model_data.iloc[3,0]
            # se = model_data.iloc[3,1]
            # p = model_data.iloc[3,3]
            # low = effect - (1.96*se)
            # high = effect + (1.96*se)
            
            # interactions.append(effect)
            # p_values.append(p)
            
            # plt.plot(effect, y_loc,'.',color='k', markersize=12)
            # plt.hlines(y_loc, xmin=low,xmax=high,linewidth=0.5,linestyle='-',color='k')    
            
            # # plot the P value
            # plt.text(effect+0.3, y_loc+0.15,'P='+str(np.round(p,4)), fontsize=8)
            
            # if p < p_value_threshold:
            #     criterion_1 = True
            #     plt.plot(effect+0.15, y_loc+0.15,"d",color='r', markersize=4)
            
            # #yticks
            # y_tick = 'Pooled (N='+str(pooled_data.shape[0]) + ')' + '\nrandom effects model'
            
            # if show_missingess_per_study:
            #     y_tick += ', missing: ' + str(int(raw_pooled_data[interaction].isna().sum()/raw_pooled_data.shape[0]*100))+'%'
            # y_ticks.append(y_tick)
            
            # ==== plot merged data, fixed effects model ============
            
            # define and fit model
            model = smf.glm(formula = formula, data=pooled_data, family=sm.families.Binomial())
            
            
            result = model.fit()
            
            effect = result.params[3]
            # print('effect:', effect)
            p = result.pvalues[3]
            
            
            # print('p:', p)
            low = result.conf_int()[0][3]
            high = result.conf_int()[1][3]
            
            interactions.append(effect)
            p_values.append(p)
            
            
            y_loc = len(studies_to_plot)+1
            
            plt.plot(effect, y_loc,'.',color='k', markersize=12)
            
            
            
            if low < lower_x_limit:
                print('check')
                plt.plot(lower_x_limit + 0.5, y_loc,'<',color='k', markersize=8)
                low = lower_x_limit + 0.5
                
            if high > upper_x_limit:
                
                plt.plot(upper_x_limit - 0.5, y_loc,'>',color='k', markersize=8)
                high = upper_x_limit - 0.5
                
            plt.hlines(y_loc, 
                       xmin=low,
                       xmax=high,
                       linewidth=0.5,
                       linestyle='-',
                       color='k')    
            
            # plot the P value
            plt.text(effect+0.3, y_loc+0.15,'P='+str(np.round(p,4)), fontsize=8)
            
            if p < p_value_threshold:
                plt.plot(effect+0.15, y_loc+0.15,"d",color='r', markersize=4)
            
            
            #yticks
            y_tick = 'Pooled (N='+str(pooled_data.shape[0]) + ')' + '\nfixed effects model'
            
            if show_missingess_per_study:
                y_tick += ', missing: ' + str(int(raw_pooled_data[interaction].isna().sum()/raw_pooled_data.shape[0]*100))+'%'
            y_ticks.append(y_tick)
                
            
            # loop for individual studies
            y_ticks.append('')
            
            low_values = []
            high_values = []
            
            # initialize list with all 'effects' (ie, the betas for the interaction terms)
            effects = []
            
            for i in range(len(studies_to_plot)):
                
                study_label = studies_to_plot[i]
                
                
                # print(study_dict[study_label])
                
                
                y_loc = len(studies_to_plot) - (i+1)
                
                # filter out specific study data
                raw_study_data = X[data.STUDY==study_label]
                study_data = data[data.STUDY==study_label]
                
                
                # add to y_ticks list
                y_tick = study_dict[study_label] 
                if show_missingess_per_study:
                    y_tick += (', missing: ' + str(int(raw_study_data[interaction].isna().sum()/raw_study_data.shape[0]*100)) + '%')
                    
                
                y_ticks.append(y_tick)
              
                    
                if study_label in studies_with_enough_data: # only if this study has enough data, calculate and plot the interaction
                
                    # define and fit model
                    model = smf.glm(formula = formula, data=study_data, family=sm.families.Binomial())
                    result = model.fit()
                    
                    assert result.nobs == study_data.shape[0], str(result.nobs) + ' vs ' + str(study_data.shape)
                    
                    effect = result.params[3]
                    
                    
                    
                    
                    p = result.pvalues[3]
                    low = result.conf_int()[0][3]
                    high = result.conf_int()[1][3]
                    
                    print(study_dict[study_label])
                    print(str(np.round(effect,1))+' (' + str(np.round(low,1)) + ' ; ' + str(np.round(high,1)) + ')')
                    
                    low_values.append(low)
                    high_values.append(high)
                    
                    # add to forest plot, only if missingness in RCT is not too high
                    plt.plot(effect, y_loc,'.',color='k', markersize=12)
                    
                    if low < lower_x_limit:
                        print('check')
                        plt.plot(lower_x_limit + 0.5, y_loc,'<',color='k', markersize=6)
                        low = lower_x_limit + 0.5
                        
                    if high > upper_x_limit:
                        
                        plt.plot(upper_x_limit - 0.5, y_loc,'>',color='k', markersize=6)
                        high = upper_x_limit - 0.5
                    
                    plt.hlines(y_loc, xmin=low, xmax=high, linewidth=0.5,linestyle='-',color='k')
                    
                    # plot the P value
                    if abs(effect) < 3:
                        plt.text(effect+0.3,y_loc+0.15,'P='+str(np.round(p,4)), fontsize=8)
                    
                    if p < p_value_threshold:
                        plt.plot(effect+0.15,y_loc+0.15,"d",color='r', markersize=4)
                        
                    
                    # if effect is very small and insignificant, make zero
                    if (effect < 0.05) & (p>0.95):
                        effect = 0
                    
                    effects.append(effect)
        
            
            # plt.xlim(min(low_values) - 0.1 , max(high_values) + 0.1)
            plt.xlim(lower_x_limit , upper_x_limit)
            plt.ylim(-1,len(studies_to_plot)+2.8)
            plt.vlines(0, -0.2, len(studies_to_plot)+2.1,linewidth=0.5,linestyle='--',color='k')
            
            
            plt.xlabel('Regression coeff. of treatment-interaction term')
            plt.yticks(list(np.arange(0,len(y_ticks),1))[::-1], y_ticks,
                fontsize=8.5
                )
            
            # plt.title(interaction+'*treatment' + ', left-out study: ' + left_out_study)
            plt.tight_layout()
            plt.savefig(directory + left_out_study+ '_'+ interaction+'.jpeg',dpi=300)
            
            
            # include effect modifier as interaction if both criterion holds
            if criterion_1:
                effect_modifiers_to_include.append(interaction)
            
        else:
            print('not enough data')

    
    return effect_modifiers_to_include

def add_risk_score(X_train_imp, Y_train, X_test_imp, Y_test):
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    outcome = 'Mort'


    clf = LogisticRegression(random_state=0).fit(X_train_imp, Y_train[outcome])

    Y_test['Mort_risk_LR'] = clf.predict_proba(X_test_imp)[:,1]
    print('AUC ICU risk model: ',roc_auc_score(Y_test[outcome], Y_test.Mort_risk_LR))

    
    Q_1 = np.percentile(Y_test.Mort_risk_LR,25)
    Q_2 = np.percentile(Y_test.Mort_risk_LR,50)
    Q_3 = np.percentile(Y_test.Mort_risk_LR,75)

    # add groups
    Y_test['group'] = 0
    Y_test.loc[Y_test.Mort_risk_LR>= Q_1,'group'] = 1
    Y_test.loc[Y_test.Mort_risk_LR>= Q_2,'group'] = 2
    Y_test.loc[Y_test.Mort_risk_LR>= Q_3,'group'] = 3


    return Y_train, Y_test


def fenotype_model(X_train_imp, Y_train, X_test_imp, Y_test, algo='GMM', n_clusters=2):
    
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    
    # fit model
    if algo == 'GMM':
        model = GaussianMixture(n_components=n_clusters, random_state=0).fit(X_train_imp)
    elif algo == 'K_means':
        model = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train_imp)
        
    # assign these to Y table
    Y_test['fenotype'] = model.predict(X_test_imp).astype(str)
    Y_train['fenotype'] = model.predict(X_train_imp).astype(str)

    # assign one-hot encoded fenotype labels if N>2
    # if n_clusters > 2:
    #     print('more than 2 categories, use one-hot-encoded labels')
    #     y_ohe = pd.get_dummies(Y_train.fenotype, prefix='feno')
    #     Y_train['feno_1'] = y_ohe.feno_0
    #     Y_train['feno_2'] = y_ohe.feno_1
    #     Y_train['feno_3'] = y_ohe.feno_2
        
    #     y_ohe = pd.get_dummies(Y_test.fenotype, prefix='feno')
    #     Y_test['feno_1'] = y_ohe.feno_0
    #     Y_test['feno_2'] = y_ohe.feno_1
    #     Y_test['feno_3'] = y_ohe.feno_2
        
    return Y_train, Y_test


def recognize_fenotype(X_train_imp, Y_train, X_test_imp, Y_test,):
    
    try:
        X_train_imp = X_train_imp.reset_index(drop=True)
        Y_train = Y_train.reset_index(drop=True)
        X_test_imp = X_test_imp.reset_index(drop=True)
        Y_test = Y_test.reset_index(drop=True)
    except:
        print('Y index already reset')
    
    # loop to find mean Urea for different groups
    groups = list(Y_train.fenotype.unique())
    urea = []
    for group in groups:
        urea.append(X_train_imp[Y_train.fenotype==group].urea.mean()) 
    
    # order based on mean urea
    sorted_idxs = sorted(range(len(urea)), key=lambda k: urea[k])
    
    if len(urea) == 2:
        
        fenotype_names = ['Hypo-inflammatory', 'hyper-inflammatory']
        
        for i in range(len(sorted_idxs)):
            Y_train.loc[Y_train.fenotype==groups[sorted_idxs[i]],'fenotype'] = fenotype_names[i]
            Y_test.loc[Y_test.fenotype==groups[sorted_idxs[i]],'fenotype'] = fenotype_names[i]
        
    
    if len(urea) == 3:
        
        fenotype_names = ['Hypo-inflammatory','Medium', 'hyper-inflammatory']
        
        for i in range(len(sorted_idxs)):
            Y_train.loc[Y_train.fenotype==groups[sorted_idxs[i]],'fenotype'] = fenotype_names[i]
            Y_test.loc[Y_test.fenotype==groups[sorted_idxs[i]],'fenotype'] = fenotype_names[i]
    
    return Y_train, Y_test




def standardized_plot(X, Y, selected_variables, directory):

    var_dict = { 'sex': 'Sex', 'age': 'Age', 'rr':'RR', 'dbp':'DBP',
                'sbp':'SBP', 'temp':'Temp.', 'hr':'Heart rate',
                'spo2': u'SaO\u2082', 'pco2':u'PaCO\u2082', 'po2':u'PaO\u2082',
            'ph':'pH', 'creat':'Creatinine', 'sodium':'Sodium', 'urea':'Urea',
            'ht':'Haematocrit', 'trombo':'Thrombocyte count', 'hb':'Haemoglobin',
            'crp':'CRP', 'albumin':'Albumin',
            'glucose': 'Glucose', 'wbc':'WBC', 'asat':'ASAT',
            'alat':'ALAT', 'ld':'LD', 'bili':'Bilirubin', 'tnfa':'TNF-α', 
            'il_1ra':'IL-1 ra', 'mcp':'MCP',
            'il_6':'IL-6', 'il_8':'IL-8', 'il_10':'IL-10',
            'procal':'Procalcitonin', 'cort':'Cortisol', 'trop': 'Troponine', 'potassium': 'K',
            'psi':'PSI',
            'random': 'corticosteriods'}

 
    
    try:
        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)
    except:
        print('Y index already reset')

    
    
    Y['Group'] = 1
    
    threshold = 0
    mask = Y.ite > threshold
    Y.loc[mask, 'Group'] = 2
    
    
    groups = [1,2]
    effect_modifier = 'Group'
       
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    s5 = []


    S = [s1,s2,s3,s4,s5]
    
    for v in selected_variables:
        # print(v)
        m = np.mean(X[v])
        s = np.std(X[v])
        
        for i in groups:
            print(Y[effect_modifier]==i)
            S[i].append((np.mean(X[Y[effect_modifier]==i][v]) - m)/s)
        
            

    fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

    df = pd.DataFrame()
    
    clean_variables = []
    for v in selected_variables:
        clean_variables.append(var_dict[v])
        
    df['v'] = clean_variables
    
    
    
    for i in groups:
     
        df[i] = S[i]
    
    df['diff'] = df[2] - df[1]
    
    # sort values
    df = df.sort_values(by='diff')

    # make plots
    colors = ['cornflowerblue', 'navy']
    labels = ['Predicted harm', 'Predicted benefit']
    
    for i in groups:
        plt.plot(df[i].values,
                 'o-',
                 label=labels[i-1],
                 color = colors[i-1])
    

    plt.xticks(np.arange(0, df.shape[0], 1.0), df.v.values)
    
    # ax.set_xticklabels(df.v.values)
    
    
    plt.ylabel('Standardized variable value')
    plt.xlabel('Variables')
    plt.xticks(rotation = 45, fontsize=8)
    plt.legend(loc='upper center', prop={'size': 9})
    
    
    plt.tight_layout()
    
    plt.savefig(directory+'/standardized_groups_plot.jpg',dpi=300)
    
    return df



def specific_subgroup(Y, X, outcome, B, directory, subgroup_variable):
    
    var_dict = { 'age': 'Age', 'rr':'Resp. rate', 'dbp':'Dias. blood pressure',
                'sbp':'Syst. blood pressure', 'temp':'Temperature', 'hr':'Heart rate',
                'spo2': u'SaO\u2082', 'pco2':u'PaCO\u2082', 'po2':u'PaO\u2082',
            'ph':'pH', 'creat':'Creatinine', 'sodium':'Sodium', 'urea':'Urea',
            'ht':'Haematocrit', 'trombo':'Thrombocyte count', 'hb':'Haemoglobin',
            'crp':'CRP', 'albumin':'Albumin',
            'glucose': 'Glucose', 'wbc':'WBC count', 'asat':'ASAT',
            'alat':'ALAT', 'ld':'LD', 'bili':'Bilirubin', 'tnfa':'TNF-α', 
            'il_1ra':'IL-1 ra', 'mcp':'MCP',
            'il_6':'IL-6', 'il_8':'IL-8', 'il_10':'IL-10',
            'procal':'Procalcitonin', 'cort':'Cortisol', 'trop': 'Troponine', 'potassium': 'K', 'psi':'PSI'}

    unit_dict = { 'age': 'years', 'rr':'[breaths/min]', 'dbp':'[mmHg]',
                'sbp':'[mmHg]', 'temp':u'[\N{DEGREE SIGN}C]', 'hr':'[bpm]',
                'spo2': '[%]', 'pco2':'[kPa]', 'po2':'[kPa]',
            'ph':'', 'creat':'[µmol/L]', 'sodium':'[mmol/L]', 'urea':'[mmol/L]',
            'ht':'[fraction]', 'trombo':'[10^9 cells/L]', 'hb':'[mmol/L]',
            'crp':'mg/L', 'albumin':'[g/L]',
            'glucose': '[mmol/L]', 'wbc':u'10\u2079 cells/L', 'asat':'[U/L]',
            'alat':'[U/L]', 'ld':'', 'bili':'[µmol/L]', 'tnfa':'[??]', 
            'il_1ra':'[??]', 'mcp':'[??]',
            'il_6':'[??]', 'il_8':'[??]', 'il_10':'[??]',
            'procal':'', 'cort':'', 'trop': '', 'potassium': '', 'psi':'[total score]'}
    
    
    # first drop NaNs
    mask = X[subgroup_variable].isna()
    X = X[~mask].reset_index(drop=True)
    Y = Y[~mask].reset_index(drop=True)
    
    effect_modifier = 'Quartile'
    
    Y[effect_modifier] = 1
    
    print(Y.shape)
    
    threshold = np.percentile(X[subgroup_variable], 25)
    print(threshold)
    print(X[X[subgroup_variable]==threshold].shape)
    mask = X[subgroup_variable] >= threshold

    
    print(mask.shape)
    Y.loc[mask, effect_modifier] = 2
    
    threshold = np.percentile(X[subgroup_variable], 50)
    print(threshold)
    print(X[X[subgroup_variable]==threshold].shape)
    mask = X[subgroup_variable] >= threshold
    Y.loc[mask, effect_modifier] = 3

    
    
    threshold = np.percentile(X[subgroup_variable], 75)
    print(threshold)
    print(X[X[subgroup_variable]==threshold].shape)
    mask = X[subgroup_variable] >= threshold
    Y.loc[mask, effect_modifier] = 4
    
    print(Y[effect_modifier].value_counts())

    groups = [1,2,3,4]
    
    
    

    # define looper (to loop over group labels which should reveal HTE)
    loop = np.arange(0,len(groups),1)
    
    
    df = pd.DataFrame()
    df['group'] = groups
    
    
    # number of patients in each group
    
    x_ticks = []
    for i in groups:
        n = Y[Y[effect_modifier]==i].shape[0]
        n_events = int(Y[Y[effect_modifier]==i][outcome].sum(0))
        
       
        
        x_tick = str(int(X[Y[effect_modifier]==i][subgroup_variable].dropna().min())) + '-' + str(int(X[Y[effect_modifier]==i][subgroup_variable].dropna().max())) + ' ' + unit_dict[subgroup_variable]
        x_tick += '\n (' + str(n_events) + '/' + str(n) +')' 
        x_ticks.append(x_tick)
    
        
    x_label = var_dict[subgroup_variable]  + ' quartile'
    
    
    
    # first calc overall/average effect
    control = Y[(Y.random==0)][outcome]
    experiment = Y[(Y.random==1)][outcome]
    EER = experiment.sum()/experiment.shape[0]
    CER = control.sum()/control.shape[0]

    # make a / b / c / d (n mort in treated / n survived in treated / n mort in untreated / n survived in untreated )
    a = experiment.sum()
    b = experiment.shape[0] - a 
    c = control.sum()
    d = control.shape[0] - c 
    
    overall_OR = ((a/b) / (c/d))
    
    print(EER)
    print(CER)
    
    overall_RR = EER/CER
    
    overall_ARR = np.round((CER - EER),3)
    overall_event_rate = Y[outcome].sum()/Y.shape[0]
    
    
    
    CERs_table = []
    CERs = []
    EERs_table = []
    EERs = []
    RRRs = []
    ARRs = []
    NNTs = []
    ORs = []
    RRs = []
    
    
    

    for i in groups:
        control = Y[(Y[effect_modifier]==i)&(Y.random==0)][outcome]
        experiment = Y[(Y[effect_modifier]==i)&(Y.random==1)][outcome]
        

        a = experiment.sum()
        b = experiment.shape[0] - a 
        c = control.sum()
        d = control.shape[0] - c 


        CER = control.sum()/control.shape[0]
        CERs.append(CER)
        CERs_table.append(str(control.sum()) + '/' + str(control.shape[0]) + ' (' + str(np.round(CER*100,1)) + '%)')
        EER = experiment.sum()/experiment.shape[0]
        EERs.append(EER)
        EERs_table.append(str(experiment.sum()) + '/' + str(experiment.shape[0]) + ' (' + str(np.round(EER*100,1)) + '%)')
        
        RRRs.append(np.round((CER - EER)/CER,3))
        ARRs.append(np.round((CER - EER),3))
        
        try:
            NNTs.append(int(np.round(1/(CER - EER),3)))
        except:
            NNTs.append(np.nan)
            
        if (EER == 0) | (CER ==0):
            ORs.append(np.nan)
            RRs.append(np.nan)
        else:
            ORs.append(((a/b) / (c/d)))
            RRs.append(EER/CER)


    # bootstrapping procedure
    CERs_boot = []
    EERs_boot = []
    RRRs_boot = []
    ARRs_boot = []
    ORs_boot = []
    RRs_boot = []

    for b in range(B):
        Y_boot = resample(Y, stratify=Y.random)
        
        CERs_sample = []
        EERs_sample = []
        RRRs_sample = []
        ARRs_sample = []
        ORs_sample = []
        RRs_sample = []
        
        for i in groups:
            
            control = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)][outcome]
            experiment = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)][outcome]
            

            a = experiment.sum()
            b = experiment.shape[0] - a 
            c = control.sum()
            d = control.shape[0] - c 

            CER = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)][outcome].sum()/Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)].shape[0]
            CERs_sample.append(CER)
            EER = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)][outcome].sum()/Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)].shape[0]
            EERs_sample.append(EER)
            RRRs_sample.append((CER - EER)/CER)
            ARRs_sample.append((CER - EER))
            ORs_sample.append(((a/b) / (c/d)))
            RRs_sample.append(EER/CER)
        
        CERs_boot.append(CERs_sample)
        EERs_boot.append(EERs_sample)
        RRRs_boot.append(RRRs_sample)
        ARRs_boot.append(ARRs_sample)
        ORs_boot.append(ORs_sample)
        RRs_boot.append(RRs_sample)


    CERs_boot = pd.DataFrame(CERs_boot)
    EERs_boot = pd.DataFrame(EERs_boot)
    RRRs_boot = pd.DataFrame(RRRs_boot)
    ARRs_boot = pd.DataFrame(ARRs_boot)
    ORs_boot = pd.DataFrame(ORs_boot)
    RRs_boot = pd.DataFrame(RRs_boot)


    CERs_low = []
    CERs_high = []
    EERs_low = []
    EERs_high = []
    RRRs_low = []
    RRRs_high = []
    ARRs_low = []
    ARRs_high = []
    ORs_low = []
    ORs_high = []
    RRs_low = []
    RRs_high = []


    for i in loop:
        CERs_low.append(np.percentile(CERs_boot.iloc[:,i],5))
        CERs_high.append(np.percentile(CERs_boot.iloc[:,i],95))
        
        EERs_low.append(np.percentile(EERs_boot.iloc[:,i],5))
        EERs_high.append(np.percentile(EERs_boot.iloc[:,i],95))
        
        RRRs_low.append(np.percentile(RRRs_boot.iloc[:,i],5))
        RRRs_high.append(np.percentile(RRRs_boot.iloc[:,i],95))
        
        ARRs_low.append(np.percentile(ARRs_boot.iloc[:,i],5))
        ARRs_high.append(np.percentile(ARRs_boot.iloc[:,i],95))
        
        ORs_low.append(np.percentile(ORs_boot.iloc[:,i],5))
        ORs_high.append(np.percentile(ORs_boot.iloc[:,i],95))
        
        RRs_low.append(np.percentile(RRs_boot.iloc[:,i],5))
        RRs_high.append(np.percentile(RRs_boot.iloc[:,i],95))


    
    df['CER'] = CERs_table
    df['EER'] = EERs_table

    RRRs_table = []
    ARRs_table = []
    ORs_table = []
    RRs_table = []
    
    for i in loop:
        RRRs_table.append(str(np.round(RRRs[i]*100,1)) + '% (' + str(np.round(RRRs_low[i]*100,1)) + ' to ' + str(np.round(RRRs_high[i]*100,1))+')')
        ARRs_table.append(str(np.round(ARRs[i]*100,1)) + '% (' + str(np.round(ARRs_low[i]*100,1)) + ' to ' + str(np.round(ARRs_high[i]*100,1))+')')
        ORs_table.append(str(np.round(ORs[i],1)) + '% (' + str(np.round(ORs_low[i],1)) + ';' + str(np.round(ORs_high[i],1))+')')    
        RRs_table.append(str(np.round(RRs[i],1)) + '% (' + str(np.round(RRs_low[i],1)) + ';' + str(np.round(RRs_high[i],1))+')')    
        
    df['RRR'] = RRRs_table
    df['ARR'] = ARRs_table
    df['OR'] = ORs_table
    df['RR'] = RRs_table
    df['NNT'] = NNTs
    
    df.index = groups
    
    
    
    # === prepare plots ======
    
    colors = ['tab:blue','tab:orange']
    x_shift = [-0.1,0.1]
    
    
    
    # =========== Event rates ====================
    plt.figure()

    for i in loop:
        if i == 0:
            
            plt.plot(i+x_shift[0],
                     CERs[i]*100,'o',color = colors[0],label='Placebo')
            
            plt.plot([i+x_shift[0],i+x_shift[0]],
                     [CERs_low[i]*100,CERs_high[i]*100],
                     color='k',linewidth=0.6)
            
            plt.plot(i+x_shift[1],
                     EERs[i]*100,'x',color = colors[1],label='Corticosteroid')
            
            plt.plot([i+x_shift[1],i+x_shift[1]],
                     [EERs_low[i]*100,EERs_high[i]*100],
                     color='k',linewidth=0.6)
        
            
        else:
            plt.plot(i+x_shift[0],CERs[i]*100,'o',color = colors[0])
            plt.plot([i+x_shift[0],i+x_shift[0]],
                     [CERs_low[i]*100,CERs_high[i]*100],
                     color='k',linewidth=0.6)
            plt.plot(i+x_shift[1],EERs[i]*100,'x',color = colors[1])
            plt.plot([i+x_shift[1],i+x_shift[1]],
                     [EERs_low[i]*100,EERs_high[i]*100],
                     color='k',linewidth=0.6)
        
    
    
    plt.hlines(y=overall_event_rate*100, xmin = 0.03, xmax = 2.98, linestyle='--', color='b', linewidth=1.5, 
              label='Overall')
    
    plt.xlabel(x_label)
    
    
    plt.ylabel('Mortality rate (%)')
    
    # plt.title('Event rates')
    plt.legend()
    
    plt.xticks(loop, x_ticks)
    plt.tight_layout()
    plt.savefig(directory+'/HTE_plot_' + subgroup_variable + '_event_rate' + '.jpg',dpi=300)

    # plt.title('HTE on '+outcome )#+ '\n P='+str(np.round(p,4)))


    # =========== Odds ratios ======================
    plt.figure()
    print('Odds ratios')
    for i in loop:
        print(ORs[i])
        plt.plot(i, ORs[i],'o',color = colors[0])
        plt.plot([i,i],[ORs_low[i], ORs_high[i]], color='k', linewidth=0.6)
        
            
    plt.axhline(y=1,linestyle='--',color='k',linewidth=0.6)        
    plt.xlabel(x_label)
    
    plt.ylabel('Odds ratio\n Benefit \u2190  \u2192 Harm')
    # plt.title('Odds ratios')
    plt.xlim(-0.1,3.1)
    plt.xticks(loop, x_ticks)
    
    plt.hlines(y=overall_OR, xmin = 0.03, xmax = 2.98, linestyle='--', color='b', linewidth=1.5, 
              label='Overall')
    
    # plt.title('HTE on '+outcome )#+ '\n P='+str(np.round(p,4)))
    plt.tight_layout()
    plt.savefig(directory+'/HTE_plot_' + subgroup_variable + '_OR' + '.jpg',dpi=300)
    
    # ====== absolute risk reduction ==========================
    plt.figure()

    for i in loop:

        plt.plot(i,ARRs[i]*100,'o',color = colors[0])
        plt.plot([i,i],[ARRs_low[i]*100,ARRs_high[i]*100],color='k',linewidth=0.6)


    # for i in loop:
    #     plt.axhline(y=ARRs[i]*100,linestyle='--',color=colors[0],linewidth=0.4)     


    plt.axhline(y=0,linestyle='--',color='k',linewidth=0.6)        

    plt.xlabel(x_label)
    plt.ylabel(u'Mortality reduction (%)\n Harm \u2190  \u2192 Benefit')
    # plt.title('Risk differences')
    plt.xlim(-0.1,3.1)
    plt.xticks(loop, x_ticks)
    
    plt.hlines(y=overall_ARR*100, xmin = 0.03, xmax = 2.98, linestyle='--', color='b', linewidth=1.5, 
              label='Overall')
        

    # plt.title('HTE on '+outcome )#+ '\n P='+str(np.round(p,4)))
    plt.tight_layout()
    plt.savefig(directory+'/HTE_plot_' + subgroup_variable + '_ARR' + '.jpg',dpi=300)
    
    return df


def HTE_table(Y, outcome, B, directory, effect_modifier='fenotype', specify_range=None):
    
    
    
    Y['Benefit tertile'] = 1

    threshold = 0
    mask = Y.ite > threshold
    Y.loc[mask, 'Benefit tertile'] = 2

    

    groups = [1,2]
    effect_modifier = 'Benefit tertile'
    
    

    # define looper (to loop over group labels which should reveal HTE)
    loop = np.arange(0,len(groups)+1,1)
    
    
    df = pd.DataFrame()
    df['group'] = ['overall'] + groups
    
    
    # number of patients in each group
    
    x_ticks = []
    for i in groups:
        n = Y[Y[effect_modifier]==i].shape[0]
        n_events = int(Y[Y[effect_modifier]==i][outcome].sum(0))
        x_tick = str(i) + '\n (' + str(n_events) + '/' + str(n) +')' 
        
        if specify_range:
            x_tick += ('\n ' + specify_range[i-1])
        
        
        x_ticks.append(x_tick)
    
    
    
    CERs_table = []
    CERs = []
    EERs_table = []
    EERs = []
    RRRs = []
    ARRs = []
    NNTs = []
    ORs = []
        
    # first calc overall/average effect
    control = Y[(Y.random==0)][outcome]
    experiment = Y[(Y.random==1)][outcome]
    EER = experiment.sum()/experiment.shape[0]
    CER = control.sum()/control.shape[0]
    
    print(EER)
    print(CER)
    
    overall_OR = (EER/(1-EER))/(CER/(1-CER))
    overall_ARR = np.round((CER - EER),3)
    overall_event_rate = Y[outcome].sum()/Y.shape[0]
    overall_RRR = np.round((CER - EER)/CER,3)
    overall_NNT = int(np.round(1/(CER - EER),3))
    
    print('overall RRR: ',overall_RRR)
    print('overall OR: ',overall_OR)
    print('overall ARR: ',overall_ARR)
    print('overall NNT: ',overall_NNT)
    
    # add overall event rates
    CERs_table.append(str(control.sum()) + '/' + str(control.shape[0]) + ' (' + str(np.round(CER*100,1)) + '%)')
    EERs_table.append(str(experiment.sum()) + '/' + str(experiment.shape[0]) + ' (' + str(np.round(EER*100,1)) + '%)')
    RRRs.append(np.round((CER - EER)/CER,3))
    ARRs.append(np.round((CER - EER),3))
    NNTs.append(int(np.round(1/(CER - EER),3)))
    ORs.append((EER/(1-EER))/(CER/(1-CER)))
    
    
    for i in groups:
        control = Y[(Y[effect_modifier]==i)&(Y.random==0)][outcome]
        experiment = Y[(Y[effect_modifier]==i)&(Y.random==1)][outcome]
        
        CER = control.sum()/control.shape[0]
        CERs.append(CER)
        CERs_table.append(str(control.sum()) + '/' + str(control.shape[0]) + ' (' + str(np.round(CER*100,1)) + '%)')
        EER = experiment.sum()/experiment.shape[0]
        EERs.append(EER)
        EERs_table.append(str(experiment.sum()) + '/' + str(experiment.shape[0]) + ' (' + str(np.round(EER*100,1)) + '%)')
        
        RRRs.append(np.round((CER - EER)/CER,3))
        ARRs.append(np.round((CER - EER),3))
        NNTs.append(int(np.round(1/(CER - EER),3)))
        ORs.append((EER/(1-EER))/(CER/(1-CER)))

    # bootstrapping procedure
    CERs_boot = []
    EERs_boot = []
    RRRs_boot = []
    ARRs_boot = []
    ORs_boot = []

    for b in range(B):
        Y_boot = resample(Y, stratify=Y.random)
        
        CERs_sample = []
        EERs_sample = []
        RRRs_sample = []
        ARRs_sample = []
        ORs_sample = []
        
        
        # add overall bootstrapped sample
        CER = Y_boot[(Y_boot.random==0)][outcome].sum()/Y_boot[(Y_boot.random==0)].shape[0]
        CERs_sample.append(CER)
        EER = Y_boot[(Y_boot.random==1)][outcome].sum()/Y_boot[(Y_boot.random==1)].shape[0]
        EERs_sample.append(EER)
        RRRs_sample.append((CER - EER)/CER)
        ARRs_sample.append((CER - EER))
        ORs_sample.append((EER/(1-EER))/(CER/(1-CER)))
                
        
        for i in groups:

            CER = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)][outcome].sum()/Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)].shape[0]
            CERs_sample.append(CER)
            EER = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)][outcome].sum()/Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)].shape[0]
            EERs_sample.append(EER)
            RRRs_sample.append((CER - EER)/CER)
            ARRs_sample.append((CER - EER))
            ORs_sample.append((EER/(1-EER))/(CER/(1-CER)))
        
        CERs_boot.append(CERs_sample)
        EERs_boot.append(EERs_sample)
        RRRs_boot.append(RRRs_sample)
        ARRs_boot.append(ARRs_sample)
        ORs_boot.append(ORs_sample)


    CERs_boot = pd.DataFrame(CERs_boot)
    EERs_boot = pd.DataFrame(EERs_boot)
    RRRs_boot = pd.DataFrame(RRRs_boot)
    ARRs_boot = pd.DataFrame(ARRs_boot)
    ORs_boot = pd.DataFrame(ORs_boot)


    CERs_low = []
    CERs_high = []
    EERs_low = []
    EERs_high = []
    RRRs_low = []
    RRRs_high = []
    ARRs_low = []
    ARRs_high = []
    ORs_low = []
    ORs_high = []

    for i in loop:
        CERs_low.append(np.percentile(CERs_boot.iloc[:,i],5))
        CERs_high.append(np.percentile(CERs_boot.iloc[:,i],95))
        EERs_low.append(np.percentile(EERs_boot.iloc[:,i],5))
        EERs_high.append(np.percentile(EERs_boot.iloc[:,i],95))
        RRRs_low.append(np.percentile(RRRs_boot.iloc[:,i],5))
        RRRs_high.append(np.percentile(RRRs_boot.iloc[:,i],95))
        ARRs_low.append(np.percentile(ARRs_boot.iloc[:,i],5))
        ARRs_high.append(np.percentile(ARRs_boot.iloc[:,i],95))
        ORs_low.append(np.percentile(ORs_boot.iloc[:,i],5))
        ORs_high.append(np.percentile(ORs_boot.iloc[:,i],95))




    df['CER'] = CERs_table
    df['EER'] = EERs_table

    RRRs_table = []
    ARRs_table = []
    ORs_table = []
    for i in loop:
        RRRs_table.append(str(np.round(RRRs[i]*100,1)) + '% (' + str(np.round(RRRs_low[i]*100,1)) + ' to ' + str(np.round(RRRs_high[i]*100,1))+')')
        ARRs_table.append(str(np.round(ARRs[i]*100,1)) + '% (' + str(np.round(ARRs_low[i]*100,1)) + ' to ' + str(np.round(ARRs_high[i]*100,1))+')')
        ORs_table.append(str(np.round(ORs[i],1)) + '% (' + str(np.round(ORs_low[i],1)) + ';' + str(np.round(ORs_high[i],1))+')')    

    df['RRR'] = RRRs_table
    df['ARR'] = ARRs_table
    df['OR'] = ORs_table

    df['NNT'] = NNTs
    
    df.index = loop
    
    df.to_excel(directory+'/results_table.xlsx')
    
    # === prepare plots ======
    
    colors = ['tab:blue','tab:orange']
    x_shift = [-0.1,0.1]
    
    
    return df


def HTE_sub_plots_binary_abstract(Y, outcome, B, directory, study, effect_modifier='fenotype', specify_range=None):
    
    print('HTE_sub_plots_binary triggered')
    
    risk_modelling = False
    
    # IF RISK MODELLING

    Y['group'] = 'Predicted harm'
    
    print('use absolute ITE values')

    threshold = 0
    mask = Y.ite > threshold
    Y.loc[mask, 'group'] = 'Predicted benefit'
    
        
    groups = ['Predicted harm', 'Predicted benefit']
    
    
    

    # define looper (to loop over group labels which should reveal HTE)
    effect_modifier = 'group'
    loop = np.arange(0,len(groups),1)
    
    
    df = pd.DataFrame()
    df['group'] = groups

    # calculate p value
    families = [sm.families.Binomial(), sm.families.Poisson(), sm.families.Gamma(), sm.families.InverseGaussian()]

    try:
        f = families[0]
        formula = 'Mort ~ group*random'  
        model = smf.glm(formula = formula, data=Y, family=f)
        result = model.fit()
        p = result.pvalues[3]
    except:
        p = np.nan
    
    try:
        f = families[0]
        formula = 'Mort ~ ite*random'  
        model = smf.glm(formula = formula, data=Y, family=f)
        result = model.fit()
        p_ite = result.pvalues[3]
    except:
        p_ite = np.nan
    
    
    
    # number of patients in each group
    
    x_ticks = ["Predicted\nharm group", "Predicted\nbenefit group"]
    
    
    
    # first calc overall/average effect
    control = Y[(Y.random==0)][outcome]
    experiment = Y[(Y.random==1)][outcome]
    EER = experiment.sum()/experiment.shape[0]
    CER = control.sum()/control.shape[0]
    
    # make a / b / c / d (n mort in treated / n survived in treated / n mort in untreated / n survived in untreated )
    a = experiment.sum()
    b = experiment.shape[0] - a 
    c = control.sum()
    d = control.shape[0] - c 
    
    overall_RR = EER/CER
    overall_OR = ((a/b) / (c/d))
    overall_ARR = np.round((CER - EER),3)
    overall_event_rate = Y[outcome].sum()/Y.shape[0]
    
    
    
    # calculate metrics per subgroup
    
    CERs_table = []
    CERs = []
    EERs_table = []
    EERs = []
    RRRs = []
    ARRs = []
    NNTs = []
    ORs = []
    RRs = []


    for i in groups:
        control = Y[(Y[effect_modifier]==i)&(Y.random==0)][outcome]
        experiment = Y[(Y[effect_modifier]==i)&(Y.random==1)][outcome]
        
        # make a / b / c / d (n mort in treated / n survived in treated / n mort in untreated / n survived in untreated )
        a = experiment.sum()
        b = experiment.shape[0] - a 
        c = control.sum()
        d = control.shape[0] - c 

        CER = control.sum()/control.shape[0]
        CERs.append(CER)
        CERs_table.append(str(control.sum()) + '/' + str(control.shape[0]) + ' (' + str(np.round(CER*100,1)) + '%)')
        EER = experiment.sum()/experiment.shape[0]
        EERs.append(EER)
        EERs_table.append(str(experiment.sum()) + '/' + str(experiment.shape[0]) + ' (' + str(np.round(EER*100,1)) + '%)')
        
        RRRs.append(np.round((CER - EER)/CER,3))
        ARRs.append(np.round((CER - EER),3))
        try:
            NNTs.append(int(np.round(1/(CER - EER),3)))
        except:
            NNTs.append(np.nan)
        ORs.append(((a/b) / (c/d)))
        RRs.append(EER/CER)
        
    # bootstrapping procedure
    CERs_boot = []
    EERs_boot = []
    RRRs_boot = []
    ARRs_boot = []
    ORs_boot = []
    RRs_boot = []
    
    #create stratification column 
    Y['strat'] = str(Y.random) + str(Y.Mort)
    
    for b in range(B):
        Y_boot = resample(Y, stratify=Y.strat)
        
        CERs_sample = []
        EERs_sample = []
        RRRs_sample = []
        ARRs_sample = []
        ORs_sample = []
        RRs_sample = []
        
        for i in groups:

            control = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)][outcome]
            experiment = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)][outcome]

            CER = control.sum()/control.shape[0]
            CERs_sample.append(CER)
            EER = experiment.sum()/experiment.shape[0]
            EERs_sample.append(EER)
            RRRs_sample.append((CER - EER)/CER)
            ARRs_sample.append((CER - EER))

            a = experiment.sum()
            b = experiment.shape[0] - a 
            c = control.sum()
            d = control.shape[0] - c 



            OR_boot = ((a/b) / (c/d))
            if (OR_boot == np.inf) | (OR_boot == -np.inf):
                ORs_sample.append(np.nan)

            else:
                ORs_sample.append(OR_boot)

            RRs_sample.append(EER/CER)
            
        CERs_boot.append(CERs_sample)
        EERs_boot.append(EERs_sample)
        RRRs_boot.append(RRRs_sample)
        ARRs_boot.append(ARRs_sample)
        ORs_boot.append(ORs_sample)
        RRs_boot.append(RRs_sample)


    CERs_boot = pd.DataFrame(CERs_boot)
    EERs_boot = pd.DataFrame(EERs_boot)
    RRRs_boot = pd.DataFrame(RRRs_boot)
    ARRs_boot = pd.DataFrame(ARRs_boot)
    ORs_boot = pd.DataFrame(ORs_boot)
    RRs_boot = pd.DataFrame(RRs_boot)

    print(ORs_boot)
    # print(ORs_boot[ORs_boot.isna()])

    CERs_low = []
    CERs_high = []
    EERs_low = []
    EERs_high = []
    RRRs_low = []
    RRRs_high = []
    ARRs_low = []
    ARRs_high = []
    ORs_low = []
    ORs_high = []
    RRs_low = []
    RRs_high = []

    for i in loop:
        CERs_low.append(np.nanpercentile(CERs_boot.iloc[:,i],5))
        CERs_high.append(np.nanpercentile(CERs_boot.iloc[:,i],95))
        EERs_low.append(np.nanpercentile(EERs_boot.iloc[:,i],5))
        EERs_high.append(np.nanpercentile(EERs_boot.iloc[:,i],95))
        RRRs_low.append(np.nanpercentile(RRRs_boot.iloc[:,i],5))
        RRRs_high.append(np.nanpercentile(RRRs_boot.iloc[:,i],95))
        ARRs_low.append(np.nanpercentile(ARRs_boot.iloc[:,i],5))
        ARRs_high.append(np.nanpercentile(ARRs_boot.iloc[:,i],95))
        ORs_low.append(np.nanpercentile(ORs_boot.iloc[:,i],5))
        ORs_high.append(np.nanpercentile(ORs_boot.iloc[:,i],95))
        RRs_low.append(np.nanpercentile(RRs_boot.iloc[:,i],5))
        RRs_high.append(np.nanpercentile(RRs_boot.iloc[:,i],95))




    df['CER'] = CERs_table
    df['EER'] = EERs_table

    RRRs_table = []
    ARRs_table = []
    ORs_table = []
    RRs_table = []
    
    for i in loop:
        RRRs_table.append(str(np.round(RRRs[i]*100,1)) + '% (' + str(np.round(RRRs_low[i]*100,1)) + ' to ' + str(np.round(RRRs_high[i]*100,1))+')')
        ARRs_table.append(str(np.round(ARRs[i]*100,1)) + '% (' + str(np.round(ARRs_low[i]*100,1)) + ' to ' + str(np.round(ARRs_high[i]*100,1))+')')
        ORs_table.append(str(np.round(ORs[i],1)) + '% (' + str(np.round(ORs_low[i],1)) + ';' + str(np.round(ORs_high[i],1))+')')    
        RRs_table.append(str(np.round(RRs[i],1)) + '% (' + str(np.round(RRs_low[i],1)) + ';' + str(np.round(RRs_high[i],1))+')')    

    df['RRR'] = RRRs_table
    df['ARR'] = ARRs_table
    df['OR'] = ORs_table
    df['RR'] = RRs_table

    df['NNT'] = NNTs
    
    df.index = groups
    
       
    # === prepare subplots ======
    
    colors = ['tab:blue','tab:red']
    x_shift = [-0.1,0.1]
    
    fig, ax = plt.subplots(nrows=1,
                           ncols=2,
                           figsize=(12,6),
                           # sharex=True,
                           # sharey=True,
                           )
    
    fontsize_ticks = 15
    fontsize_label = 19
    fontsize_subtitle = 15
    legend_size = 13
    
    markersize=10
    
    
    # =========== Event rates ====================
    for i in loop:
        
        
        
        if i == 0:
            
            ax[0].plot(i+x_shift[0],
                     CERs[i]*100,'D', markersize=markersize, color = colors[0],label='Placebo')
            
            ax[0].plot([i+x_shift[0],i+x_shift[0]],
                     [CERs_low[i]*100,CERs_high[i]*100],
                     color='k',linewidth=1)
            
            ax[0].plot(i+x_shift[1],
                     EERs[i]*100,'^', markersize=markersize, color = colors[1],label='Corticosteroid')
            
            ax[0].plot([i+x_shift[1],i+x_shift[1]],
                     [EERs_low[i]*100,EERs_high[i]*100],
                     color='k',linewidth=1)
        
            
        else:
            ax[0].plot(i+x_shift[0],CERs[i]*100,'D', markersize=markersize,color = colors[0])
            
            ax[0].plot([i+x_shift[0],i+x_shift[0]],
                     [CERs_low[i]*100,CERs_high[i]*100],
                     color='k',linewidth=1)
            
            ax[0].plot(i+x_shift[1],EERs[i]*100,'^', markersize=markersize, color = colors[1])
            
            ax[0].plot([i+x_shift[1],i+x_shift[1]],
                     [EERs_low[i]*100,EERs_high[i]*100],
                     color='k',linewidth=1)
        
            
    # ax[0,0].set_xlabel(x_label, fontsize=fontsize_label)
    ax[0].set_xlim(-0.2, (len(groups) - 1) + 0.2)
    try:
        ax[0].set_ylim(-0.2, np.max([np.max(CERs_high), np.max(EERs_high)])*100 + 2)
    except:
        print('check')

    ax[0].set_ylabel('%', fontsize=fontsize_label)
    ax[0].set_title('Mortality Rate', fontsize=fontsize_label)
    ax[0].set_xticks(loop, x_ticks)
    ax[0].tick_params(labelsize=fontsize_ticks)
    
    ax[0].axhline(y=overall_event_rate*100, xmin = 0.03, xmax = 0.98, linestyle='--', color='k', linewidth=1.5, label='Population average')
    # ax[0].text(0.2, overall_event_rate*100+0.35, 'Population average', fontsize=15, color='b')        
    
    
    # plt.title('Event rates')
    ax[0].legend(loc='upper center', prop={'size': legend_size})
    
    # ax[0].set_title('External validation in 7 RCTs\n (LOTO-CV procedure)', fontsize=fontsize_subtitle)
    
    
    # =========== Odds ratios ======================
    
    for i in loop:
    
        ax[1].plot(i,ORs[i],'o', markersize=markersize, color = 'k')
        ax[1].plot([i,i],[ORs_low[i], ORs_high[i]], color='k',linewidth=1)
    
        
    
    # plot average effect
    ax[1].axhline(y=overall_OR, xmin = 0.03, xmax = 0.98, linestyle='--', color='k', linewidth=1.5, label='Population average')
    ax[1].axhline(y=1, linestyle='-',color='k',linewidth=0.6)        
    
    or_range = (np.nanmax(ORs_high) - np.nanmin(ORs_low))
    y_loc_p_value = 2.51
                

    # ax[1].text(0.15, overall_OR- (or_range*0.1), 'Population average', fontsize=15, color='b')        

    # plot p value
    

    ax[1].text(0.5, y_loc_p_value, 
            #    'P for interaction = '+str(np.round(p,3)) + '*', 
               '*',
               fontsize=12, color='k')

    # ax[1,0].text(0.35, 3, 'P_ite='+str(np.round(p_ite,4)), fontsize=15, color='b')

    # ax[1,0].set_xlabel(x_label, fontsize=fontsize_label)
    ax[1].set_ylabel('Benefit \u2190  \u2192 Harm             ', fontsize=fontsize_label)
    ax[1].set_title('Odds Ratio', fontsize=fontsize_label)
    ax[1].set_xlim(-0.2, (len(groups) - 1) + 0.2)
    ax[1].set_ylim(np.nanmin(ORs_low) - 0.7, 
                3.2
                )
    
    # p value bracket
    ax[1].plot([0, 0],[ORs_high[0]+0.2, ORs_high[0]+0.3], linewidth=0.8, color='k')
    ax[1].plot([1, 1],[ORs_high[0]+0.2, ORs_high[0]+0.3], linewidth=0.8, color='k')

    ax[1].plot([0, 1],[ORs_high[0]+0.3, ORs_high[0]+0.3], linewidth=0.8, color='k')
    
    # plt.title('Odds ratios')
    
    ax[1].set_xticks(loop, x_ticks)
    ax[1].tick_params(labelsize=fontsize_ticks)
    ax[1].legend(loc='upper center', prop={'size': legend_size})
    
    
    
    plt.tight_layout()
    plt.savefig(directory+'/HTE_subplot_binary_'+ study + '_ABSTRACT.jpg',dpi=300)
    
    return df

def HTE_sub_plots_binary(Y, outcome, B, directory, study, effect_modifier='fenotype', specify_range=None):
    
    print('HTE_sub_plots_binary triggered')
    
    risk_modelling = False
    
    # IF RISK MODELLING

    if risk_modelling:    

        Y['group'] = 'Q1'
        
        threshold = np.percentile(Y.ite, 25)    
        mask = Y.ite > threshold
        Y.loc[mask, 'group'] = 'Q2'

        threshold = np.percentile(Y.ite, 50)    
        mask = Y.ite > threshold
        Y.loc[mask, 'group'] = 'Q3'

        threshold = np.percentile(Y.ite, 75)    
        mask = Y.ite > threshold
        Y.loc[mask, 'group'] = 'Q4'
            
        groups = ['Q1', 'Q2', 'Q3', 'Q4']


    # IF EFFECT MODELLING
    else:
        
        # add quartiles for interaction test
        Y['quartile'] = 1

        threshold = np.percentile(Y.ite, 25)    
        mask = Y.ite > threshold
        Y.loc[mask, 'quartile'] = 2

        threshold = np.percentile(Y.ite, 50)    
        mask = Y.ite > threshold
        Y.loc[mask, 'quartile'] = 3

        threshold = np.percentile(Y.ite, 75)    
        mask = Y.ite > threshold
        Y.loc[mask, 'quartile'] = 4
            
        
        
        
        # make binary groups
        
        Y['group'] = 'Predicted harm'
        
        print('use absolute ITE values')

        threshold = 0
        mask = Y.ite > threshold
        Y.loc[mask, 'group'] = 'Predicted benefit'
        
            
        groups = ['Predicted harm', 'Predicted benefit']
        
    
    

    # define looper (to loop over group labels which should reveal HTE)
    effect_modifier = 'group'
    loop = np.arange(0,len(groups),1)
    
    
    df = pd.DataFrame()
    df['group'] = groups

    # calculate p value
    families = [sm.families.Binomial(), sm.families.Poisson(), sm.families.Gamma(), sm.families.InverseGaussian()]

    try:
        f = families[0]
        formula = 'Mort ~ group*random'  
        model = smf.glm(formula = formula, data=Y, family=f)
        result = model.fit()
        p = result.pvalues[3]
    except:
        p = np.nan
    
    try:
        f = families[0]
        formula = 'Mort ~ ite*random'  
        model = smf.glm(formula = formula, data=Y, family=f)
        result = model.fit()
        p_ite = result.pvalues[3]
    except:
        p_ite = np.nan

    try:
        f = families[0]
        formula = 'Survival ~ ite*quartile'  
        model = smf.glm(formula = formula, data=Y, family=f)
        result = model.fit()
        p_quartile = result.pvalues[3]
    except:
        p_quartile = np.nan
    
    
    
    # number of patients in each group
    # x_ticks = ['Predicted\nharm group', 'Predicted\nbenefit group']
    x_ticks = []

    for i in groups:
        n = Y[Y[effect_modifier]==i].shape[0]
        n_events = int(Y[Y[effect_modifier]==i][outcome].sum(0))

        x_tick = str(i) + '\n(n=' + str(n) + ')'
        
        if specify_range:
            x_tick += ('\n ' + specify_range[i-1])
        
        
        x_ticks.append(x_tick)
    
        
    
    
    
    
    # first calc overall/average effect
    control = Y[(Y.random==0)][outcome]
    experiment = Y[(Y.random==1)][outcome]
    EER = experiment.sum()/experiment.shape[0]
    CER = control.sum()/control.shape[0]
    
    # make a / b / c / d (n mort in treated / n survived in treated / n mort in untreated / n survived in untreated )
    a = experiment.sum()
    b = experiment.shape[0] - a 
    c = control.sum()
    d = control.shape[0] - c 
    
    overall_RR = EER/CER
    overall_OR = ((a/b) / (c/d))
    overall_ARR = np.round((CER - EER),3)
    overall_event_rate = Y[outcome].sum()/Y.shape[0]
    
    
    
    # calculate metrics per subgroup
    
    CERs_table = []
    CERs = []
    EERs_table = []
    EERs = []
    RRRs = []
    ARRs = []
    NNTs = []
    ORs = []
    RRs = []


    for i in groups:
        control = Y[(Y[effect_modifier]==i)&(Y.random==0)][outcome]
        experiment = Y[(Y[effect_modifier]==i)&(Y.random==1)][outcome]
        
        # make a / b / c / d (n mort in treated / n survived in treated / n mort in untreated / n survived in untreated )
        a = experiment.sum()
        b = experiment.shape[0] - a 
        c = control.sum()
        d = control.shape[0] - c 

        CER = control.sum()/control.shape[0]
        CERs.append(CER)
        CERs_table.append(str(control.sum()) + '/' + str(control.shape[0]) + ' (' + str(np.round(CER*100,1)) + '%)')
        EER = experiment.sum()/experiment.shape[0]
        EERs.append(EER)
        EERs_table.append(str(experiment.sum()) + '/' + str(experiment.shape[0]) + ' (' + str(np.round(EER*100,1)) + '%)')
        
        RRRs.append(np.round((CER - EER)/CER,3))
        ARRs.append(np.round((CER - EER),3))
        try:
            NNTs.append(int(np.round(1/(CER - EER),3)))
        except:
            NNTs.append(np.nan)
        ORs.append(((a/b) / (c/d)))
        RRs.append(EER/CER)
        
    # bootstrapping procedure
    CERs_boot = []
    EERs_boot = []
    RRRs_boot = []
    ARRs_boot = []
    ORs_boot = []
    RRs_boot = []
    
    #create stratification column 
    Y['strat'] = str(Y.random) + str(Y.Mort)
    
    for b in range(B):
        Y_boot = resample(Y, stratify=Y.strat)
        
        CERs_sample = []
        EERs_sample = []
        RRRs_sample = []
        ARRs_sample = []
        ORs_sample = []
        RRs_sample = []
        
        for i in groups:

            control = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)][outcome]
            experiment = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)][outcome]

            CER = control.sum()/control.shape[0]
            CERs_sample.append(CER)
            EER = experiment.sum()/experiment.shape[0]
            EERs_sample.append(EER)
            RRRs_sample.append((CER - EER)/CER)
            ARRs_sample.append((CER - EER))

            a = experiment.sum()
            b = experiment.shape[0] - a 
            c = control.sum()
            d = control.shape[0] - c 



            OR_boot = ((a/b) / (c/d))
            if (OR_boot == np.inf) | (OR_boot == -np.inf):
                ORs_sample.append(np.nan)

            else:
                ORs_sample.append(OR_boot)

            RRs_sample.append(EER/CER)
            
        CERs_boot.append(CERs_sample)
        EERs_boot.append(EERs_sample)
        RRRs_boot.append(RRRs_sample)
        ARRs_boot.append(ARRs_sample)
        ORs_boot.append(ORs_sample)
        RRs_boot.append(RRs_sample)


    CERs_boot = pd.DataFrame(CERs_boot)
    EERs_boot = pd.DataFrame(EERs_boot)
    RRRs_boot = pd.DataFrame(RRRs_boot)
    ARRs_boot = pd.DataFrame(ARRs_boot)
    ORs_boot = pd.DataFrame(ORs_boot)
    RRs_boot = pd.DataFrame(RRs_boot)

    print(ORs_boot)
    # print(ORs_boot[ORs_boot.isna()])

    CERs_low = []
    CERs_high = []
    EERs_low = []
    EERs_high = []
    RRRs_low = []
    RRRs_high = []
    ARRs_low = []
    ARRs_high = []
    ORs_low = []
    ORs_high = []
    RRs_low = []
    RRs_high = []

    for i in loop:
        CERs_low.append(np.nanpercentile(CERs_boot.iloc[:,i],5))
        CERs_high.append(np.nanpercentile(CERs_boot.iloc[:,i],95))
        EERs_low.append(np.nanpercentile(EERs_boot.iloc[:,i],5))
        EERs_high.append(np.nanpercentile(EERs_boot.iloc[:,i],95))
        RRRs_low.append(np.nanpercentile(RRRs_boot.iloc[:,i],5))
        RRRs_high.append(np.nanpercentile(RRRs_boot.iloc[:,i],95))
        ARRs_low.append(np.nanpercentile(ARRs_boot.iloc[:,i],5))
        ARRs_high.append(np.nanpercentile(ARRs_boot.iloc[:,i],95))
        ORs_low.append(np.nanpercentile(ORs_boot.iloc[:,i],5))
        ORs_high.append(np.nanpercentile(ORs_boot.iloc[:,i],95))
        RRs_low.append(np.nanpercentile(RRs_boot.iloc[:,i],5))
        RRs_high.append(np.nanpercentile(RRs_boot.iloc[:,i],95))




    df['CER'] = CERs_table
    df['EER'] = EERs_table

    RRRs_table = []
    ARRs_table = []
    ORs_table = []
    RRs_table = []
    
    for i in loop:
        RRRs_table.append(str(np.round(RRRs[i]*100,1)) + '% (' + str(np.round(RRRs_low[i]*100,1)) + ' to ' + str(np.round(RRRs_high[i]*100,1))+')')
        ARRs_table.append(str(np.round(ARRs[i]*100,1)) + '% (' + str(np.round(ARRs_low[i]*100,1)) + ' to ' + str(np.round(ARRs_high[i]*100,1))+')')
        ORs_table.append(str(np.round(ORs[i],1)) + '% (' + str(np.round(ORs_low[i],1)) + ';' + str(np.round(ORs_high[i],1))+')')    
        RRs_table.append(str(np.round(RRs[i],1)) + '% (' + str(np.round(RRs_low[i],1)) + ';' + str(np.round(RRs_high[i],1))+')')    

    df['RRR'] = RRRs_table
    df['ARR'] = ARRs_table
    df['OR'] = ORs_table
    df['RR'] = RRs_table

    df['NNT'] = NNTs
    
    df.index = groups
    
       
    # === prepare subplots ======
    
    colors = ['tab:blue','tab:red']
    x_shift = [-0.1,0.1]
    
    fig, ax = plt.subplots(nrows=1,
                           ncols=3,
                           figsize=(18,6),
                           # sharex=True,
                           # sharey=True,
                           )
    
    fontsize_ticks = 11
    fontsize_label = 19
    fontsize_subtitle = 15
    legend_size = 15
    
    markersize=10
    
    
    # =========== Event rates ====================
    
    
    for i in loop:
        
        
        
        if i == 0:
            
            ax[0].plot(i+x_shift[0],
                     CERs[i]*100,'D', markersize=markersize, color = colors[0],label='Placebo')
            
            ax[0].plot([i+x_shift[0],i+x_shift[0]],
                     [CERs_low[i]*100,CERs_high[i]*100],
                     color='k',linewidth=1)
            
            ax[0].plot(i+x_shift[1],
                     EERs[i]*100,'^', markersize=markersize, color = colors[1],label='Corticosteroid')
            
            ax[0].plot([i+x_shift[1],i+x_shift[1]],
                     [EERs_low[i]*100,EERs_high[i]*100],
                     color='k',linewidth=1)
        
            
        else:
            ax[0].plot(i+x_shift[0],CERs[i]*100,'D', markersize=markersize,color = colors[0])
            
            ax[0].plot([i+x_shift[0],i+x_shift[0]],
                     [CERs_low[i]*100,CERs_high[i]*100],
                     color='k',linewidth=1)
            
            ax[0].plot(i+x_shift[1],EERs[i]*100,'^', markersize=markersize, color = colors[1])
            
            ax[0].plot([i+x_shift[1],i+x_shift[1]],
                     [EERs_low[i]*100,EERs_high[i]*100],
                     color='k',linewidth=1)
        
            
    # ax[0,0].set_xlabel(x_label, fontsize=fontsize_label)
    ax[0].set_xlim(-0.2, (len(groups) - 1) + 0.2)
    try:
        ax[0].set_ylim(-0.2, np.max([np.max(CERs_high), np.max(EERs_high)])*100 + 2)
    except:
        print('check')

    ax[0].set_title('Mortality rate',fontsize=fontsize_subtitle)
    ax[0].set_ylabel('%', fontsize=fontsize_label)
    ax[0].set_xticks(loop, x_ticks)
    ax[0].tick_params(labelsize=fontsize_ticks)
    
    ax[0].axhline(y=overall_event_rate*100, xmin = 0.03, xmax = 0.98, linestyle='--', color='k', linewidth=1.5, label='Population average')
    # ax[0].text(0.2, overall_event_rate*100+0.35, 'Population average', fontsize=15, color='b')        
    
    
    # plt.title('Event rates')
    ax[0].legend(loc='upper center', prop={'size': legend_size})
    # ax[0].set_title('External validation in 7 RCTs\n (LOTO-CV procedure)', fontsize=fontsize_subtitle)
    
    
    # =========== Odds ratios ======================
    if study in ['Confalonieri', 'Fernandez']:
        # OR not defined
        pass
    else:

        for i in loop:
        
            ax[1].plot(i,ORs[i],'o', markersize=markersize, color = 'k')
            ax[1].plot([i,i],[ORs_low[i], ORs_high[i]], color='k',linewidth=1)
        
            
        
        # plot average effect
        ax[1].axhline(y=overall_OR, xmin = 0.03, xmax = 0.98, linestyle='--', color='k', linewidth=1.5, label='Population average')
        ax[1].axhline(y=1, linestyle='-',color='k',linewidth=0.6)        
        
        or_range = (np.nanmax(ORs_high) - np.nanmin(ORs_low))
        y_loc_p_value = np.nanmin(ORs_low) + or_range*0.7
                    

        # ax[1].text(0.15, overall_OR- (or_range*0.1), 'Population average', fontsize=15, color='b')        

        

        # ax[1,0].text(0.35, 3, 'P_ite='+str(np.round(p_ite,4)), fontsize=15, color='b')

        # ax[1,0].set_xlabel(x_label, fontsize=fontsize_label)
        ax[1].set_title('Odds Ratio', fontsize=fontsize_subtitle)
        ax[1].set_ylabel('Benefit \u2190  \u2192 Harm', fontsize=fontsize_label)
        
        ax[1].set_xlim(-0.2, (len(groups) - 1) + 0.2)
        ax[1].set_ylim(np.nanmin(ORs_low) - 0.8, 
                    np.nanmax(ORs_high) + 0.8
                    )
        
        # plt.title('Odds ratios')
        
        ax[1].set_xticks(loop, x_ticks)
        ax[1].tick_params(labelsize=fontsize_ticks)
    
        if p<.05:
            # p value bracket
            ax[1].plot([0, 0],[ORs_high[0]+0.2, ORs_high[0]+0.3], linewidth=0.8, color='k')
            ax[1].plot([1, 1],[ORs_high[0]+0.2, ORs_high[0]+0.3], linewidth=0.8, color='k')
            ax[1].plot([0, 1],[ORs_high[0]+0.3, ORs_high[0]+0.3], linewidth=0.8, color='k')
            
            # plot p value
            if risk_modelling:
                ax[1].text(0.35, y_loc_p_value, 'P = '+str(np.round(p_ite,4)), fontsize=15, color='k')
            else:
                
                # ax[1].text(0.35, y_loc_p_value, 'P = '+str(np.round(p,4)), fontsize=15, color='k')

                ax[1].text(0.5, ORs_high[0]+0.32, '*', fontsize=15, color='k')
                # ax[1].text(0.35, y_loc_p_value -1, 'P = '+str(np.round(p_quartile,4)), fontsize=15, color='b')

        ax[1].legend(loc='lower center', prop={'size': legend_size})
    
    
    # ====== absolute risk reduction ==========================

    for i in loop:

        ax[2].plot(i,ARRs[i]*100,'o', markersize=markersize, color = 'k')
        ax[2].plot([i,i],[ARRs_low[i]*100,ARRs_high[i]*100],color='k',linewidth=1)

    
    # plot average effect
    ax[2].axhline(y=overall_ARR*100, xmin = 0.03, xmax = 0.98, linestyle='--', color='k', linewidth=1.5, label='Population average')
    ax[2].axhline(y=0,linestyle='-',color='k',linewidth=0.6)        
    
    # ax[2].text(0.1, overall_ARR*100+.4, 'Population average', fontsize=15, color='b')        

    # ax[2,0].set_xlabel(x_label, fontsize=fontsize_label)
    ax[2].set_title('Mortality reduction' ,fontsize=fontsize_subtitle)
    ax[2].set_ylabel(u'%\n Harm \u2190  \u2192 Benefit', fontsize=fontsize_label)
    
    ax[2].set_ylim(np.nanmin(ARRs_low)*100 - 1, 
                   np.nanmax(ARRs_high)*100 + 1
                   )
    ax[2].set_xlim(-0.2, (len(groups) - 1) + 0.2)
    
    # plt.title('Risk differences')

    ax[2].set_xticks(loop, x_ticks)
    ax[2].tick_params(labelsize=fontsize_ticks)
    ax[2].legend(loc='upper center', prop={'size': legend_size})
    
    plt.tight_layout()
    plt.savefig(directory+'/HTE_subplot_binary_'+ study + '.jpg',dpi=300)
    
    return df

def HTE_sub_plots(Y, outcome, B, directory, grouping_method, effect_modifier='fenotype', specify_range=None):
    
    
    
    Y['Benefit group'] = 'Harm'
    
    if grouping_method == 'tertiles':
        
        print('use ITE tertiles')
        
        threshold1 = np.percentile(Y.ite,33.3)
        mask = Y.ite > threshold1
        Y.loc[mask, 'Benefit group'] = 'Neutral'
        
        threshold2 = np.percentile(Y.ite,66.7)
        mask = Y.ite >= threshold2
        Y.loc[mask, 'Benefit group'] = 'Benefit'
        
        print('harm group ITE range:')
        print(str(Y.ite.min()) + ' ; ' + str(threshold1))
        
        print('neutral group ITE range:')
        print(str(threshold1) + ' ; ' + str(threshold2))
        
        print('benefit group ITE range:')
        print(str(threshold2) + ' ; ' + str(Y.ite.max()))
        
    elif grouping_method == 'quartiles':
        
        print('use ITE quartiles')
        
        threshold1 = np.percentile(Y.ite,25)
        mask = Y.ite > threshold1
        Y.loc[mask, 'Benefit group'] = 'Neutral'
    
        threshold2 = np.percentile(Y.ite,75)
        mask = Y.ite >= threshold2
        Y.loc[mask, 'Benefit group'] = 'Benefit'
        
        print('harm group ITE range:')
        print(str(Y.ite.min()) + ' ; ' + str(threshold1))
        
        print('neutral group ITE range:')
        print(str(threshold1) + ' ; ' + str(threshold2))
        
        print('benefit group ITE range:')
        print(str(threshold2) + ' ; ' + str(Y.ite.max()))
        
    groups = ['Harm', 'Neutral', 'Benefit']
    effect_modifier = 'Benefit group'
    
    

    # define looper (to loop over group labels which should reveal HTE)
    loop = np.arange(0,len(groups),1)
    
    
    df = pd.DataFrame()
    df['group'] = groups
    
    
    # number of patients in each group
    
    x_ticks = []
    for i in groups:
        n = Y[Y[effect_modifier]==i].shape[0]
        n_events = int(Y[Y[effect_modifier]==i][outcome].sum(0))
        x_tick = str(i) + '\n (' + str(n_events) + '/' + str(n) +')' 
        
        if specify_range:
            x_tick += ('\n ' + specify_range[i-1])
        
        
        x_ticks.append(x_tick)
    
        
    
    
    
    
    # first calc overall/average effect
    control = Y[(Y.random==0)][outcome]
    experiment = Y[(Y.random==1)][outcome]
    EER = experiment.sum()/experiment.shape[0]
    CER = control.sum()/control.shape[0]
    
    print(EER)
    print(CER)
    
    overall_RR = EER/CER
    overall_OR = (EER/(1-EER))/(CER/(1-CER))
    overall_ARR = np.round((CER - EER),3)
    overall_event_rate = Y[outcome].sum()/Y.shape[0]
    
    
    print('overall OR: ',overall_OR)
    print('overall ARR: ',overall_ARR)
    
    # calculate metrics per subgroup
    
    CERs_table = []
    CERs = []
    EERs_table = []
    EERs = []
    RRRs = []
    ARRs = []
    NNTs = []
    ORs = []
    RRs = []


    for i in groups:
        control = Y[(Y[effect_modifier]==i)&(Y.random==0)][outcome]
        experiment = Y[(Y[effect_modifier]==i)&(Y.random==1)][outcome]
        
        CER = control.sum()/control.shape[0]
        CERs.append(CER)
        CERs_table.append(str(control.sum()) + '/' + str(control.shape[0]) + ' (' + str(np.round(CER*100,1)) + '%)')
        EER = experiment.sum()/experiment.shape[0]
        EERs.append(EER)
        EERs_table.append(str(experiment.sum()) + '/' + str(experiment.shape[0]) + ' (' + str(np.round(EER*100,1)) + '%)')
        
        RRRs.append(np.round((CER - EER)/CER,3))
        ARRs.append(np.round((CER - EER),3))
        NNTs.append(int(np.round(1/(CER - EER),3)))
        ORs.append((EER/(1-EER))/(CER/(1-CER)))
        RRs.append(EER/CER)
        
    # bootstrapping procedure
    CERs_boot = []
    EERs_boot = []
    RRRs_boot = []
    ARRs_boot = []
    ORs_boot = []
    RRs_boot = []
    
    #create stratification column 
    Y['strat'] = str(Y.random) + str(Y.Mort)
    
    for b in range(B):
        Y_boot = resample(Y, stratify=Y.strat)
        
        CERs_sample = []
        EERs_sample = []
        RRRs_sample = []
        ARRs_sample = []
        ORs_sample = []
        RRs_sample = []
        
        for i in groups:

            CER = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)][outcome].sum()/Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==0)].shape[0]
            CERs_sample.append(CER)
            EER = Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)][outcome].sum()/Y_boot[(Y_boot[effect_modifier]==i)&(Y_boot.random==1)].shape[0]
            EERs_sample.append(EER)
            RRRs_sample.append((CER - EER)/CER)
            ARRs_sample.append((CER - EER))
            ORs_sample.append((EER/(1-EER))/(CER/(1-CER)))
            RRs_sample.append(EER/CER)
            
        CERs_boot.append(CERs_sample)
        EERs_boot.append(EERs_sample)
        RRRs_boot.append(RRRs_sample)
        ARRs_boot.append(ARRs_sample)
        ORs_boot.append(ORs_sample)
        RRs_boot.append(RRs_sample)


    CERs_boot = pd.DataFrame(CERs_boot)
    EERs_boot = pd.DataFrame(EERs_boot)
    RRRs_boot = pd.DataFrame(RRRs_boot)
    ARRs_boot = pd.DataFrame(ARRs_boot)
    ORs_boot = pd.DataFrame(ORs_boot)
    RRs_boot = pd.DataFrame(RRs_boot)


    CERs_low = []
    CERs_high = []
    EERs_low = []
    EERs_high = []
    RRRs_low = []
    RRRs_high = []
    ARRs_low = []
    ARRs_high = []
    ORs_low = []
    ORs_high = []
    RRs_low = []
    RRs_high = []

    for i in loop:
        CERs_low.append(np.percentile(CERs_boot.iloc[:,i],5))
        CERs_high.append(np.percentile(CERs_boot.iloc[:,i],95))
        EERs_low.append(np.percentile(EERs_boot.iloc[:,i],5))
        EERs_high.append(np.percentile(EERs_boot.iloc[:,i],95))
        RRRs_low.append(np.percentile(RRRs_boot.iloc[:,i],5))
        RRRs_high.append(np.percentile(RRRs_boot.iloc[:,i],95))
        ARRs_low.append(np.percentile(ARRs_boot.iloc[:,i],5))
        ARRs_high.append(np.percentile(ARRs_boot.iloc[:,i],95))
        ORs_low.append(np.percentile(ORs_boot.iloc[:,i],5))
        ORs_high.append(np.percentile(ORs_boot.iloc[:,i],95))
        RRs_low.append(np.percentile(RRs_boot.iloc[:,i],5))
        RRs_high.append(np.percentile(RRs_boot.iloc[:,i],95))




    df['CER'] = CERs_table
    df['EER'] = EERs_table

    RRRs_table = []
    ARRs_table = []
    ORs_table = []
    RRs_table = []
    
    for i in loop:
        RRRs_table.append(str(np.round(RRRs[i]*100,1)) + '% (' + str(np.round(RRRs_low[i]*100,1)) + ' to ' + str(np.round(RRRs_high[i]*100,1))+')')
        ARRs_table.append(str(np.round(ARRs[i]*100,1)) + '% (' + str(np.round(ARRs_low[i]*100,1)) + ' to ' + str(np.round(ARRs_high[i]*100,1))+')')
        ORs_table.append(str(np.round(ORs[i],1)) + '% (' + str(np.round(ORs_low[i],1)) + ';' + str(np.round(ORs_high[i],1))+')')    
        RRs_table.append(str(np.round(RRs[i],1)) + '% (' + str(np.round(RRs_low[i],1)) + ';' + str(np.round(RRs_high[i],1))+')')    

    df['RRR'] = RRRs_table
    df['ARR'] = ARRs_table
    df['OR'] = ORs_table
    df['RR'] = RRs_table

    df['NNT'] = NNTs
    
    df.index = groups
    
       
    # === prepare subplots ======
    
    colors = ['tab:blue','tab:red']
    x_shift = [-0.1,0.1]
    
    fig, ax = plt.subplots(nrows=3,
                           ncols=2,
                           figsize=(15,18),
                           # sharex=True,
                           # sharey=True,
                           )
    
    fontsize_ticks = 16
    fontsize_label = 19
    fontsize_subtitle = 15
    legend_size = 15
    
    markersize=10
    
    
    # =========== Event rates ====================
    

    for i in loop:
        if i == 0:
            
            ax[0,0].plot(i+x_shift[0],
                     CERs[i]*100,'o', markersize=markersize, color = colors[0],label='Placebo')
            
            ax[0,0].plot([i+x_shift[0],i+x_shift[0]],
                     [CERs_low[i]*100,CERs_high[i]*100],
                     color='k',linewidth=1)
            
            ax[0,0].plot(i+x_shift[1],
                     EERs[i]*100,'^', markersize=markersize, color = colors[1],label='Corticosteroid')
            
            ax[0,0].plot([i+x_shift[1],i+x_shift[1]],
                     [EERs_low[i]*100,EERs_high[i]*100],
                     color='k',linewidth=1)
        
            
        else:
            ax[0,0].plot(i+x_shift[0],CERs[i]*100,'o', markersize=markersize,color = colors[0])
            
            ax[0,0].plot([i+x_shift[0],i+x_shift[0]],
                     [CERs_low[i]*100,CERs_high[i]*100],
                     color='k',linewidth=1)
            
            ax[0,0].plot(i+x_shift[1],EERs[i]*100,'^', markersize=markersize, color = colors[1])
            
            ax[0,0].plot([i+x_shift[1],i+x_shift[1]],
                     [EERs_low[i]*100,EERs_high[i]*100],
                     color='k',linewidth=1)
        
            
    # ax[0,0].set_xlabel(x_label, fontsize=fontsize_label)
    ax[0,0].set_ylabel('Mortality rate (%)', fontsize=fontsize_label)
    ax[0,0].set_xticks(loop, x_ticks)
    ax[0,0].tick_params(labelsize=fontsize_ticks)
    
    ax[0,0].axhline(y=overall_event_rate*100, xmin = 0.03, xmax = 0.98, linestyle='--', color='b', linewidth=1.5)
    ax[0,0].text(0.3, overall_event_rate*100+0.35, 'Average mortality rate', fontsize=15, color='b')        
    
    
    # plt.title('Event rates')
    ax[0,0].legend(loc='upper center', prop={'size': legend_size})
    ax[0,0].set_title('Internal validation', fontsize=fontsize_subtitle)
    
    
    # =========== Risk ratios ======================
    for i in loop:
    
        ax[1,0].plot(i,RRs[i],'o', markersize=markersize, color = colors[0])
        ax[1,0].plot([i,i],[RRs_low[i], RRs_high[i]], color='k',linewidth=1)
    
        
    
    # plot average effect
    ax[1,0].axhline(y=overall_RR, xmin = 0.03, xmax = 0.98, linestyle='--', color='b', linewidth=1.5)
    ax[1,0].axhline(y=1, linestyle='-',color='k',linewidth=0.6)        
    
    ax[1,0].text(0.15, overall_RR-0.3, 'Average risk ratio', fontsize=15, color='b')        

    # ax[1,0].set_xlabel(x_label, fontsize=fontsize_label)
    ax[1,0].set_ylabel('Risk ratio\n Benefit \u2190  \u2192 Harm', fontsize=fontsize_label)
    
    
    ax[1,0].set_ylim(-1.3,3)
    # plt.title('Odds ratios')
    
    ax[1,0].set_xticks(loop, x_ticks)
    ax[1,0].tick_params(labelsize=fontsize_ticks)
    
    
    # ====== absolute risk reduction ==========================

    for i in loop:

        ax[2,0].plot(i,ARRs[i]*100,'o', markersize=markersize, color = colors[0])
        ax[2,0].plot([i,i],[ARRs_low[i]*100,ARRs_high[i]*100],color='k',linewidth=1)

    
    # plot average effect
    ax[2,0].axhline(y=overall_ARR*100, xmin = 0.03, xmax = 0.98, linestyle='--', color='b', linewidth=1.5)
    ax[2,0].axhline(y=0,linestyle='-',color='k',linewidth=0.6)        
    
    ax[2,0].text(0.1, overall_ARR*100+.4, 'Average mortality\nreduction', fontsize=15, color='b')        

    # ax[2,0].set_xlabel(x_label, fontsize=fontsize_label)
    ax[2,0].set_ylabel(u'Mortality reduction (%)\n Harm \u2190  \u2192 Benefit', fontsize=fontsize_label)
    ax[2,0].set_ylim(-8.5,8.5)
    
    # plt.title('Risk differences')

    ax[2,0].set_xticks(loop, x_ticks)
    ax[2,0].tick_params(labelsize=fontsize_ticks)
    
    
    
    # EXTERNAL VALIDATION
    
    ax[0,1].set_title('External validation', fontsize=fontsize_subtitle)
    
    plt.tight_layout()
    plt.savefig(directory+'/HTE_subplot.jpg',dpi=300)
    
    return df

def risk_model(data, Y, outcome, learner, penalty):
    
    # internal validation --> train datra = test data !!!
    
    # normalize and impute
    X_imp, _ = normalize_and_impute(data, data)
    
    risk_predictions, model = risk_learner(X_imp, Y, X_imp, learner, outcome, penalty)

    return risk_predictions, model

def risk_model_LORO(include, data, Y, outcome, learner, penalty, C):

    total_predictions = list()
    
    for i in include: # loop over all included RCT studies
        print(i)
        
        # split in train and test
        X_train = data[Y.STUDY!=i].reset_index(drop=True)
        X_test = data[Y.STUDY==i].reset_index(drop=True)
        
        Y_train = Y[Y.STUDY!=i].reset_index(drop=True)
        
        # normalize and impute
        X_train_imp, X_test_imp = normalize_and_impute(X_train, X_test)
        

        
        predictions, _ = risk_learner(X_train_imp, Y_train, X_test_imp, learner, outcome, penalty, C)
        
            
        # adds ites to total ites
        total_predictions = total_predictions + predictions
            
    return total_predictions

def generic_backward_pruning(X, Y, X_obs, imp, B, C, penalty,
                       outer_loop_left_out_study, metric,
                       initial_main_effects, initial_interactions,
                       save_directory, 
                       main_effects_pruning = False,
                       high_score_main_effects_pruning = None,
                       paired_main_effect_interaction=False,
                       causal_forest=False
                       ):
    
    print('generic_backward_pruning triggered')

    study_dict = {0:'Meijvis', 1:'Endeman', 2:'Blum', 3:'Wittermans', 
                  4:'Torres', 5:'Snijders', 6:'Confalonieri', 7: 'Fernández-Serrano'}
    
    studies_inner_loro = Y.STUDY.unique()
    
    
    print('\nusing metric: ', metric)
    print('\nusing '+ penalty +  ' penalization, inverse strength = ', C)
    print('\nInner loro loop studies: ')
    
    for study in studies_inner_loro:
        print(study_dict[study])
        
    if metric in ['etd', 'eqd', 'abs_ite_lift']:
        tol = -0.2
    else: 
        tol = -10


    if main_effects_pruning:
        highscore = high_score_main_effects_pruning
    
    else:
        # initialize base c-for-benefit
        if metric in ['etd', 'eqd', 'abs_ite_lift', 'auc_benefit']:
            highscore = 0

        else:
            highscore = 0.5
    
    # initialize candidates to prune
    if paired_main_effect_interaction:
        candidates = initial_main_effects

    elif main_effects_pruning:

        interactions_to_lock = []
        for variable in initial_interactions:
            interactions_to_lock.append(variable[:-7])

        candidates = [x for x in initial_main_effects if x not in interactions_to_lock]

    else:
        candidates = initial_interactions


    # initialize pruned candidates
    pruned_main_effects = []
    pruned_interactions = []

    # ==== Loop 1: prune one feature per iteration
    for round_number in np.arange(1,len(candidates),1):
        
        print('==== ROUND '+ str(round_number) + ' =====')
        
        boxplot_data = pd.DataFrame()
        
        if (paired_main_effect_interaction) | (main_effects_pruning):
            left_over_candidates = [x for x in candidates if x not in pruned_main_effects]

        else:
            left_over_candidates = [x for x in candidates if x not in pruned_interactions]

        print('left over candidates:')
        print(left_over_candidates)
        
        # empty list for mean CV score per candidate 
        overall_score_per_candidate = []
        
        # ======= LOOP 2 ; over candidate features left in this round ===========
        for candidate in left_over_candidates:
            
            print('\nevaluating: ' + candidate)
            
            # collect feature set for this iteration
            left_over_main_effects = [x for x in initial_main_effects if x not in pruned_main_effects]
            
            if (paired_main_effect_interaction) | (main_effects_pruning):
                main_effects_to_include = [x for x in left_over_main_effects if x not in [candidate]]
            
            if main_effects_pruning:
                left_over_interactions = [x for x in interactions_to_lock if x not in pruned_interactions]
            else:
                left_over_interactions = [x for x in initial_interactions if x not in pruned_interactions]
            interactions_to_include = [x for x in left_over_interactions if x not in [candidate]]
            
            features = []

            if (paired_main_effect_interaction) | (main_effects_pruning):
                for variable in main_effects_to_include:
                    features.append(variable)
            else:
                for variable in left_over_main_effects:
                    features.append(variable)
            
            for variable in interactions_to_include:
                features.append(variable+'*random')
            
            
            print('features in model: ', features)
            
            
            Y_test_stacked = pd.DataFrame()
            
            # ======= LOOP 3 ; inner LORO to evaluate candidate ===========
            for study in studies_inner_loro:
                
                # print('left-out RCT: ', study_dict[study])
                # split train and test set in LORO fashion
                
                # train data from original df
                mask = Y.STUDY!=study
                X_train = X[mask]
                Y_train = Y[mask]
                
                # test data from modified dfs (untreated and treated)
                mask = Y.STUDY==study
                X_test = X[mask]
                Y_test = Y[mask]
                
                assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
                assert Y_train.shape[0] + Y_test.shape[0] == Y.shape[0]
                
                # NORMALIZE / IMPUTE
                X_train_imp, X_test_imp = normalize_and_impute(X_train, X_test, X_obs, imp)
                
                assert X_train_imp.shape[0] + Y_train.shape[0]
                assert X_test_imp.shape[0] + Y_test.shape[0]
                
                # first reset index
                X_train_imp = X_train_imp.reset_index(drop=True)
                Y_train = Y_train.reset_index(drop=True)
                X_test_imp = X_test_imp.reset_index(drop=True)
                Y_test = Y_test.reset_index(drop=True)
                
                # add treatment variable
                X_train_inc = pd.concat([X_train_imp, Y_train['random']], axis=1)
                X_test_inc = pd.concat([X_test_imp, Y_test['random']], axis=1)
                
                # prepare test dfs for treated and untreated situation
                X_test_treated = X_test_inc.copy()
                X_test_treated['random'] = 1
                
                X_test_untreated = X_test_inc.copy()
                X_test_untreated['random'] = 0
                
                # ADD ALL INTERACTIONS
                for variable in initial_main_effects:
                    col = variable+'*random'
                    # train set
                    X_train_inc[col] = X_train_inc.random * X_train_inc[variable]
                    
                    # test set under treatment
                    X_test_treated[col] = X_test_treated.random * X_test_treated[variable]
                    
                    # test set under no treatment
                    X_test_untreated[col] = X_test_untreated.random * X_test_untreated[variable]
                    
                # SELECT ONLY THE FEATURES INCLUDED IN THIS ITERATION
                X_train_inc = X_train_inc[features]
                
                if causal_forest:
                    X_test_inc = X_test_inc[features]

                X_test_treated = X_test_treated[features]
                X_test_untreated = X_test_untreated[features]
                
                # calculate and add ITEs
                if causal_forest:
                    ite, _ = generic_causal_forest(X_train_inc, Y_train, X_test_inc)
                else:
                    ite = generic_s_learner(X_train_inc, Y_train, X_test_treated, X_test_untreated, C, penalty)
                
                Y_test['ite'] = ite
                
                # add to stacked Y test df
                Y_test_stacked = pd.concat([Y_test_stacked, Y_test], axis=0)
                
            assert Y_test_stacked.shape[0] == Y.shape[0]
        
            print('Point estimate score:')
            score = ite_discrimination_score(Y_test_stacked, metric)
            print(score)

            # make new variable to enable stratified bootstrap samples
            Y_test_stacked['strat'] = Y_test_stacked['Survival'].astype(str) + Y_test_stacked['random'].astype(str)
            
            print('start bootstrapping')
            bootstrapped_scores = []
            
            n = 1
            if B>10:
                n = 10
            if B>100:
                n = 100
            if B>1000:
                n = 1000

            for b in range(B):

                if b%n == 0:
                    print(str(b)+'/'+str(B))

                Y_boot = resample(Y_test_stacked,
                                  stratify=Y_test_stacked.strat
                                  )
                assert Y_test_stacked.shape == Y_boot.shape
                
                bootstrapped_scores.append(ite_discrimination_score(Y_boot, metric))

            stable_score = np.median(bootstrapped_scores)
            print('\nstabalized score: ', stable_score)
            overall_score_per_candidate.append(stable_score)

            # add to boxplot dataframe    
            temp_df = pd.DataFrame()
            temp_df['SCORE'] = bootstrapped_scores
            temp_df['Candidate'] = candidate
            boxplot_data = pd.concat([boxplot_data, temp_df], axis=0)
            
            
        print('===== Looped through all candidates =====')
        
        # define highscore increase
        score_increases = [(x - highscore) for x in overall_score_per_candidate]
        max_idx = np.nanargmax(score_increases)
        highscore_increase = score_increases[max_idx]
        
        plt.figure()
        
        sns.boxplot(x="Candidate",
            y="SCORE", 
            data=boxplot_data, 
             palette="Blues")
        
        if metric == 'etd':
            plt.ylabel("ETD", size=10)
        
        elif metric == 'eqd':
            plt.ylabel("EQD", size=10)

        elif metric == 'auc_benefit':
            plt.ylabel("AUC-benefit", size=10)
        
        elif metric == 'abs_ite_lift':
            plt.ylabel("Lift [benefit]", size=10)

        else:
            plt.ylabel("c-for-benefit", size=10)
        
        plt.xlabel("Interaction term candidates", size=8)
        plt.hlines(y=highscore, xmin=0, xmax=len(left_over_candidates)+0.5, 
                   linestyle='--', linewidth=1, color='r', label='previous highscore')
        plt.hlines(y=0, xmin=0, xmax=len(left_over_candidates)+0.5, linestyle='--', linewidth=1.5, color='k')
        
        if highscore_increase > tol:
           
            plt.plot(max_idx, overall_score_per_candidate[max_idx], marker="*",
                     markersize=10, color='r',
                     label='selected')
        
        plt.xticks(rotation = 90)
        plt.legend(loc='upper left')
        plt.title('outer LORO left out study: ' + outer_loop_left_out_study + ', round: ' + str(round_number), size=10)
        plt.tight_layout()
        if main_effects_pruning:
            plt.savefig('../overall_pipeline_results/' + save_directory + 'boxplot_'+outer_loop_left_out_study+'_backward_step_round_'+str(round_number)+'.jpg', dpi=300)
        else:
            plt.savefig('../overall_pipeline_results/' + save_directory + 'boxplot_'+outer_loop_left_out_study+'_round_'+str(round_number)+'.jpg', dpi=300)
         
        
        # update highscore
        highscore = overall_score_per_candidate[max_idx]
        
        print('updated highscore:')
        print(highscore)
                    
        # check if tolarence is reached
        if highscore_increase > tol:
            if metric == 'overall':
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)*100) + '%')

            elif metric == 'auc_benefit':
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)))

            else:
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)) + '%')
            
            # add interaction with highest increase
            
            if (paired_main_effect_interaction) | (main_effects_pruning):
                pruned_main_effects.append(left_over_candidates[max_idx])

            if not main_effects_pruning:
                pruned_interactions.append(left_over_candidates[max_idx])

            print('pruned main effects:')
            print(pruned_main_effects)
            print('pruned interactions:')
            print(pruned_interactions)

            
        else:
            print('tolerance not reached')
            print('FINAL pruned main effects:')
            print(pruned_main_effects)
            print('FINAL pruned interactions:')
            print(pruned_interactions)

            break
        
    print('done')

    selected_main_effects = [x for x in initial_main_effects if x not in pruned_main_effects]
    
    selected_interactions = [x for x in initial_interactions if x not in pruned_interactions]
    selected_interactions_return = []

    for variable in selected_interactions:
        selected_interactions_return.append(variable+'*random')

    return selected_main_effects, selected_interactions_return    



def select_features(X, Y, X_obs, imp, B, C, penalty, 
                             outer_loop_left_out_study,
                             metric, 
                             save_directory,
                             direction,
                             missingness_threshold,
                             interaction_filtering=False,
                             paired_main_effect_interaction=False,
                             causal_forest=False,
                             main_effects_pruning=False):
    
    
    # select only main effects to start with
    main_effects = list(X.columns)
    
    
    # finally add the treatment variable as a candidate
    # if (paired_main_effect_interaction == False) & (causal_forest==False):
    #     candidates.append('random')
    
    initial_candidates = []

    if interaction_filtering:

        # initialize lists to keep track of p values per candidate
        p_values = []
        Filtered = []

        for var in main_effects:
            
            df_mixed = pd.read_csv('../pooled_data/update/'+outer_loop_left_out_study+'_'+ var + '_random_intercept_slope' + '.csv')
            p = df_mixed.iloc[3,3]
            p_values.append(np.round(p,2))


            if p < 0.3:
                initial_candidates.append(var)
                Filtered.append(0)
            else:
                Filtered.append(1)

        # # save df for filtering step
        # df = pd.DataFrame()
        # df['covariates'] = main_effects
        # df['p_for_interaction'] = p_values
        # df['Filtered'] = Filtered
        # df.to_excel('../overall_pipeline_results/' + save_directory + 'Filtering_table_' + outer_loop_left_out_study + '.xlsx')



    else:
        for var in main_effects:
            initial_candidates.append(var)

    


    if direction == 'forward':
        
        
        included_main_effects, included_interactions, high_score_forward_selection = basic_forward_selection(X, Y, X_obs, 
                                                                                                             main_effects, initial_candidates, 
                                                                                                            penalty, C, metric, B, imp, 
                                                                                                            outer_loop_left_out_study, 
                                                                                                            save_directory, missingness_threshold
                                                                                                                )
        
        
        

        if main_effects_pruning:
            print('\n ===== START EXTRA PRUNING STEP AFTER FORWARD SELECTION ==== ')
            included_main_effects, _ = generic_backward_pruning(X, Y, X_obs, imp, B, C, penalty,
                                                                outer_loop_left_out_study, metric,
                                                                included_main_effects, included_interactions,
                                                                save_directory, 
                                                                main_effects_pruning = main_effects_pruning,
                                                                high_score_main_effects_pruning=high_score_forward_selection,
                                                                paired_main_effect_interaction=paired_main_effect_interaction,
                                                                causal_forest=causal_forest
                                                                )
            
    elif direction == 'backward':
        
        included_main_effects, included_interactions = generic_backward_pruning(X, Y, X_obs, imp, B, C, penalty,
                                                                outer_loop_left_out_study, metric,
                                                                main_effects, main_effects,
                                                                save_directory, 
                                                                main_effects_pruning = main_effects_pruning,
                                                                paired_main_effect_interaction=paired_main_effect_interaction,
                                                                causal_forest=causal_forest
                                                                )


    return included_main_effects, included_interactions


def meta_learner_generic(data_train, Y_train, data_left_out, Y_left_out, 
                         X_obs, outcome, learner, interactions, imp, 
                         penalties=None, penalty=None, C=None, hyper_tuning=True):
    
    print('meta-learner triggered')
    
    if hyper_tuning:
        print('Tune hyper-parameter and penalization')

    else:
        print('Set penalty and hyperparam; penalty='+penalty+' , C='+str(C))
    
    # internal validation --> train datra = test data !!!
    
    # normalize and impute
    X_train_imp, X_test_imp = normalize_and_impute(data_train, data_left_out, X_obs, imp)
    
    # fit S learner
    ite_train, ite, model, optimal_C, optimal_penalty = S_learner(X_train_imp, Y_train, X_test_imp,
                                                                  learner, outcome, interactions,
                                                                  penalties, penalty, C, hyper_tuning)
    
    # add ITE predictions to Y table of lef-out set
    Y_train['ite'] = ite_train
    Y_left_out['ite'] = ite
    
        
    return Y_train, Y_left_out, optimal_C, optimal_penalty, model, X_test_imp


def meta_learner(data, Y, outcome, learner, method, interactions, penalty, C):
    
    assert method in ['S_learner', 'T_learner', 'X_learner']
    
    # internal validation --> train datra = test data !!!
    
    # normalize and impute
    X_imp, _ = normalize_and_impute(data, data)
    
    # initialize dummy if model is not returned by learner
    model = None
    optimal_C = None
    
    if method == 'S_learner':
        
        ite_train, ite, model, optimal_C, _ = S_learner(X_imp, Y, X_imp, learner, outcome, interactions, penalty, C)

    elif method == 'T_learner':
        
        ite = T_learner(X_imp, Y, X_imp, learner, outcome, penalty)
    
    
    elif method == 'X_learner':
    
        ite = X_learner(X_imp, Y, X_imp, learner, outcome, penalty)
        
        
    return ite, model, optimal_C


def meta_learner_LORO(include, data, Y, outcome, learner, method, interactions, penalty, C):

    total_ites = list()
    optimal_Cs = []
    tuning_results = None
    
    for i in include: # loop over all included RCT studies
        print(i)
        
        # split in train and test
        X_train = data[Y.STUDY!=i].reset_index(drop=True)
        X_test = data[Y.STUDY==i].reset_index(drop=True)
        
        Y_train = Y[Y.STUDY!=i].reset_index(drop=True)
        
        # normalize and impute
        X_train_imp, X_test_imp = normalize_and_impute(X_train, X_test)
        

        
        if method == 'S_learner':
            
            ite_train, ite, _, optimal_C, tuning_results = S_learner(X_train_imp, Y_train, X_test_imp, learner, outcome, interactions, penalty, C)
            optimal_Cs.append(optimal_C)
            
            # save tuning results per penalty type
            # tuning_results.to_csv('tuning_results_'+ penalty+ '_' + str(i) + '_11_01_FINE.csv')
            
        elif method == 'T_learner':
            
            ite = T_learner(X_train_imp, Y_train, X_test_imp, learner, outcome, penalty)
        
        
        elif method == 'X_learner':
          
            ite = X_learner(X_train_imp, Y_train, X_test_imp, learner, outcome, penalty)
            
        # adds ites to total ites
        total_ites = total_ites + ite
            
    return total_ites, optimal_Cs



def risk_learner(X, Y, X_test, learner, outcome, penalty, C):
    
    model = fit_classifier(X, Y[outcome], learner, penalty, C)
    
    risk_predictions = model.predict_proba(X_test)[:,1]
    
    return list(risk_predictions), model



def S_learner(X, Y, X_test, learner, outcome, interactions, penalties, penalty, C, hyper_tuning):
    
    import statsmodels.api as sm
    
    
    optimal_C = None
    tuning_results = None
    
    if (learner == 'linear') & (len(interactions)>0):
        # add interactions to train data
        for interaction in interactions:
            col = interaction+'*random'
            X[col] = X.random * X[interaction]

            
        
        
        
        # optional, drop treatment variable
        X = X.drop(['random'], axis=1)
    
        
    # if feature_forward_selection:
    #     print('==== Select features with forward selection =====')
        
        
        
    
    
    if hyper_tuning:
        
        model, optimal_C, optimal_penalty = fit_classifier_tuned(X, Y, penalties, interactions, outcome)
    else:


        #HEREEE
        X_interactions_only = X.drop(interactions, axis=1)
        model = fit_classifier(X, Y, learner, penalty, C, outcome)
        optimal_penalty = penalty
        optimal_C = C
        
    # ===== Internal validation; ITEs for train data ===========
    
    # ====  estimates under T=0 ==========
    X['random'] = 0
    
    if (learner == 'linear') & (len(interactions)>0):
        # add interactions to test data
        for interaction in interactions:
            col = interaction+'*random'
            X[col] = X.random * X[interaction]
        
        X = X.drop(['random'], axis=1)
    
    #HEREEE
    X_interactions_only = X.drop(interactions, axis=1)
    outcome_T_0 = model.predict_proba(X)[:,1]
    
    # ====  estimates under T=1 ==========
    X['random'] = 1
    
    
    
    if (learner == 'linear') & (len(interactions)>0):
        # add interactions to test data
        for interaction in interactions:
            col = interaction+'*random'
            X[col] = X.random * X[interaction]
        
        X = X.drop(['random'], axis=1)
    
    #HEREEE
    X_interactions_only = X.drop(interactions, axis=1)
    outcome_T_1 = model.predict_proba(X)[:,1]
    
    # ITE
    # logit_ite_train = list(logit(outcome_T_1) - logit(outcome_T_0))
    # ite_train = inv_logit(logit_ite_train)
    ite_train = outcome_T_1 - outcome_T_0
    
    
    # ===== External validation; ITEs for test data ===========
    # ======  mortality estimates under T=0 ==========
    X_test['random'] = 0
    
    
    
    # add interactions to test data
    for interaction in interactions:
        col = interaction+'*random'
        X_test[col] = X_test.random * X_test[interaction]
    
    if (learner == 'linear') & (len(interactions)>0):
        
        # optional, drop treatment variable
        X_test = X_test.drop(['random'], axis=1)
        
        try:
            #HEREEE
            X_test_interactions_only = X_test.drop(interactions, axis=1)
            outcome_T_0 = model.predict_proba(X_test)[:,1]
        except:
            # add constant
            X_test_dropped_inc_constant = sm.add_constant(X_test)
            #HEREEE
            X_test_interactions_only = X_test_dropped_inc_constant.drop(interactions, axis=1)
            outcome_T_0 = model.predict(X_test_dropped_inc_constant)
    else:
        outcome_T_0 = model.predict_proba(X_test)[:,1]
    
    
    # ======  mortality estimates under T=1 ==========
    X_test['random'] = 1
    
    
    
    # add interactions to test data
    for interaction in interactions:
        col = interaction+'*random'
        X_test[col] = X_test.random * X_test[interaction]
    
    if (learner == 'linear') & (len(interactions)>0):
        
        # optional, drop treatment variable
        X_test = X_test.drop(['random'], axis=1)
        
        try:
            #HEREEE
            X_test_interactions_only = X_test.drop(interactions, axis=1)
            outcome_T_1 = model.predict_proba(X_test)[:,1]
        except:
            X_test_dropped_inc_constant = sm.add_constant(X_test)
            #HEREEE
            X_test_interactions_only = X_test_dropped_inc_constant.drop(interactions, axis=1)
            outcome_T_1 = model.predict(X_test_dropped_inc_constant)
            
    else:
        outcome_T_1 = model.predict_proba(X_test)[:,1]
        
    
    
    # ======== calculate ITE in test data --> Y(T=1) - Y(T=0) ==================
    
    # logit_ite = list(logit(outcome_T_1) - logit(outcome_T_0))
    # ite = inv_logit(logit_ite)
    ite = outcome_T_0 - outcome_T_1 
    
    return ite_train, ite, model, optimal_C, optimal_penalty

def T_learner(X, Y, X_test, learner, outcome, penalty):
    
    
    # drop treatment variable 
    X = X.drop(['random'], axis=1)
    X_test = X_test.drop(['random'], axis=1)
    
    # mortality estimates under T=0
    mask = Y.random == 0
    model_T0 = fit_classifier(X[mask], Y.loc[mask,outcome], learner, penalty)
    outcome_T_0 = model_T0.predict_proba(X_test)[:,1]
    
    # mortality estimates under T=1
    mask = Y.random == 1
    model_T1 = fit_classifier(X[mask], Y.loc[mask,outcome], learner, penalty)
    outcome_T_1 = model_T1.predict_proba(X_test)[:,1]
    
    # calculate ITE in test data
    ite = list(outcome_T_1 - outcome_T_0)
    
    return ite


def X_learner(X, Y, X_test, learner, outcome, penalty):
    
    # drop treatment variable 
    X = X.drop(['random'], axis=1)
    X_test = X_test.drop(['random'], axis=1)
    
    # fit propensity model
    g = fit_classifier(X, Y.random, learner, penalty)
    
    
    # create masks
    mask_T0 = Y.random == 0
    mask_T1 = Y.random == 1
    
    # risk model under T=0
    model_T0 = fit_classifier(X[mask_T0], Y.loc[mask_T0,outcome], learner, penalty)
    
    # risk model under T=1
    model_T1 = fit_classifier(X[mask_T1], Y.loc[mask_T1,outcome], learner, penalty)
    
    # calculate ite estimates in T=1 group train data using OBSERVED OUTCOME and predicted outcome under T=0
    ites_T1 = Y.loc[mask_T1,outcome] - model_T0.predict_proba(X[mask_T1])[:,1]
    
    # calculate ite estimates in T=0 group train data using OBSERVED OUTCOME and predicted outcome under T=1
    ites_T0 = model_T1.predict_proba(X[mask_T0])[:,1] - Y.loc[mask_T0,outcome]
    
    # fit ite model using data under T=0
    regressor_T0 = fit_regressor(X[mask_T0], ites_T0, learner)
    
    # fit ite model using data under T=1
    regressor_T1 = fit_regressor(X[mask_T1], ites_T1, learner)
                
    # get final ite estimates by combining two ite models (weighted by propensity score) 
    ite = list(g.predict_proba(X_test)[:,1] * regressor_T0.predict(X_test) + g.predict_proba(X_test)[:,0] * regressor_T1.predict(X_test))
    
    return ite


def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def fit_classifier(X, Y, learner, penalty, C, outcome):
    
    import statsmodels.api as sm
    # from lightgbm import LGBMRegressor, LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    # from xgboost import XGBClassifier
    import statsmodels.api as sm
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    
    if learner == 'linear':
        
        if penalty == 'none':
            print('using no penalization ')
            
            # ==== formula api =========
            
            # formula = 'Survival ~ '
            # for covariate in X.columns:
            #     formula += (covariate + ' + ')
            # formula = formula[:-3]
            
            # data = pd.concat([X,Y],axis=1)
            # data.columns = list(X.columns) + ['Survival']
            
            # drop two columns with interaction terms (as the api handles these)
            
            # model = smf.logit(formula = str(f), data = hgc).fit()
            
            # ===== no formula api =========
            
            # X_inc_constant = sm.add_constant(X)
            # model = sm.Logit(Y, X_inc_constant).fit()
            
            
            # ======== sklearn's no penalty logistic regression ===========
            
            model = LogisticRegression(fit_intercept = True, penalty='none', solver='saga').fit(X, Y[outcome])
        else:
            
            print('using '+ penalty +  ' penalization, inverse strength = ', C)
            model = LogisticRegression(fit_intercept = True, C=C,penalty=penalty, l1_ratio=0.5, solver='saga').fit(X, Y[outcome])
            
            
    elif learner == 'RF':
        model = RandomForestClassifier(n_estimators=500, max_depth=5).fit(X, Y[outcome])
        
    elif learner == 'XGB':
        
        
        # initialize model
        model = XGBClassifier(n_estimators=500).fit(X,Y[outcome])
        
        # xgb = XGBClassifier(learning_rate=0.02, n_estimators=500, objective='binary:logistic',
        #             silent=True, nthread=1)
        
        # # hyperparameter tuning
        # param_grid = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
        #                 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
        #                 "min_child_weight" : [ 1, 3, 5, 7 ],
        #                 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
        #                 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
        
        # folds = 5
        # param_comb = 5
        
        # skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
        
        # random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )
        # random_search.fit(X, Y)
        
        # model = random_search.best_estimator_
        
        
    return model



def fit_classifier_tuned(X, Y, penalties, interactions, outcome):
    
    from sklearn.linear_model import LogisticRegression, LinearRegression
    import random
    
    # reset indices of both dataframes
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
        
    # hyperparam tuning optimizing c-for-benefit
    gridsize = 10
    grid = list(np.logspace(0, 1, gridsize))
    print('search grid for C:')
    print(grid)
    
    # make K-fold cross-validation splits    
    K = 10
    
    # initiate 'highest c-for-benefit so far'
    max_c_for_benefit = 0
    
    print('START HYPER-PARAMETER TUNING OPTIMIZING C-FOR-BENEFIT')
    
    # loop over different penalizations
    
    for penalty in penalties:
        print('evaluate penalization:', penalty)
        
        # loop over grid
        for C in grid:
            
            print('evaluate C=', C)
            
            #initiate list for performances at different CV splits
            c_for_benefits=[]
            
            # loop over cross-validations
            for i in range(K):
                
                print('fold '+ str(i) + '/' + str(K))
                
                # make random 50/50 split
                split_indices = random.sample(list(X.index), int(X.shape[0]/2))
                
                X_train = X.drop(X.index[split_indices])
                Y_train = Y.drop(Y.index[split_indices])
            
                X_val = X.iloc[split_indices,:]
                Y_val = Y.iloc[split_indices,:]
                
                # fit model with train data
                LR = LogisticRegression(fit_intercept = True, penalty=penalty, l1_ratio=0.5, solver='saga', C=C).fit(X_train, Y_train[outcome])    
                
                # create ITE estimates in validation data using fit model
                ite = S_learner_CV(X_val, interactions, LR)
                Y_val['ite'] = ite
                
                # calculate c-for-benefit using ITE estimates in validation set 
                c_for_benefits.append(c_for_benefit(X_val, Y_val))
            
            
            assert len(c_for_benefits)==K
            
            # check if new found mean c-for-benefit is higher than highest so far
            if np.mean(c_for_benefits) > max_c_for_benefit:
                
                max_c_for_benefit = np.mean(c_for_benefits)
                optimal_C = C
                optimal_penalty = penalty
    
    
    print('optimal penalization: ', optimal_penalty)
    print('optimal C (inverse regularization strength): ', optimal_C)
    

    # fit model on whole set using optimal penalization and strength
    model = LogisticRegression(fit_intercept = True, penalty=optimal_penalty, l1_ratio=0.5, solver='saga', C=optimal_C).fit(X, Y[outcome])
    
    
    return model, optimal_C, optimal_penalty


        

def fit_regressor(X, Y, learner):
    
    from lightgbm import LGBMRegressor, LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    import statsmodels.api as sm
    
    
    if learner == 'linear':
        model = LinearRegression(fit_intercept = True).fit(X, Y)
                
    elif learner == 'RF':
        model = LinearRegression(fit_intercept = True).fit(X, Y)
        # model = RandomForestRegressor().fit(X, Y)
        
    elif learner == 'XGB':
        model = LinearRegression(fit_intercept = True).fit(X, Y)
        # model = LGBMRegressor().fit(X, Y)
        
    return model


def calc_absolute_benefit(Y):
    
    # calculate observed benefit
    
    E = Y[Y.random==1]
    C = Y[Y.random==0]
    
    # print(E.shape)
    # print(C.shape)
    
    E_survival_rate = E.Survival.sum()/E.shape[0]
    C_survival_rate = C.Survival.sum()/C.shape[0]
    
    # print(E_survival_rate)
    # print(C_survival_rate)
    
    absolute_benefit = E_survival_rate - C_survival_rate
    
    return absolute_benefit*100

def plot_ite_calibration(Y, save_name, B, directory):
    
    
    ite = Y.ite*100
    

    predicted = [] # list if lists with ITE predictions
    observed_boot = [] # list of lists with bootstrapped oberved benefits
    observed_point = [] # list of lists with point estimate benefits
    n_samples = [] # list woth sample sizes per quantile
    n_events = [] # list with number of events per quartile
    
    plt.figure()
    
    # ==== Q1 ======
    mask = (ite < np.percentile(ite,25))
    predicted.append(list(ite[mask].values))
    
    Y_quartile = Y[mask].reset_index(drop=True)
    
    # print(Y_quartile.shape)
    # print(Y_quartile.random.value_counts())
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
    
    # ==== Q2 ========
    mask = ((ite >= np.percentile(ite,25)) & (ite < np.percentile(ite,50)))
    predicted.append(list(ite[mask].values))
    
    Y_quartile = Y[mask].reset_index(drop=True)
    # print(Y_quartile.random.value_counts())
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
    
    # ==== Q3 ========
    mask = ((ite >= np.percentile(ite,50)) & (ite < np.percentile(ite,75)))
    predicted.append(list(ite[mask].values))
    
    Y_quartile = Y[mask].reset_index(drop=True)
    print(Y_quartile.random.value_counts())
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
    
    # ==== Q4 =========
    mask = (ite >= np.percentile(ite,75))
    predicted.append(list(ite[mask].values))
    
    Y_quartile = Y[mask].reset_index(drop=True)
    # print(Y_quartile.random.value_counts())
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
     
    # the list named ticks, summarizes or groups
    # the summer and winter rainfall as low, mid
    # and high
    
    ticks = [
             '1 \n(n='+str(n_samples[0]) + ')',
             '2 \n(n='+str(n_samples[1]) + ')',
             '3 \n(n='+str(n_samples[2]) + ')',
             '4 \n(n='+str(n_samples[3]) + ')',
             ]
     
    
    
    
    # predicted ITE plots (violins)
    pos = np.array(np.arange(len(predicted)))*2.0-0.35
    
    violin_parts = plt.violinplot(predicted, pos, 
                   # points=60,
                   widths=0.7, 
                   # showmeans=True,
                     # showextrema=True, 
                     showmedians=True, 
                     bw_method='scott',
                     # quantiles=[[0.1], [], [], [0.175, 0.954], [0.75], [0.25]]
                     
                     )
    
    for pc in violin_parts['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('black')
    
    violin_parts['bodies'][1].set_label('Distribution of predicted ITEs')
    
    # observed plots (95% CIs)
    
    pos = np.array(np.arange(len(predicted)))*2.0+0.35
    count=0
    for i in range(len(predicted)):
        if count==0:
            plt.plot(pos[i],observed_point[i],'o',color = 'r', label='Observed mortality reduction (95% CI)')
        else:
            plt.plot(pos[i],observed_point[i],'o',color = 'r')
            
        plt.plot(
            [pos[i],pos[i]],
            [np.percentile(observed_boot[i],2.5),np.percentile(observed_boot[i],97.5)],
            color='k',linewidth=0.6)
    
        count +=1
    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
     
    
    
    # set the limit for x axis
    
    plt.xlim(-2, len(ticks)*2)
    
    
    
    
    plt.xlabel('ITE Quartile')
    
    
    plt.ylabel(u'%', fontsize=8)
    lowest_value = np.min(ite)
    highest_value = np.max(ite)
    # plt.ylim(lowest_value - 6, highest_value + 2)
    plt.ylim(-15, 15)
    
    plt.legend(loc='upper left', 
           prop={'size': 11},
        ncols=1,
        )
    
    # set the title
    
    
    # plot line through 0
    plt.plot([-3,8],[0,0], color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(directory+'/Cali_tertiles' + save_name+'.jpg',dpi=300)
    


#Defining the bell shaped kernel function - used for plotting later on
def kernel_function(xi,x0,tau= .005): 
    return np.exp( - (xi - x0)**2/(2*tau)   )

def lowess_bell_shape_kern(x, y, tau = .005):
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest
    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The kernel function is the bell shaped function with parameter tau. Larger tau will result in a
    smoother curve. 
    """
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([np.exp(- (x - x[i])**2/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return list(yest)


def plot_smooth_calibration(Y, save_name, B, directory):
    
    ite = Y.ite*100

    predicted = [] # list if lists with ITE predictions
    mean_predicted = []
    
    observed_boot = [] # list of lists with bootstrapped oberved benefits
    observed_point = [] # list of lists with point estimate benefits
    n_samples = [] # list woth sample sizes per quantile
    n_events = [] # list with number of events per quartile
    
    
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 4), (2, 0),colspan=4)
    ax3.set_yscale("log")
    
    # ==== Q1 ======
    mask = (ite < np.percentile(ite,33.3))
    predicted.append(list(ite[mask].values))
    
    Y_quartile = Y[mask].reset_index(drop=True)
    
    mean_predicted.append(Y_quartile.ite.mean())
    
    # print(Y_quartile.shape)
    # print(Y_quartile.random.value_counts())
    
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
    
    # ==== Q2 ========
    mask = ((ite >= np.percentile(ite,33.3)) & (ite < np.percentile(ite,66.7)))
    predicted.append(list(ite[mask].values))
    
    Y_quartile = Y[mask].reset_index(drop=True)
    mean_predicted.append(Y_quartile.ite.mean())
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
    
    # ==== Q3 =========
    mask = (ite >= np.percentile(ite,66.7))
    predicted.append(list(ite[mask].values))
    
    Y_quartile = Y[mask].reset_index(drop=True)
    mean_predicted.append(Y_quartile.ite.mean())
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
    

    
    # the list named ticks, summarizes or groups
    # the summer and winter rainfall as low, mid
    # and high
    
    
    # plot error bars observed ITE (violins)
    for i in range(len(predicted)):
        
        if i ==0:
            ax1.plot(mean_predicted[i]*100,
                     observed_point[i],
                     'o',color = 'r', label='Observed treatment effect in tertile')
        else:
            ax1.plot(mean_predicted[i]*100,
                     observed_point[i],
                     'o',color = 'r')
        ax1.plot(
            [mean_predicted[i]*100,mean_predicted[i]*100],
            [np.percentile(observed_boot[i],2.5),np.percentile(observed_boot[i],97.5)],
            color='k',linewidth=0.6)
        
        if i ==0:
            ax2.plot(mean_predicted[i]*100,
                     observed_point[i],
                     'o',color = 'r', label='Observed treatment effect in tertile')
        else:
            ax2.plot(mean_predicted[i]*100,
                     observed_point[i],
                     'o',color = 'r')
            
        ax2.plot(
            [mean_predicted[i]*100,mean_predicted[i]*100],
            [np.percentile(observed_boot[i],2.5),np.percentile(observed_boot[i],97.5)],
            color='k',linewidth=0.6)
        
    # ===== make pairswise observed benefits using pairing with ITEs  ========
    # first balance treatment arms by randomly selecting patients
    # Y_balanced = balance_treatment_arms(X, Y)
    Y_balanced = balance_treatment_arms(Y)
    
    # X_balanced = X_balanced.reset_index(drop=True)
    Y_balanced = Y_balanced.reset_index(drop=True)
    
    # split experiment and control group
    # X_E = X_balanced[Y_balanced.random==1].reset_index(drop=True)
    # X_C = X_balanced[Y_balanced.random==0].reset_index(drop=True)
    
    Y_E = Y_balanced[Y_balanced.random==1].reset_index(drop=True)
    Y_C = Y_balanced[Y_balanced.random==0].reset_index(drop=True)
    
    
    df_pair = pair_instances(Y_E, Y_C) #make matched df
    # sort by pairwise observed benefits
    df_pair = df_pair.sort_values(by='predicted_ite')
    
    # ====== confidence intervals via bootstrapping ============
    
    
    mean = np.linspace(-1, 1, 100)
    pred = list(df_pair.predicted_ite.values)
    obs = list(df_pair.true_ite)
    y_boot = []
    
    
    for b in range(B):
        if b%10 == 0:
            print('Bootstrap: '+ str(b) + '/' + str(B))
        
        pred_boot, obs_boot = resample(pred, obs, 
                                        stratify=obs, 
                                       random_state=b)
        
        df_boot = pd.DataFrame()
        df_boot['pred'] = pred_boot
        df_boot['obs'] = obs_boot
        df_boot = df_boot.sort_values(by='pred')
            
        y_sample = lowess_bell_shape_kern(list(df_boot.pred.values), list(df_boot.obs.values))
        
        # ax1.plot(df_boot.pred, y_sample, color='k', linewidth=0.5)
        
        y_interp = np.interp(mean, df_boot.pred, y_sample)
        y_boot.append(y_interp)
        
    y_lower = np.percentile(y_boot,2.5,axis=0)
    y_upper = np.percentile(y_boot, 97.5,axis=0)
    
    
    # ====== point estimate ============
    y = lowess_bell_shape_kern(pred, obs)
    
    
    # ==== PLOT =======
    # smooth line through observed and predicted benefits
    print(len(pred))
    print(len(y))
    
    print(len(mean))
    print(len(y_lower))
    print(len(y_upper))
    
    plot_pred = [i * 100 for i in pred]
    plot_y = [i * 100 for i in y]
    plot_mean = [i * 100 for i in mean]
    plot_y_lower = [i * 100 for i in y_lower]
    plot_y_upper = [i * 100 for i in y_upper]
    
    
    ax1.plot(plot_pred, plot_y, color='b', label='LOWESS smoothed ITE predictions')
    
    ax2.plot(plot_pred, plot_y, color='b', label='LOWESS smoothed ITE predictions (ZOOM-IN)')
   
    
    ax1.fill_between(plot_mean, plot_y_lower, plot_y_upper, color='b', alpha=.2,
                       where=(mean>=np.min(pred))&(mean<=np.max(pred))
                    # label=r'$\pm$ 1 std. dev.'
                    )
    
    ax2.fill_between(plot_mean, plot_y_lower, plot_y_upper, color='b', alpha=.2,
                       where=(mean>=np.min(pred))&(mean<=np.max(pred))
                    # label=r'$\pm$ 1 std. dev.'
                    )
    
    # plot line for perfect calibration
    ax1.plot([-100,100],[-100,100], color='k', linestyle='--', linewidth=0.5, label='Perfect calibration')
    ax2.plot([-100,100],[-100,100], color='k', linestyle='--', linewidth=0.5, label='Perfect calibration')
    
    
    
    zoom_x_min = -0.10
    zoom_x_max = 0.10
    zoom_y_min = -0.10
    zoom_y_max = 0.10
    
    
    # plot histogram of predicted ITEs
    frac_in_zoom = np.round(df_pair[(df_pair.predicted_ite >= zoom_x_min) & (df_pair.predicted_ite <= zoom_x_max)].shape[0]/df_pair.shape[0]*100, 2)
    ax3.hist(ite, bins=100,linestyle='-',linewidth = 1.5,
              label = 'Treatment effect predictions: '+ str(frac_in_zoom) + '% within ['+ str(int(zoom_x_min*100))+';'+ str(int(zoom_x_max*100))+'%] range',
              histtype="step", lw=2,color='b')
    
    
    
    
    ax3.axvline(x=zoom_x_min*100, ymin=0,ymax=1000,linestyle='--',color='r', label='zoom-in zone')
    ax3.axvline(x=zoom_x_max*100, ymin=0,ymax=1000,linestyle='--',color='r')
    
    # plot boundary for zoom-in
    
    
    
    ax1.plot([zoom_x_min,zoom_x_max],[zoom_y_min,zoom_y_min], color='r', linestyle='--', linewidth=1.2, label='Zoom-in zone')
    ax1.plot([zoom_x_min,zoom_x_max],[zoom_y_max,zoom_y_max], color='r', linestyle='--', linewidth=1.2)
    ax1.plot([zoom_x_min,zoom_x_min],[zoom_y_min,zoom_y_max], color='r', linestyle='--', linewidth=1.2)
    ax1.plot([zoom_x_max,zoom_x_max],[zoom_y_min,zoom_y_max], color='r', linestyle='--', linewidth=1.2)
    
    
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    
    ax2.set_xlim(zoom_x_min*100, zoom_x_max*100)
    ax2.set_ylim(zoom_y_min*100, zoom_y_max*100)
    
    ax1.set_ylabel('Observed pairwise treatment effect (%)')
    ax3.set_ylabel('Counts')
    
    # add other metrics as text
    # ax1.text(0.24, -0.15, 'c-for-benefit:'+str(np.round(c_for_benefit,3)))
    
    
    ax1.legend()
    ax2.legend()
    
    plt.xlabel('Predicted treatment effect')
    
    

    
    # set the title
    # plt.title('Left-out study: '+left_out_study+ ', c-for-benefit='+str(np.round(c_for_benefit,3)) + ', ETD='+str(np.round(etd,3)))
    
    
    
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory+'/Cali_smooth_' + save_name +'.jpeg',dpi=300)
    

def plot_ite_calibration_2_groups(Y, save_name, B, c_for_benefit, directory):
    import statsmodels.formula.api as smf
    
    # initialize fenotype variable
    Y['feno'] = 0
    
    ite = Y.ite

    predicted = [] # list if lists with ITE predictions
    observed_boot = [] # list of lists with bootstrapped oberved benefits
    observed_point = [] # list of lists with point estimate benefits
    n_samples = [] # list woth sample sizes per quantile
    n_events = [] # list with number of events per quartile
    
    plt.figure()
    
    # ==== predicted harm group ======
    mask = ite <= 0
    
    predicted.append(list(ite[mask].values))
    
    Y_quartile = Y[mask].reset_index(drop=True)
    
    # print(Y_quartile.shape)
    # print(Y_quartile.random.value_counts())
    
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
    
    # ==== predicted benefit group ========
    mask = ite > 0
    predicted.append(list(ite[mask].values))
    
    Y.loc[mask,'feno'] = 1
    
    Y_quartile = Y[mask].reset_index(drop=True)
    # print(Y_quartile.random.value_counts())
    
    # bootstrap observed benefit
    bootstrapped_observed = []
    for b in range(B):
        Y_boot = resample(Y_quartile, stratify=Y_quartile.random, replace=True)
        bootstrapped_observed.append(calc_absolute_benefit(Y_boot))
    
    observed_boot.append(bootstrapped_observed)
    observed_point.append(calc_absolute_benefit(Y_quartile))
    n_samples.append(int(Y_quartile.shape[0]))
    n_events.append(int(Y_quartile.Mort.sum()))
    
    # the list named ticks, summarizes or groups
    # the summer and winter rainfall as low, mid
    # and high
    
    ticks = [
             'Harm \n n='+str(n_samples[0]) + '\n ' + str(n_events[0]) + ' events',
             'Benefit \n n='+str(n_samples[1]) + '\n ' + str(n_events[1]) + ' events',
             # '3 \n n='+str(n_samples[2]) + '\n ' + str(n_events[2]) + ' events',
             # '4 \n n='+str(n_samples[3]) + '\n ' + str(n_events[3]) + ' events',
             ]
     
    
    
    
    # predicted ITE plots (violins)
    pos = np.array(np.arange(len(predicted)))*2.0-0.35
    
    violin_parts = plt.violinplot(predicted, pos, 
                   # points=60,
                   widths=0.7, 
                   # showmeans=True,
                     # showextrema=True, 
                      showmedians=True, 
                     bw_method='silverman',
                     # quantiles=[[0.1], [], [], [0.175, 0.954], [0.75], [0.25]]
                     
                     )
    
    for pc in violin_parts['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('black')
    
    
    
    # observed plots (95% CIs)
    
    pos = np.array(np.arange(len(predicted)))*2.0+0.35
    
    for i in range(len(predicted)):
        
        plt.plot(pos[i],observed_point[i],'o',color = 'r')
        plt.plot(
            [pos[i],pos[i]],
            [np.percentile(observed_boot[i],2.5),np.percentile(observed_boot[i],97.5)],
            color='k',linewidth=0.6)
    
    
    
    
    # p value for two fenotypes
    
    families = [sm.families.Binomial(), sm.families.Poisson(), sm.families.Gamma(), sm.families.InverseGaussian()]

    f = families[0]
    formula = 'Mort ~ feno*random'  
    model = smf.glm(formula = formula, data=Y, family=f)
    result = model.fit()
    effect = result.params[3]
    p = result.pvalues[3]
    
    plt.text(1.5, -0.10, 'P='+str(np.round(p,4)))
    
    
    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
     
    
    
    # set the limit for x axis
    plt.xlim(-2, len(ticks)*2)
    
    
    # set the limit for y axis
    low = np.percentile([i for lis in predicted for i in lis],1)
    high = np.percentile([i for lis in predicted for i in lis],99)
    # plt.ylim(low-0.02, high+0.02)
    
    plt.ylim(-0.2, 0.1)
    
    plt.xlabel('Predicted benefit/harm')
    plt.ylabel('Observed benefit (ARR)')
    

    
    # set the title
    plt.title(save_name + ', c-for-benefit='+str(np.round(c_for_benefit,3)))
    
    # plot line through 0
    plt.plot([-3,8],[0,0], color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(directory+'/Cali_dicho_' + save_name +'.jpeg',dpi=300)

def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()


def ETD(Y):
    # print(np.percentile(Y.ite,33.3333))
    # seperate first tertile
    mask = Y.ite < np.percentile(Y.ite,33.3333)
    t1 = Y[mask] 
    
    # seperate third tertile
    # print(np.percentile(Y.ite,66.6666))
    mask = Y.ite >= np.percentile(Y.ite,66.6666)
    t3 = Y[mask] 
    
    observed_benefit_t1 = calc_absolute_benefit(t1)
    observed_benefit_t3 = calc_absolute_benefit(t3)
    
    # print(observed_benefit_t1)
    # print(observed_benefit_t3)
    
    
    ETD = observed_benefit_t3 - observed_benefit_t1
    
    return ETD

def calculate_abs_ite_lift(Y):
    
    # seperate first tertile
    mask = Y.ite <= 0
    predicted_harm = Y[mask] 
    
    # seperate third tertile
    mask = Y.ite > 0
    predicted_benefit = Y[mask] 
    
    observed_harm = calc_absolute_benefit(predicted_harm)
    observed_benefit = calc_absolute_benefit(predicted_benefit)
    
    abs_ite_lift = observed_benefit - observed_harm
    
    return abs_ite_lift

def calculate_population_benefit(Y):

    # create dummy variable 'agree'
    Y['agree'] = 0

    Y.loc[(Y.ite > 0) & (Y.random == 1), 'agree'] = 1
    Y.loc[(Y.ite < 0) & (Y.random == 0), 'agree'] = 1
    
    agree = Y[Y.agree == 1]
    disagree = Y[Y.agree == 0]

    mortality_rate_agree = agree.Mort.sum() / agree.shape[0]
    mortality_rate_disagree = disagree.Mort.sum() / disagree.shape[0]
    
    population_benefit = (mortality_rate_disagree - mortality_rate_agree)*100

    return population_benefit

def EQD(Y):
    
    # seperate first tertile
    mask = Y.ite < np.percentile(Y.ite,25)
    q1 = Y[mask] 
    
    # seperate third tertile
    mask = Y.ite >= np.percentile(Y.ite,75)
    q4 = Y[mask] 
    
    observed_benefit_q1 = calc_absolute_benefit(q1)
    observed_benefit_q4 = calc_absolute_benefit(q4)
    
    EQD = observed_benefit_q4 - observed_benefit_q1
    
    return EQD

def S_learner_CV(X_test, interactions, model):

    # mortality estimates under T=0
    X_test['random'] = 0
    
    # add interactions to test data
    for interaction in interactions:
        col = interaction+'*random'
        X_test[col] = X_test.random * X_test[interaction]
        
        X_test_dropped = X_test.drop(['random'], axis=1)
        
        outcome_T_0 = model.predict_proba(X_test_dropped)[:,1]    
    
        
    # mortality estimates under T=1
    X_test['random'] = 1
    
    # add interactions to test data
    for interaction in interactions:
        col = interaction+'*random'
        X_test[col] = X_test.random * X_test[interaction]
        
        X_test_dropped = X_test.drop(['random'], axis=1)
        
        outcome_T_1 = model.predict_proba(X_test_dropped)[:,1]

    
    # calculate ITE in test data
    ite = list(outcome_T_1 - outcome_T_0)

    return ite  

def select_main_effects(a_priori_selection, 
                        X_combined, 
                        X_test, 
                        missingness_threshold,
                        
                        ):
    
    # print('\nSelect main effects function triggered')
    

    # initialize
    selected_main_effects = []
    

    for covariate in a_priori_selection:
        # print(covariate)
        # initialize flags for criteria to include covariate
        enough_data_in_left_out = False
        enough_data_in_train = False
        
        # check missingness in left-out set (only if this set is there)
        
        missingness = X_test[covariate].isna().sum()/X_test.shape[0]
        # print(missingness)
        if missingness < missingness_threshold:
            enough_data_in_left_out = True    
            
        
        # check missingness in train set combined with observational set
        missingness = X_combined[covariate].isna().sum()/X_combined.shape[0]
        # print(missingness)
        if missingness < missingness_threshold:
            enough_data_in_train = True    
        
        # check if both criteria are met, then include
        if enough_data_in_left_out & enough_data_in_train:
            selected_main_effects.append(covariate)
        
    # print('done')
    # print('selected main effects:')
    # print(selected_main_effects)

    
    return selected_main_effects


def tune_lambdas(X, 
                Y, 
                X_obs, 
                imp, 
                B, 
                elastic_net_value,
                left_out_study,
                metric,
                save_directory,
                missingness_threshold,
                
                ):
    
    
    study_dict = {0:'Meijvis', 1:'Endeman', 2:'Blum', 3:'Wittermans', 
                  4:'Torres', 5:'Snijders', 6:'Confalonieri', 7: 'Fernández-Serrano'}
    
    studies_inner_loro = Y.STUDY.unique()
    
    
    print('\nusing metric: ', metric)
    print('\nusing '+ str(elastic_net_value) +  ' penalization ')
    
    print('\nInner LOTO loop studies: ')
    for study in studies_inner_loro:
        print(study_dict[study])
        


    print('\nstart loop lambda combination')
    
    n_steps = 30
    print('n steps in grid search:')
    print(n_steps)
    
    # range_main = list(np.logspace(-1, -0.3, num=n_steps))
    range_main = list(np.logspace(-3, -1, num=n_steps))
    # range_main = [0.8]
    print(range_main)
    
    
    range_interactions = list(np.logspace(-3, -1, num=n_steps))
    # range_interactions = list(np.logspace(-3, 1, num=n_steps))
    # range_interactions = [0.0268]
    print(range_interactions)
    
    high_score = -1000 # set highscore super low to initialize
    full_list = []
    full_pair_list = []

    for lambda_1 in range_main:

        overall_score_per_candidate_pair = []
        pair_list = []

        for lambda_2 in range_interactions:
            
            print("\nlambda 1: " + str(lambda_1))
            print("lambda 2: " + str(lambda_2))

            pair_list.append((lambda_1, lambda_2))

            print('\nstart inner LOTO-CV')

            Y_test_stacked = pd.DataFrame()
            
            # initialize flag to spot a lambda combination where all treatment variables are shrunk to zero
            all_zeros = False

            # ======= LOOP 3 ; inner LOTO to evaluate candidate ===========
            for study in studies_inner_loro:
                
                print('\nleft out study inner LOTO:')
                print(study_dict[study])
                
                # train data from original df
                mask = Y.STUDY != study
                X_train = X[mask]
                Y_train = Y[mask]
                
                # test data from modified dfs (untreated and treated)
                mask = Y.STUDY == study
                X_test = X[mask]
                Y_test = Y[mask]
                
                assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
                assert Y_train.shape[0] + Y_test.shape[0] == Y.shape[0]
                
                # stack train set and observational set
                X_combined = pd.concat([X_train, X_obs], axis=0)

                # select main effects in inner loop based on availability
                selected_main_effects = select_main_effects(
                                                            X_train.columns, 
                                                            X_combined, 
                                                            X_test, 
                                                            missingness_threshold,
                                                            )

                # select the variables with sufficient data in train and lef-out datasets
                X_train_selection = X_train[selected_main_effects]
                X_test_selection = X_test[selected_main_effects]
                X_obs_selection = X_obs[selected_main_effects]


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
                
                


                # ADD INTERACTIONS
                print('\nADD INTERACTIONS')
                for variable in selected_main_effects:
                # for variable in ['crp']:
                    
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


                n_features = len(selected_main_effects)

                # add intercepts
                X_train_norm.insert(0, 'intercept', np.ones(X_train_norm.shape[0]))
                X_test_treated_norm.insert(0, 'intercept', np.ones(X_test_treated_norm.shape[0]))
                X_test_untreated_norm.insert(0, 'intercept', np.ones(X_test_untreated_norm.shape[0]))

                # fit a LASSO/RIDGE model with different penalization strengths
                model = sm.GLM(Y_train.Mort, X_train_norm, family=sm.families.Binomial())
                
                # penalties =  [lambda_1] + [lambda_2] * (2*n_features - 1)

                penalties = [0] + [lambda_1] * n_features + [lambda_2] * (n_features - 1)

                # penalties =  [lambda_intercept] + [lambda_1] * n_features + [lambda_2] * (n_features -1)
                # penalties = [lambda_1] + [lambda_2] * (2*n_features -1)
                # penalties =  [lambda_1] * n_features + [lambda_2] * (n_features -1)
                # penalties = [lambda_1] * (n_features - 1) + [lambda_2] * n_features

                print('THEREE')
                print(penalties)
                assert len(penalties) == X_train_norm.shape[1] == X_test_treated_norm.shape[1] == X_test_untreated_norm.shape[1]
                results = model.fit_regularized(alpha=penalties, L1_wt=elastic_net_value)

                print(results.params)
                print(results.params[0])
                print(results.params[0] == 0)

                if results.params[0] == 0:
                    print('intercept is shrunk to zero')
                    all_zeros = True
                

                # calculate and add ITEs for test data
                y_pred_treated = results.predict(X_test_treated_norm)
                y_pred_untreated = results.predict(X_test_untreated_norm)
                ite = list(y_pred_untreated - y_pred_treated)
                Y_test['ite'] = ite

                # if 
                if (Y_test.ite==0).sum() == Y_test.shape[0]:
                    all_zeros = True
                    
                # add to stacked Y test df
                Y_test_stacked = pd.concat([Y_test_stacked, Y_test], axis=0)

                # END of inner LOTO loop
            print('inner LOTO-CV done')

            print(Y_test_stacked.ite)
            print('n predicted benefit:')
            print(Y_test_stacked[Y_test_stacked.ite>0].shape)
            print('n predicted harm:')
            print(Y_test_stacked[Y_test_stacked.ite<=0].shape)
            
            if all_zeros:
                print('\nOnly zero ITEs in at least one fold of inner LORO --> no estimate')
                overall_score_per_candidate_pair.append(np.nan)
            else:

                print('Point estimate score:')
                score = ite_discrimination_score(Y_test_stacked, metric)
                print(score)

                # make new variable to enable stratified bootstrap samples
                Y_test_stacked['strat'] = Y_test_stacked['Survival'].astype(str) + Y_test_stacked['random'].astype(str)
                
                print('start bootstrapping')
                bootstrapped_scores = []
                
                n = 1
                if B>10:
                    n = 10
                if B>100:
                    n = 100
                if B>1000:
                    n = 1000

                for b in range(B):

                    if b%n == 0:
                        print(str(b)+'/'+str(B))

                    Y_boot = resample(Y_test_stacked,
                                    stratify=Y_test_stacked.strat
                                    )
                    assert Y_test_stacked.shape == Y_boot.shape
                    
                    bootstrapped_scores.append(ite_discrimination_score(Y_boot, metric))

                stable_score = np.median(bootstrapped_scores)
                print('\nstabalized score: ', stable_score)
                
                overall_score_per_candidate_pair.append(stable_score)
                

                if stable_score > high_score:

                    # first update highscore
                    high_score = stable_score
                    
                    # then update optimal pair
                    optimal_pair = (lambda_1, lambda_2) 


        full_list.append(overall_score_per_candidate_pair)
        full_pair_list.append(pair_list)

    print('FINISHED GRID SEARCH')
    print('optimal pair:')
    print(optimal_pair)

    df_pair = pd.DataFrame(full_pair_list)

    df = pd.DataFrame(full_list)
    df.index = [str(a) for a in range_main]
    df.columns = [str(a) for a in range_interactions]

    
    # Create the heatmap
    plt.figure()
    
    sns.heatmap(df, cmap='coolwarm',
                cbar_kws={'label': 'd-benefit'},
                # annot=True
                vmin=-3, vmax=8
                )
    
    #annotate max value
    max_value = np.nanmax(df.values)
    print(max_value)
    for k in range(len(df)):

        for j in range(len(df.columns)):
            if df.iloc[k, j] == max_value:
                print('found')
                x = j
                y = k    
    plt.gca().add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='k', lw=1.5))

    plt.ylabel('\u03BB intercept')
    plt.xlabel('\u03BB predictors')
    # plt.title('\u03BB intercept = ' + str(lambda_intercept))
    plt.tight_layout()
    plt.savefig('../overall_pipeline_results/' + save_directory + 'TUNE_HEATMAP_' + left_out_study + '.jpg', dpi=300)


    df_pair.to_excel('../overall_pipeline_results/' + save_directory + 'df_pairs_' + left_out_study + '.xlsx')
    df.to_excel('../overall_pipeline_results/' + save_directory + 'df_scores_' + left_out_study + '.xlsx')
    
    return optimal_pair

def tune_single_lambda(X, 
                Y, 
                X_obs, 
                imp, 
                B, 
                elastic_net_value,
                left_out_study,
                metric,
                save_directory,
                missingness_threshold,
                remove_intercept
                ):
    
    
    study_dict = {0:'Meijvis', 1:'Endeman', 2:'Blum', 3:'Wittermans', 
                  4:'Torres', 5:'Snijders', 6:'Confalonieri', 7: 'Fernández-Serrano'}
    
    studies_inner_loro = Y.STUDY.unique()
    
    
    print('\nusing metric: ', metric)
    print('\nusing '+ str(elastic_net_value) +  ' penalization ')
    
    print('\nInner LOTO loop studies: ')
    for study in studies_inner_loro:
        print(study_dict[study])
        
    print('\nstart loop lambda combination')
    
    n_steps = 200
    print('n steps in grid search:')
    print(n_steps)
    lambda_range = list(np.logspace(-6, 3, num=n_steps))
    print(lambda_range)
    high_score = -1000 # set highscore super low to initialize
    
    score_list = []
    bootstrapped_score_list = []
    bootstrap_df = pd.DataFrame()

    for lambda_candidate in lambda_range:

        print("lambda: " + str(lambda_candidate))
        
        print('\nstart inner LOTO-CV')

        Y_test_stacked = pd.DataFrame()

        # initialize flag to spot a lambda combination where all treatment variables are shrunk to zero
        all_zeros = False

        # ======= LOOP 3 ; inner LOTO to evaluate candidate ===========
        for study in studies_inner_loro:
            
            print('\nleft out study inner LOTO:')
            print(study_dict[study])
            
            # train data from original df
            mask = Y.STUDY != study
            X_train = X[mask]
            Y_train = Y[mask]
            
            # test data from modified dfs (untreated and treated)
            mask = Y.STUDY == study
            X_test = X[mask]
            Y_test = Y[mask]
            
            assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
            assert Y_train.shape[0] + Y_test.shape[0] == Y.shape[0]
            
            # stack train set and observational set
            X_combined = pd.concat([X_train, X_obs], axis=0)

            # select main effects in inner loop based on availability
            selected_main_effects = select_main_effects(
                                                        X_train.columns, 
                                                        X_combined, 
                                                        X_test, 
                                                        missingness_threshold,
                                                        )

            # select the variables with sufficient data in train and lef-out datasets
            X_train_selection = X_train[selected_main_effects]
            X_test_selection = X_test[selected_main_effects]
            X_obs_selection = X_obs[selected_main_effects]


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
            
            
            
            # ADD INTERACTIONS
            print('\nADD INTERACTIONS')
            for variable in selected_main_effects:
            # for variable in ['crp']:
                
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


            
            
            n_features = X_train_norm.shape[1]
            

            if not remove_intercept:
                # add intercepts
                X_train_norm.insert(0, 'intercept', np.ones(X_train_norm.shape[0]))
                X_test_treated_norm.insert(0, 'intercept', np.ones(X_test_treated_norm.shape[0]))
                X_test_untreated_norm.insert(0, 'intercept', np.ones(X_test_untreated_norm.shape[0]))

            
            if remove_intercept:
                penalties = [lambda_candidate]*n_features
            else:
                penalties = [0] + [lambda_candidate]*n_features

            assert len(penalties) == X_train_norm.shape[1] == X_test_treated_norm.shape[1] == X_test_untreated_norm.shape[1]
            

            
            model = sm.GLM(Y_train.Mort, X_train_norm, family=sm.families.Binomial())

            print('penalties:')
            print(penalties)
            results = model.fit_regularized(alpha=penalties, L1_wt=elastic_net_value)
            print('THEREE')
            print(results.params)

            if results.params[0] == 0:
                print('intercept is shrunk to zero')
                all_zeros = True
                
            
            # calculate and add ITEs for test data
            y_pred_treated = results.predict(X_test_treated_norm)
            y_pred_untreated = results.predict(X_test_untreated_norm)
            ite = list(y_pred_untreated - y_pred_treated)
            Y_test['ite'] = ite

            # if 
            if (Y_test.ite==0).sum() == Y_test.shape[0]:
                all_zeros = True
                
            # add to stacked Y test df
            Y_test_stacked = pd.concat([Y_test_stacked, Y_test], axis=0)

            # END of inner LOTO loop
        print('inner LOTO-CV done')

        print(Y_test_stacked.ite)
        
        
        print('stacked test set:')
        print(Y_test_stacked[Y_test_stacked.ite>0].shape)
        print(Y_test_stacked[Y_test_stacked.ite<=0].shape)
        print(Y_test_stacked[Y_test_stacked.ite==0].shape)

        if all_zeros:
                print('\nOnly zero ITEs in at least one fold of inner LORO --> no estimate')
                score_list.append(np.nan)
        else:

            print('Point estimate score:')
            score = ite_discrimination_score(Y_test_stacked, metric)
            print(score)



            # make new variable to enable stratified bootstrap samples
            Y_test_stacked['strat'] = Y_test_stacked['Survival'].astype(str) + Y_test_stacked['random'].astype(str)
            
            print('start bootstrapping')
            bootstrapped_scores = []
            
            n = 1
            if B>10:
                n = 10
            if B>100:
                n = 100
            if B>1000:
                n = 1000

            for b in range(B):

                if b%n == 0:
                    print(str(b)+'/'+str(B))

                Y_boot = resample(Y_test_stacked,
                                stratify=Y_test_stacked.strat
                                )
                assert Y_test_stacked.shape == Y_boot.shape
                
                bootstrapped_scores.append(ite_discrimination_score(Y_boot, metric))

            stable_score = np.median(bootstrapped_scores)
            print('\nstabalized score: ', stable_score)
            
            score_list.append(stable_score)
            bootstrapped_score_list.append(bootstrapped_scores)

            temp_df = pd.DataFrame()
            temp_df['d_benefit'] = bootstrapped_scores
            temp_df['lambda'] = lambda_candidate

            bootstrap_df = pd.concat([bootstrap_df, temp_df])


            if stable_score > high_score:

                # first update highscore
                high_score = stable_score
                
                # then update optimal pair
                optimal_lambda = lambda_candidate


    
    df_results = pd.DataFrame()
    df_results['lambda'] = lambda_range
    df_results['score'] = score_list
    df_results['left_out_study'] = left_out_study

    # Create Tune plot
    plt.figure()
    
    # create dotted line plot
    plt.plot(df_results['lambda'], df_results['score'], linestyle='dotted')

    # find maximum y value and its corresponding x value
    max_y = df_results['score'].max()
    max_x = df_results.loc[df_results['score'] == max_y, 'lambda'].iloc[0]

    # add asterisk at maximum value
    plt.plot(max_x, max_y, marker='*', markersize=10, color='red')
    plt.xscale("log")

    plt.ylabel('d-benefit')
    plt.xlabel('\u03BB')
    plt.tight_layout()
    plt.savefig('../overall_pipeline_results/' + save_directory + 'TUNE_plot_' + left_out_study + '.jpg', dpi=300)

    # Create Tune plot inc boxplots
    # plt.figure()

    # sns.boxplot(x="lambda", y="d_benefit",
    #         # hue="smoker", 
    #         palette="Blues",
            
    #         data=bootstrap_df,
    #         width=0.05)
    
    # plt.xticks(rotation=90)

    # plt.tight_layout()
    # plt.savefig('../overall_pipeline_results/' + save_directory + 'TUNE_plot_BOXPLOT_' + left_out_study + '.jpg', dpi=300)




    return optimal_lambda, df_results


def tune_C(X, Y, X_treated, X_untreated, penalty):

    print('triggered tune_C')

    np.logspace(-5, 1, num=50)
    scores = []

    for C_candidate in grid:
        print('candidate: ' + str(C_candidate))

        Y_val_stacked = pd.DataFrame()

        for left_out_study in Y.STUDY.unique():
            
            mask = Y.STUDY != left_out_study
            X_train = X[mask]
            Y_train = Y[mask]

            mask = Y.STUDY == left_out_study
            X_val = X[mask]
            Y_val = Y[mask]
            X_val_treated = X_treated[mask]
            X_val_untreated = X_untreated[mask]


            ite, _ = generic_s_learner(X_train, Y_train, X_val_treated, X_val_untreated, C_candidate, penalty)
            Y_val['ite'] = ite

            Y_val_stacked = pd.concat([Y_val_stacked, Y_val], axis=0)

        
        assert Y.shape[0] == Y_val_stacked.shape[0]

        score = ite_discrimination_score(Y_val_stacked, 'population_benefit')
        scores.append(score)
        print('score:' + str(score))

    max_idx = np.nanargmax(scores)
    tuned_C = grid[max_idx]

    return tuned_C

def generic_s_learner(X_train, Y_train, X_test_treated, X_test_untreated, C, penalty):
    
    from sklearn.linear_model import LogisticRegression
    
    # fit model on (original) train data
    model = LogisticRegression(fit_intercept = True, C=C,
                               penalty=penalty, l1_ratio=0.5,
                               solver='saga').fit(X_train, Y_train.Mort)
    
    # print(model.intercept_)
    # print(model.feature_names_in_)
    # print(model.coef_)
    
    # predict mortality under treatment
    outcome_T_1 = model.predict_proba(X_test_treated)[:,1]
    
    # print('predictions under treatment')
    # print(outcome_T_1)
    
    # predict mortality under placebo
    outcome_T_0 = model.predict_proba(X_test_untreated)[:,1]    
    # print('predictions under no treatment')
    # print(outcome_T_0)
    
    # calculate ITE in test data
    ite = list(outcome_T_0 - outcome_T_1)
    
    return ite, model


def generic_causal_forest(X_train, Y_train, X_test):

    # print('\n ==== START BUILDING CF in PYTHON =======')
    # print('features:')
    # print(X_train.columns)
    max_depth = 3
    # print('max depth: ' + str(max_depth))

    est = CausalForestDML(model_y='auto', model_t='auto', 
                                  discrete_treatment=True, categories=[0,1],
                                  cv=2, mc_iters=None, mc_agg='mean', drate=True,
                                  n_estimators=100, criterion='het', max_depth=max_depth,
                                  min_samples_split=10, min_samples_leaf=5,
                                  min_weight_fraction_leaf=0.0, min_var_fraction_leaf=None,
                                  min_var_leaf_on_val=False, max_features='auto',
                                  min_impurity_decrease=0.0, max_samples=0.45, min_balancedness_tol=0.45,
                                  honest=True, inference=True, fit_intercept=True,
                                  subforest_size=4, n_jobs=- 1, random_state=None, verbose=0)

    est.fit(Y_train.Survival, Y_train.random, X=X_train, W=None)

    ite_left_out = list(est.effect(X_test))
    ite_train = list(est.effect(X_train))

    return ite_left_out, ite_train


# def calculate_auc(Y):
    

def ite_discrimination_score(Y, metric):
    # print('calculate: ', metric)

    if metric == 'etd':
        score = ETD(Y)
        
    elif metric == 'eqd':
        score = EQD(Y)
    
    elif metric == 'population_benefit':
        score = calculate_population_benefit(Y)
        
    elif metric == 'auc_benefit':
        score = calculate_auc_benefit(Y)
    
    elif metric == 'abs_ite_lift':
        score = calculate_abs_ite_lift(Y)

    elif metric == 'auc':
        score = calculate_auc(Y)

    else:
        score, _ = c_for_benefit(None, Y)
        
    return score

def generic_forward_selection(X, Y, X_obs, main_effects, initial_candidates, 
                              penalty, C, metric, B, imp, outer_loop_left_out_study, 
                              save_directory,
                              paired_main_effect_interaction=False,
                              causal_forest=False):
    
    print('forward selection tester Triggered')
    from sklearn.model_selection import train_test_split
    import sys 
    from sklearn.linear_model import LogisticRegression

    # stdoutOrigin=sys.stdout 
    # sys.stdout = open('../forward_selection/INCLUDED_INTERACTIONS_' + metric+ '.txt', 'w')
    
    
    
    study_dict = {0:'Meijvis', 1:'Endeman', 2:'Blum', 3:'Wittermans', 
                  4:'Torres', 5:'Snijders', 6:'Confalonieri', 7: 'Fernández-Serrano'}
    
    studies_inner_loro = Y.STUDY.unique()
    
    
    print('\nusing metric: ', metric)
    print('\nusing '+ penalty +  ' penalization, inverse strength = ', C)
    print('\nInner loro loop studies: ')
    for study in studies_inner_loro:
        print(study_dict[study])
        
    if metric in ['etd', 'eqd', 'abs_ite_lift']:
        tol = 0.2

    elif metric == 'population_benefit':
        tol = 0.1
    else: 
        tol = 10

    # initialize base c-for-benefit
    if metric in ['etd', 'eqd', 'abs_ite_lift', 'auc_benefit', 'population_benefit']:
        highscore = 0
        
    else:
        highscore = 0.5
        
    # initialize included features
    if not paired_main_effect_interaction:
        included_main_effects = main_effects
    else:
        included_main_effects = []    
    
    included_interactions = []

    

    # ==== Loop 1: add one feature per iteration
    for round_number in np.arange(1,len(initial_candidates),1):
        
        print('==== ROUND '+ str(round_number) + ' =====')
        
        boxplot_data = pd.DataFrame()
        
        left_over_candidates = [x for x in initial_candidates if x not in included_interactions]
        print('left over candidates:')
        print(left_over_candidates)
        
        # empty list for mean CV score per candidate 
        overall_score_per_candidate = []
        
        # ======= LOOP 2 ; over candidate features left in this round ===========
        for candidate in left_over_candidates:
            
            print('\nevaluating: ' + candidate)
            
            # collect feature set for this iteration
            if causal_forest:
                features = included_main_effects + [candidate]
            
            elif paired_main_effect_interaction:
                
                main_effects_iteration = included_main_effects + [candidate]
                
                iteractions_iteration = []
                for variable in main_effects_iteration:
                    iteractions_iteration.append(variable+'*random')
                
                features = main_effects_iteration + iteractions_iteration
                    
            else:
                iteractions_iteration = included_interactions + [candidate]
                features = []
                features = features + included_main_effects

                for variable in iteractions_iteration:
                    features.append(variable+'*random')
            

            print('features in model: ', features)
            
            
            Y_test_stacked = pd.DataFrame()
            
            
            
            # ======= LOOP 3 ; inner LOTO to evaluate candidate ===========
            for study in studies_inner_loro:
                
                # print('left-out RCT: ', study_dict[study])
                # split train and test set in LORO fashion
                
                # train data from original df
                mask = Y.STUDY!=study
                X_train = X[mask]
                Y_train = Y[mask]
                
                # test data from modified dfs (untreated and treated)
                mask = Y.STUDY==study
                X_test = X[mask]
                Y_test = Y[mask]
                
                assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
                assert Y_train.shape[0] + Y_test.shape[0] == Y.shape[0]
                
                # NORMALIZE / IMPUTE
                X_train_imp, X_test_imp = normalize_and_impute(X_train, X_test, X_obs, imp)
                
                assert X_train_imp.shape[0] + Y_train.shape[0]
                assert X_test_imp.shape[0] + Y_test.shape[0]
                
                # first reset index
                X_train_imp = X_train_imp.reset_index(drop=True)
                Y_train = Y_train.reset_index(drop=True)
                X_test_imp = X_test_imp.reset_index(drop=True)
                Y_test = Y_test.reset_index(drop=True)
                
                # add treatment variable
                X_train_inc = pd.concat([X_train_imp, Y_train['random']], axis=1)
                X_test_inc = pd.concat([X_test_imp, Y_test['random']], axis=1)
                
                # prepare test dfs for treated and untreated situation
                X_test_treated = X_test_inc.copy()
                X_test_treated['random'] = 1
                
                X_test_untreated = X_test_inc.copy()
                X_test_untreated['random'] = 0
                
                # ADD ALL INTERACTIONS
                for variable in main_effects:
                    col = variable+'*random'
                    # train set
                    X_train_inc[col] = X_train_inc.random * X_train_inc[variable]
                    
                    # test set under treatment
                    X_test_treated[col] = X_test_treated.random * X_test_treated[variable]
                    
                    # test set under no treatment
                    X_test_untreated[col] = X_test_untreated.random * X_test_untreated[variable]
                    
                # SELECT ONLY THE FEATURES INCLUDED IN THIS ITERATION
                X_train_inc = X_train_inc[features]
                
                if causal_forest:
                    X_test_inc = X_test_inc[features]

                X_test_treated = X_test_treated[features]
                X_test_untreated = X_test_untreated[features]
                
                # calculate and add ITEs
                if causal_forest:
                    ite, _ = generic_causal_forest(X_train_inc, Y_train, X_test_inc)
                else:
                    ite = generic_s_learner(X_train_inc, Y_train, X_test_treated, X_test_untreated, C, penalty)
                
                Y_test['ite'] = ite
                
                # add to stacked Y test df
                Y_test_stacked = pd.concat([Y_test_stacked, Y_test], axis=0)
                
            assert Y_test_stacked.shape[0] == Y.shape[0]
        
            print('Point estimate score:')
            score = ite_discrimination_score(Y_test_stacked, metric)
            print(score)

            # make new variable to enable stratified bootstrap samples
            Y_test_stacked['strat'] = Y_test_stacked['Survival'].astype(str) + Y_test_stacked['random'].astype(str)
            
            print('start bootstrapping')
            bootstrapped_scores = []
            
            n = 1
            if B>10:
                n = 10
            if B>100:
                n = 100
            if B>1000:
                n = 1000

            for b in range(B):

                if b%n == 0:
                    print(str(b)+'/'+str(B))

                Y_boot = resample(Y_test_stacked,
                                  stratify=Y_test_stacked.strat
                                  )
                assert Y_test_stacked.shape == Y_boot.shape
                
                bootstrapped_scores.append(ite_discrimination_score(Y_boot, metric))

            stable_score = np.median(bootstrapped_scores)
            print('\nstabalized score: ', stable_score)
            overall_score_per_candidate.append(stable_score)

            # add to boxplot dataframe    
            temp_df = pd.DataFrame()
            temp_df['SCORE'] = bootstrapped_scores
            temp_df['Candidate'] = candidate
            boxplot_data = pd.concat([boxplot_data, temp_df], axis=0)
            
            
        print('===== Looped through all candidates =====')
        
        # define highscore increase
        score_increases = [(x - highscore) for x in overall_score_per_candidate]
        max_idx = np.nanargmax(score_increases)
        highscore_increase = score_increases[max_idx]
        
        plt.figure()
        
        sns.boxplot(x="Candidate",
            y="SCORE", 
            data=boxplot_data, 
             palette="Blues")
        
        if metric == 'etd':
            plt.ylabel("ETD", size=10)
        
        elif metric == 'eqd':
            plt.ylabel("EQD", size=10)

        elif metric == 'population_benefit':
            plt.ylabel("Population benefit (%)", size=10)

        elif metric == 'auc_benefit':
            plt.ylabel("AUC-benefit", size=10)

        elif metric == 'abs_ite_lift':
            plt.ylabel("Risk difference (%)", size=10)

        else:
            plt.ylabel("c-for-benefit", size=10)
        
        plt.xlabel("Interaction term candidates", size=8)
        plt.hlines(y=highscore, xmin=0, xmax=len(left_over_candidates)+0.5, 
                   linestyle='--', linewidth=1, color='r', label='previous highscore')
        plt.hlines(y=0, xmin=0, xmax=len(left_over_candidates)+0.5, linestyle='--', linewidth=1.5, color='k')
        
        if highscore_increase > tol:
           
            plt.plot(max_idx, overall_score_per_candidate[max_idx], marker="*",
                     markersize=10, color='r',
                     label='selected')
        
        plt.xticks(rotation = 90)
        plt.legend(loc='upper left')
        plt.title('outer LORO left out study: ' + outer_loop_left_out_study + ', round: ' + str(round_number), size=10)
        plt.tight_layout()
        print('../overall_pipeline_results/' + save_directory + 'boxplot_'+outer_loop_left_out_study+'_round_'+str(round_number)+'.jpg')
        plt.savefig('../overall_pipeline_results/' + save_directory + 'boxplot_'+outer_loop_left_out_study+'_round_'+str(round_number)+'.jpg', dpi=300)
        
         
        
        
                    
        # check if tolarence is reached
        if highscore_increase > tol:

            # update highscore
            highscore = overall_score_per_candidate[max_idx]
            
            print('updated highscore:')
            print(highscore)


            if metric == 'overall':
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)*100) + '%')

            elif metric == 'auc_benefit':
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)))

            else:
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)) + '%')
            
            print('included feature:')
            print(left_over_candidates[max_idx])
            
            # add interaction with highest increase
            
            if causal_forest:
                included_main_effects.append(left_over_candidates[max_idx])

            elif paired_main_effect_interaction:
                included_main_effects.append(left_over_candidates[max_idx])
                included_interactions.append(left_over_candidates[max_idx])
            else:
                included_interactions.append(left_over_candidates[max_idx])
            
            print('included main effects:')
            print(included_main_effects)

            print('included interactions:')
            print(included_interactions)

        else:
            print('tolerance not reached')
            
            print('FINAL included main effects:')
            print(included_main_effects)

            print('FINAL included interactions:')
            print(included_interactions)

            break
    
    included_interactions_return = []
    for variable in included_interactions:
        included_interactions_return.append(variable+'*random')
    

    print('done')



    return included_main_effects, included_interactions_return, highscore   



def basic_forward_selection(X, Y, X_obs, initial_candidates, 
                              penalty, C, metric, B, imp, outer_loop_left_out_study, 
                              save_directory, missingness_threshold
                              ):
    
    print('forward selection tester Triggered')
    from sklearn.model_selection import train_test_split
    import sys 
    from sklearn.linear_model import LogisticRegression

    # stdoutOrigin=sys.stdout 
    # sys.stdout = open('../forward_selection/INCLUDED_INTERACTIONS_' + metric+ '.txt', 'w')
    
    
    
    study_dict = {0:'Meijvis', 1:'Endeman', 2:'Blum', 3:'Wittermans', 
                  4:'Torres', 5:'Snijders', 6:'Confalonieri', 7: 'Fernández-Serrano'}
    
    studies_inner_loro = Y.STUDY.unique()
    
    
    print('\nusing metric: ', metric)
    print('\nusing '+ penalty +  ' penalization, inverse strength = ', C)
    print('\nInner loro loop studies: ')
    for study in studies_inner_loro:
        print(study_dict[study])
        
    if metric in ['etd', 'eqd', 'abs_ite_lift']:
        tol = 0.1

    elif metric == 'population_benefit':
        tol = 0
    else: 
        tol = 10

    # initialize base c-for-benefit
    if metric in ['etd', 'eqd', 'abs_ite_lift', 'auc_benefit', 'population_benefit']:
        highscore = 0
        
    else:
        highscore = 0.5
        
    # initialize included interactions
    included_interactions = []

    

    # ==== Loop 1: add one feature per iteration
    for round_number in np.arange(1,len(initial_candidates),1):
        
        print('==== ROUND '+ str(round_number) + ' =====')
        
        boxplot_data = pd.DataFrame()
        
        left_over_candidates = [x for x in initial_candidates if x not in included_interactions]
        print('left over candidates:')
        print(left_over_candidates)
        
        # empty list for mean CV score per candidate 
        overall_score_per_candidate = []
        
        # ======= LOOP 2 ; over candidate features left in this round ===========
        for candidate in left_over_candidates:
            
            print('\nevaluating: ' + candidate)

            iteractions_iteration = included_interactions + [candidate]

            Y_test_stacked = pd.DataFrame()
            
            print('start inner LOTO-CV')
            
            # ======= LOOP 3 ; inner LOTO to evaluate candidate ===========
            for study in studies_inner_loro:
                
                print('left out study inner LOTO:')
                print(study)
                
                # train data from original df
                mask = Y.STUDY != study
                X_train = X[mask]
                Y_train = Y[mask]
                
                # test data from modified dfs (untreated and treated)
                mask = Y.STUDY == study
                X_test = X[mask]
                Y_test = Y[mask]
                
                assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
                assert Y_train.shape[0] + Y_test.shape[0] == Y.shape[0]
                
                # stack train set and observational set
                X_combined = pd.concat([X_train, X_obs], axis=0)

                # select main effects in inner loop based on availability
                selected_main_effects = select_main_effects(
                                             X.columns, 
                                             X_combined, 
                                             X_test, 
                                             missingness_threshold,
                                             )

                # select the variables with sufficient data in train and lef-out datasets
                X_train_selection = X_train[selected_main_effects]
                X_test_selection = X_test[selected_main_effects]
                X_obs_selection = X_obs[selected_main_effects]


                # keep only interactions of which main effect is left over
                selected_interactions = [x for x in iteractions_iteration if x in selected_main_effects]

                if candidate not in selected_interactions:
                    print('candidate not included due to missingness, skipped in inner LOTO')

                else:

                    # FIRST IMPUTE
                    # print('\nFIRST IMPUTE')
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
                    # print('\nCREATE TREATED/UNTREATED DATASETS')
                    X_test_treated = X_test_imp.copy()
                    X_test_treated['random'] = 1

                    X_test_untreated = X_test_imp.copy()
                    X_test_untreated['random'] = 0
                    
                    # NORMALIZE
                    # print('\nNORMALIZE')
                    X_train_norm, X_test_treated_norm, X_test_untreated_norm = normalize_only(X_train_imp,
                                                                                            X_test_treated,
                                                                                            X_test_untreated)
                    
                    
                    # ADD INTERACTIONS
                    # print('\nADD INTERACTIONS')
                    for variable in selected_interactions:
                    # for variable in ['crp']:
                        
                        col = variable+'*random'
                        # train set
                        X_train_norm[col] = X_train_norm.random * X_train_norm[variable]
                        
                        # test set under treatment
                        X_test_treated_norm[col] = X_test_treated_norm.random * X_test_treated_norm[variable]
                        
                        # test set under no treatment
                        X_test_untreated_norm[col] = X_test_untreated_norm.random * X_test_untreated_norm[variable]

                    
                    # DROP TREATEMENT VARIABLE
                    X_train_norm = X_train_norm.drop(['random'], axis=1)
                    X_test_treated_norm = X_test_treated_norm.drop(['random'], axis=1)
                    X_test_untreated_norm = X_test_untreated_norm.drop(['random'], axis=1)

                    assert X_train_norm.columns.shape == X_test_treated_norm.columns.shape == X_test_untreated_norm.columns.shape
                    print('features in model: ', X_train_norm.columns)
                    
                    
                    # fit a LASSO/RIDGE model with different penalization strengths
                    model = sm.GLM(Y_train.Mort, X_train_norm, family=sm.families.Binomial())
                    results = model.fit_regularized(alpha=0, L1_wt=1)

                    print(results.params)
                    
                    # calculate and add ITEs for test data
                    y_pred_treated = results.predict(X_test_treated_norm)
                    y_pred_untreated = results.predict(X_test_untreated_norm)

                    ite = list(y_pred_untreated - y_pred_treated)
                    Y_test['ite'] = ite
                    
                    #add to stacked Y dfs
                    Y_test_stacked = pd.concat([Y_test_stacked, Y_test], axis=0)
                    
                
            # assert Y_test_stacked.shape[0] == Y.shape[0]
            print('\nratio harm/benefit:')
            print(Y_test_stacked[Y_test_stacked.ite<=0].shape)
            print(Y_test_stacked[Y_test_stacked.ite>0].shape)


            if Y_test_stacked.shape[0] == 0:
                print('candidate missing in all inner LOTO folds, input NaN')
                stable_score = np.nan
            else:
                print('Point estimate score:')
                score = ite_discrimination_score(Y_test_stacked, metric)
                print(score)

                # make new variable to enable stratified bootstrap samples
                Y_test_stacked['strat'] = Y_test_stacked['Survival'].astype(str) + Y_test_stacked['random'].astype(str)
                
                print('start bootstrapping')
                bootstrapped_scores = []
                
                n = 1
                if B>10:
                    n = 10
                if B>100:
                    n = 100
                if B>1000:
                    n = 1000

                for b in range(B):

                    if b%n == 0:
                        print(str(b)+'/'+str(B))

                    Y_boot = resample(Y_test_stacked,
                                    stratify=Y_test_stacked.strat
                                    )
                    assert Y_test_stacked.shape == Y_boot.shape
                    
                    bootstrapped_scores.append(ite_discrimination_score(Y_boot, metric))


                # add to boxplot dataframe    
                temp_df = pd.DataFrame()
                temp_df['SCORE'] = bootstrapped_scores
                temp_df['Candidate'] = candidate
                boxplot_data = pd.concat([boxplot_data, temp_df], axis=0)

                stable_score = np.median(bootstrapped_scores)


            print('\nstabalized score: ', stable_score)
            overall_score_per_candidate.append(stable_score)

            
            
            
        print('===== Looped through all candidates =====')
        
        # define highscore increase
        score_increases = [(x - highscore) for x in overall_score_per_candidate]
        max_idx = np.nanargmax(score_increases)
        highscore_increase = score_increases[max_idx]
        
        plt.figure()
        
        sns.boxplot(x="Candidate",
            y="SCORE", 
            data=boxplot_data, 
             palette="Blues")
        
        if metric == 'etd':
            plt.ylabel("ETD", size=10)
        
        elif metric == 'eqd':
            plt.ylabel("EQD", size=10)

        elif metric == 'population_benefit':
            plt.ylabel("Population benefit (%)", size=10)

        elif metric == 'auc_benefit':
            plt.ylabel("AUC-benefit", size=10)

        elif metric == 'abs_ite_lift':
            plt.ylabel("Risk difference (%)", size=10)

        else:
            plt.ylabel("c-for-benefit", size=10)
        
        plt.xlabel("Interaction term candidates", size=8)
        plt.hlines(y=highscore, xmin=0, xmax=len(left_over_candidates)+0.5, 
                   linestyle='--', linewidth=1, color='r', label='previous highscore')
        plt.hlines(y=0, xmin=0, xmax=len(left_over_candidates)+0.5, linestyle='--', linewidth=1.5, color='k')
        
        if highscore_increase > tol:
           
            plt.plot(max_idx, overall_score_per_candidate[max_idx], marker="*",
                     markersize=10, color='r',
                     label='selected')
        
        plt.xticks(rotation = 90)
        plt.legend(loc='upper left')
        plt.title('outer LORO left out study: ' + outer_loop_left_out_study + ', round: ' + str(round_number), size=10)
        plt.tight_layout()
        print('../overall_pipeline_results/' + save_directory + 'boxplot_'+outer_loop_left_out_study+'_round_'+str(round_number)+'.jpg')
        plt.savefig('../overall_pipeline_results/' + save_directory + 'boxplot_'+outer_loop_left_out_study+'_round_'+str(round_number)+'.jpg', dpi=300)
        
         
                    
        # check if tolarence is reached
        if highscore_increase > tol:

            # update highscore
            highscore = overall_score_per_candidate[max_idx]
            
            print('updated highscore:')
            print(highscore)


            if metric == 'overall':
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)*100) + '%')

            elif metric == 'auc_benefit':
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)))

            else:
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)) + '%')
            
            print('included feature:')
            print(left_over_candidates[max_idx])
            
            # add interaction with highest increase
            included_interactions.append(left_over_candidates[max_idx])
            
            print('included interactions:')
            print(included_interactions)

        else:
            print('tolerance not reached')
            
            print('FINAL included interactions:')
            print(included_interactions)

            break
    
    print('done')



    return included_interactions 

def free_for_all_forward_selection(X, Y, X_obs, main_effects, candidates, 
                              penalty, C, metric, B, imp, outer_loop_left_out_study, 
                              save_directory,
                              ):
    
    print('free-for-all forward selection')

    from sklearn.model_selection import train_test_split
    import sys 
    from sklearn.linear_model import LogisticRegression

    # stdoutOrigin=sys.stdout 
    # sys.stdout = open('../forward_selection/INCLUDED_INTERACTIONS_' + metric+ '.txt', 'w')
    
    
    study_dict = {0:'Meijvis', 1:'Endeman', 2:'Blum', 3:'Wittermans', 
                  4:'Torres', 5:'Snijders', 6:'Confalonieri', 7: 'Fernández-Serrano'}
    
    studies_inner_loro = Y.STUDY.unique()
    
    
    print('\nusing metric: ', metric)
    print('\nusing '+ penalty +  ' penalization, inverse strength = ', C)
    print('\nInner loro loop studies: ')
    for study in studies_inner_loro:
        print(study_dict[study])
        
    if metric in ['etd', 'eqd', 'abs_ite_lift']:
        tol = 0.2
    else: 
        tol = 10

    # initialize base c-for-benefit
    if metric in ['etd', 'eqd', 'abs_ite_lift', 'auc_benefit']:
        highscore = 0
        
    else:
        highscore = 0.5
        
    
    included_candidates = []

    # ==== Loop 1: add one feature per iteration
    for round_number in np.arange(1,len(candidates),1):
        
        print('==== ROUND '+ str(round_number) + ' =====')
        
        boxplot_data = pd.DataFrame()
        
        left_over_candidates = [x for x in candidates if x not in included_candidates]
        print('left over candidates:')
        print(left_over_candidates)
        
        # empty list for mean CV score per candidate 
        overall_score_per_candidate = []
        
        # ======= LOOP 2 ; over candidate features left in this round ===========
        for candidate in left_over_candidates:
            
            print('\nevaluating: ' + candidate)
            
            # collect feature set for this iteration
            features = included_candidates + [candidate]
            print('features in model: ', features)
            
            
            
            
            # ======= LOOP 3 ; inner LORO to evaluate candidate ===========
            Y_test_stacked = pd.DataFrame()

            for study in studies_inner_loro:
                
                # train data from original df
                mask = Y.STUDY!=study
                X_train = X[mask]
                Y_train = Y[mask]
                
                # test data from modified dfs (untreated and treated)
                mask = Y.STUDY==study
                X_test = X[mask]
                Y_test = Y[mask]
                
                assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
                assert Y_train.shape[0] + Y_test.shape[0] == Y.shape[0]
                
                # NORMALIZE / IMPUTE
                X_train_imp, X_test_imp = normalize_and_impute(X_train, X_test, X_obs, imp)
                
                assert X_train_imp.shape[0] + Y_train.shape[0]
                assert X_test_imp.shape[0] + Y_test.shape[0]
                
                # first reset index
                X_train_imp = X_train_imp.reset_index(drop=True)
                Y_train = Y_train.reset_index(drop=True)
                X_test_imp = X_test_imp.reset_index(drop=True)
                Y_test = Y_test.reset_index(drop=True)
                
                # add treatment variable
                X_train_inc = pd.concat([X_train_imp, Y_train['random']], axis=1)
                X_test_inc = pd.concat([X_test_imp, Y_test['random']], axis=1)
                
                # prepare test dfs for treated and untreated situation
                X_test_treated = X_test_inc.copy()
                X_test_treated['random'] = 1
                
                X_test_untreated = X_test_inc.copy()
                X_test_untreated['random'] = 0
                
                # ADD ALL INTERACTIONS
                for variable in main_effects:
                    col = variable+'*random'
                    # train set
                    X_train_inc[col] = X_train_inc.random * X_train_inc[variable]
                    
                    # test set under treatment
                    X_test_treated[col] = X_test_treated.random * X_test_treated[variable]
                    
                    # test set under no treatment
                    X_test_untreated[col] = X_test_untreated.random * X_test_untreated[variable]
                    
                # SELECT ONLY THE FEATURES INCLUDED IN THIS ITERATION
                X_train_inc = X_train_inc[features]
                X_test_treated = X_test_treated[features]
                X_test_untreated = X_test_untreated[features]
                
                ite = generic_s_learner(X_train_inc, Y_train, X_test_treated, X_test_untreated, C, penalty)
                Y_test['ite'] = ite
                
                # add to stacked Y test df
                Y_test_stacked = pd.concat([Y_test_stacked, Y_test], axis=0)
                
            assert Y_test_stacked.shape[0] == Y.shape[0]
        
            print('Point estimate score:')
            score = ite_discrimination_score(Y_test_stacked, metric)
            print(score)

            # make new variable to enable stratified bootstrap samples
            Y_test_stacked['strat'] = Y_test_stacked['Survival'].astype(str) + Y_test_stacked['random'].astype(str)
            
            if pd.Series(score).isna()[0]:
                print('skip bootstrapping')
                stable_score = np.nan
                bootstrapped_scores = list(np.ones(B)*np.nan)
            else:

                
                
                print('start bootstrapping')
                bootstrapped_scores = []
                
                n = 1
                if B>10:
                    n = 10
                if B>100:
                    n = 100
                if B>1000:
                    n = 1000

                for b in range(B):

                    if b%n == 0:
                        print(str(b)+'/'+str(B))

                    Y_boot = resample(Y_test_stacked,
                                    stratify=Y_test_stacked.strat
                                    )
                    assert Y_test_stacked.shape == Y_boot.shape
                    
                    bootstrapped_scores.append(ite_discrimination_score(Y_boot, metric))

                stable_score = np.median(bootstrapped_scores)
                print('\nstabalized score: ', stable_score)
            
            overall_score_per_candidate.append(stable_score)

            # add to boxplot dataframe    
            temp_df = pd.DataFrame()
            temp_df['SCORE'] = bootstrapped_scores
            temp_df['Candidate'] = candidate
            boxplot_data = pd.concat([boxplot_data, temp_df], axis=0)
            
            
        print('===== Looped through all candidates =====')
        
        # define highscore increase
        score_increases = [(x - highscore) for x in overall_score_per_candidate]
        max_idx = np.nanargmax(score_increases)
        highscore_increase = score_increases[max_idx]
        
        plt.figure()
        
        sns.boxplot(x="Candidate",
            y="SCORE", 
            data=boxplot_data, 
             palette="Blues")
        
        if metric == 'etd':
            plt.ylabel("ETD", size=10)
        
        elif metric == 'eqd':
            plt.ylabel("EQD", size=10)

        elif metric == 'auc_benefit':
            plt.ylabel("AUC-benefit", size=10)

        elif metric == 'abs_ite_lift':
            plt.ylabel("Lift [benefit]", size=10)

        else:
            plt.ylabel("c-for-benefit", size=10)
        
        plt.xlabel("Interaction term candidates", size=8)
        plt.hlines(y=highscore, xmin=0, xmax=len(left_over_candidates)+0.5, 
                   linestyle='--', linewidth=1, color='r', label='previous highscore')
        plt.hlines(y=0, xmin=0, xmax=len(left_over_candidates)+0.5, linestyle='--', linewidth=1.5, color='k')
        
        if highscore_increase > tol:
           
            plt.plot(max_idx, overall_score_per_candidate[max_idx], marker="*",
                     markersize=10, color='r',
                     label='selected')
        
        plt.xticks(rotation = 90)
        plt.legend(loc='upper left')
        plt.title('outer LORO left out study: ' + outer_loop_left_out_study + ', round: ' + str(round_number), size=10)
        plt.tight_layout()
        print('../overall_pipeline_results/' + save_directory + 'boxplot_'+outer_loop_left_out_study+'_round_'+str(round_number)+'.jpg')
        plt.savefig('../overall_pipeline_results/' + save_directory + 'boxplot_'+outer_loop_left_out_study+'_round_'+str(round_number)+'.jpg', dpi=300)
        
                    
        # check if tolarence is reached
        if highscore_increase > tol:

            # update highscore
            highscore = overall_score_per_candidate[max_idx]
            
            print('updated highscore:')
            print(highscore)


            if metric == 'overall':
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)*100) + '%')

            elif metric == 'auc_benefit':
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)))

            else:
                print('Highscore increase > tolerance:\n +'+ str(np.round(highscore_increase,2)) + '%')
            
            print('included feature:')
            print(left_over_candidates[max_idx])
            
            # add interaction with highest increase
            included_candidates.append(left_over_candidates[max_idx])
            
            print('included features:')
            print(included_candidates)

        else:
            print('tolerance not reached')
            
            print('FINAL included features:')
            print(included_candidates)

            break
    
    
    print('done')



    return included_candidates, highscore 


def calculate_auc_benefit(Y, lower=5, upper=95):
    
    from sklearn.metrics import auc
    # loop over percentile boundaries to split group in two
    # dx=5
    # delta_benefits = []
    # threshold_range = np.arange(lower, upper, dx) 

    
    # for threshold in threshold_range:

    #     # seperate harm/neutral group
    #     mask = Y.ite < np.percentile(Y.ite, threshold)
    #     group_1 = Y[mask] 
        
    #     # seperate benefit group
    #     mask = Y.ite >= np.percentile(Y.ite, threshold)
    #     group_2 = Y[mask] 
        
    #     observed_benefit_1 = calc_absolute_benefit(group_1)
    #     observed_benefit_2 = calc_absolute_benefit(group_2)
        
    #     delta_benefits.append(observed_benefit_2 - observed_benefit_1)
    
    
    # auc_benefit = auc(threshold_range, delta_benefits)
    
    x, y = calculate_cumu_benenfit_curve(Y)
    auc_benefit = auc(x, y)
    

    return auc_benefit

def calculate_delta_benefit(Y, lower, upper):

    # loop over percentile boundaries to split group in two

    delta_benefits = []
    threshold_range = np.arange(lower, upper, 0.5) 

    for threshold in threshold_range:

        # print('\nthreshold:')
        # print(threshold)

        # seperate harm/neutral group
        mask = Y.ite < np.percentile(Y.ite, threshold)
        group_1 = Y[mask] 
        
        # print('n samples/events lower ITE group')
        # print(str(group_1.shape[0]) + '/' + str(group_1.Mort.sum()))


        # seperate benefit group
        mask = Y.ite >= np.percentile(Y.ite, threshold)
        group_2 = Y[mask] 
        
        # print('n samples/events lower ITE group')
        # print(str(group_2.shape[0]) + '/' + str(group_2.Mort.sum()))

        observed_benefit_1 = calc_absolute_benefit(group_1)
        observed_benefit_2 = calc_absolute_benefit(group_2)
        
        delta_benefits.append(observed_benefit_2 - observed_benefit_1)
    
    return threshold_range, delta_benefits


def calculate_cumu_benenfit_curve(Y):

    # loop over percentile boundaries to split group in two

    cumu_gains = []
    
    # rank ITEs
    Y = Y.sort_values(by='ite', ascending = False)

    N = Y.shape[0]
    fraction_range = np.arange(0.1, 1.01, 0.01)

    y = []

    for fraction in fraction_range:
        # print(fraction)
        n_chunk = int(N*fraction)
        
        Y_chunk = Y[:n_chunk]
        y.append(calc_absolute_benefit(Y_chunk))
        

    # print(y)
    return fraction_range, y

def calculate_cumu_gains(Y):

    # loop over percentile boundaries to split group in two

    cumu_gains = []
    
    # rank ITEs
    Y = Y.sort_values(by='ite', ascending = False)

    N = Y.shape[0]
    total_responders = Y[(Y.random == 1) & (Y.Survival == 1)].shape[0]
    print('total responders:')
    print(total_responders)
    fraction_range = np.arange(0.01, 1.01, 0.01)

    y = []

    for fraction in fraction_range:
        # print(fraction)
        n_chunk = int(N*fraction)
        
        Y_chunk = Y[:n_chunk]
        # print(Y_chunk.shape)

        n_responders = Y_chunk[(Y_chunk.random == 1) & (Y_chunk.Survival == 1)].shape[0]
        # print('responders:')
        # print(n_responders)

        y.append(n_responders/total_responders)

    print(y)
    return fraction_range, y

def c_for_benefit(X, Y, pairing_method='ITE_based'):
    
    from itertools import combinations
    from scipy.spatial import distance
    from sklearn.impute import KNNImputer
    
    # first balance treatment arms by randomly selecting patients
    # X_balanced, Y_balanced = balance_treatment_arms(X, Y)
    Y_balanced = balance_treatment_arms(Y)
    
    
    # print
    
    # # impute values in X_balanced (not used when pairing based on ITE)
    # print(X_balanced.shape)
    # cols = X_balanced.columns
    # print(cols)
    # imputer = KNNImputer(n_neighbors=5)
    # print(imputer.feature_names_in_)
    # X_balanced = pd.DataFrame(imputer.fit_transform(X_balanced))
    # print(X_balanced.shape)
    # X_balanced.columns = cols
    
    
    # X_balanced = X_balanced.reset_index(drop=True)
    Y_balanced = Y_balanced.reset_index(drop=True)
    
    # split experiment and control group
    # X_E = X_balanced[Y_balanced.random==1].reset_index(drop=True)
    # X_C = X_balanced[Y_balanced.random==0].reset_index(drop=True)
    
    Y_E = Y_balanced[Y_balanced.random==1].reset_index(drop=True)
    Y_C = Y_balanced[Y_balanced.random==0].reset_index(drop=True)
    
    
    # df_pair = pair_instances(X_E, X_C, Y_E, Y_C, pairing_method=pairing_method) #make matched df
    df_pair = pair_instances(Y_E, Y_C, pairing_method=pairing_method) #make matched df
    
    
    idxs = df_pair.index
    cc = list(combinations(idxs,2))

    c = 0
    d=0
    u=0
    for comb in cc:
        if df_pair['true_ite'][comb[0]] == df_pair['true_ite'][comb[1]]: # uninformative
            u += 1
        elif df_pair['true_ite'][comb[0]] < df_pair['true_ite'][comb[1]]: # concordant
            c += 1
        else:
            d += 1
    
    if (c+d) == 0:
        print('WARNING: no informative pairs found')
        c_benefit = 0.5
    else:
        c_benefit = c/(c+d)
    
    return np.round(c_benefit,3), df_pair


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = scipy.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def pair_instances(Y_E, Y_C, pairing_method='ITE_based'):
    
    from itertools import combinations
    from scipy.spatial import distance 
    
    # print('matching patients, matching method:')
    # print(pairing_method)
    
    if pairing_method == 'ITE_based': # rank dataframes based on ite
        
        E_matched = Y_E.sort_values(by='ite').reset_index(drop=True)
        C_matched = Y_C.sort_values(by='ite').reset_index(drop=True)
        
        

    elif pairing_method == 'mahalanobis':
        
        # calculate Maha metric for both treated and untreated patients
        X_total = pd.concat([X_E, X_C], axis=0)
        
        Y_E['mahala'] = mahalanobis(x=X_E, data=X_total, cov=None)
        Y_C['mahala'] = mahalanobis(x=X_C, data=X_total, cov=None)
        
        E_matched = Y_E.sort_values(by='mahala').reset_index(drop=True)
        C_matched = Y_C.sort_values(by='mahala').reset_index(drop=True)
        
    
    # add mean ITE predictions and observed ITE (-1, 0, 1) per pair 
    predicted_benefits = []
    observed_benefits = []
    
    for i in range(E_matched.shape[0]):
        predicted_benefit = np.mean([E_matched['ite'][i], C_matched['ite'][i]])
        predicted_benefits.append(predicted_benefit)
        
        observed_benefit = calc_observed_benefit(E_matched, C_matched, i)
        observed_benefits.append(observed_benefit)
    
    # create dataframe with patient pairs
    df_pair = pd.DataFrame()
    
    df_pair['E_ite'] = E_matched.ite
    df_pair['E_survival'] = E_matched['Survival']
    df_pair['C_ite'] = C_matched.ite
    df_pair['C_survival'] = C_matched['Survival']
    df_pair['predicted_ite'] = predicted_benefits
    df_pair['true_ite'] = observed_benefits
    
    
        
    return df_pair
        
        
     


def balance_treatment_arms(Y):
    
    import random
    
    #split in treatment arms
    # X_E = X[Y.random==1].reset_index(drop=True)
    # X_C = X[Y.random==0].reset_index(drop=True)
    
    Y_E = Y[Y.random==1].reset_index(drop=True)
    Y_C = Y[Y.random==0].reset_index(drop=True)
    
    # print('shape treated:', Y_E.shape)
    # print('shape control:', Y_C.shape)
    
    if Y_E.shape[0] > Y_C.shape[0]:
        
        all_idx = list(Y_E.index)
        sampled_idx = random.sample(all_idx, Y_C.shape[0])
        
        # X_E = X_E.iloc[sampled_idx,:]
        Y_E = Y_E.iloc[sampled_idx,:]
        
        
    elif Y_C.shape[0] > Y_E.shape[0]:
        
        all_idx = list(Y_C.index)
        sampled_idx = random.sample(all_idx, Y_E.shape[0])
        
        # X_C = X_C.iloc[sampled_idx,:]
        Y_C = Y_C.iloc[sampled_idx,:]
    
    else:
        # print('No balancing needed')
        pass
    
    # assert X_C.shape[0] == X_E.shape[0]
    assert Y_C.shape[0] == Y_E.shape[0]
    
    # stack together again, reset indices
    # X_balanced = pd.concat([X_E, X_C], axis=0)
    Y_balanced = pd.concat([Y_E, Y_C], axis=0)
    
    
    
    return Y_balanced

def calc_observed_benefit(E,C,i):
    
    e = E['Survival'][i]
    c = C['Survival'][i]
    
    if c == 1:
        if e == 1:          # both survived (no effect: 0)
            effect = 0
        else:
            effect = -1      # control survived, experiment died (harm: -1)
    
    else:
        if e == 1:
            effect = 1     # control died, experiment survived (benefit: 1)
        else:   
            effect = 0      # both died (no effect: 0)
    
    return effect




# def iceman(effects, p_value):
    
#     # credibility is driven by the items that decrease credibility.
#     probably_decrease = 0
#     definitely_decrease = 0
    
#     # 8 leading questions
    
    
#     # (1) Is the analysis of effect modification based on comparison within rather than between trials?
    
#     # always probably increase, because only IPD
    
#     # (2)  For within-trial comparisons, is the effect modification similar from trial to trial? 
    # if effects
    


#%% Tables for LOS

# outcome = 'LOS'


# C_table = []
# E_table = []
# R_diffs = []
# A_diffs = []


# for i in loop:
#     control = Y_RCT[(Y_RCT.group==i)&(Y_RCT.random==0)][outcome]
#     experiment = Y_RCT[(Y_RCT.group==i)&(Y_RCT.random==1)][outcome]
    
#     C_table.append(str(control.median()) + ' (' + str(np.percentile(control,25)) + '-' + str(np.percentile(control,75)) + ')')
#     E_table.append(str(experiment.median()) + ' (' + str(np.percentile(experiment,25)) + '-' + str(np.percentile(experiment,75)) + ')')
    
#     R_diffs.append((control.median()-experiment.median())/control.median()*100)
#     A_diffs.append((control.median()-experiment.median()))
    
# # bootstrapping procedure
# R_diff_boot = []
# A_diff_boot = []

# for b in range(B):
#     Y_boot = resample(Y_RCT, stratify=Y_RCT.random)
    
#     R_diff_sample = []
#     A_diff_sample = []
    
#     for i in loop:

#         control = Y_boot[(Y_boot.group==i)&(Y_boot.random==0)][outcome]
#         experiment = Y_boot[(Y_boot.group==i)&(Y_boot.random==1)][outcome]
        
#         R_diff_sample.append((control.median()-experiment.median())/control.median()*100)
#         A_diff_sample.append((control.median()-experiment.median()))
            

#     R_diff_boot.append(R_diff_sample)
#     A_diff_boot.append(A_diff_sample)


# R_diff_boot = pd.DataFrame(R_diff_boot)
# A_diff_boot = pd.DataFrame(A_diff_boot)


# R_diff_low = []
# R_diff_high = []
# A_diff_low = []
# A_diff_high = []


# for i in loop:
    
#     R_diff_low.append(np.percentile(R_diff_boot.iloc[:,i],5))
#     R_diff_high.append(np.percentile(R_diff_boot.iloc[:,i],95))
#     A_diff_low.append(np.percentile(A_diff_boot.iloc[:,i],5))
#     A_diff_high.append(np.percentile(A_diff_boot.iloc[:,i],95))


# df['Control'] = C_table
# df['Experiment'] = E_table

# R_diff_table = []
# A_diff_table = []

# for i in loop:
#     R_diff_table.append(str(np.round(R_diffs[i],2)) + ' (' + str(np.round(R_diff_low[i],2)) + ';' + str(np.round(R_diff_high[i],2))+')')
#     A_diff_table.append(str(np.round(A_diffs[i],2)) + ' (' + str(np.round(A_diff_low[i],2)) + ';' + str(np.round(A_diff_high[i],2))+')')
    

# df['Relative reduction (median)'] = R_diff_table
# df['Absolute reduction (median)'] = A_diff_table


# # df.to_excel('HTE_overview_table_LOS_LCA_3_' + outcome + '.xlsx')

# #%% Visualize HTE in plots --> LOS
# colors = ['tab:blue','tab:orange']
# x_shift = [-0.1,0.1]

# families = [sm.families.Binomial(), sm.families.Poisson(), sm.families.Gamma(), sm.families.InverseGaussian()]

# f = families[1]
# formula = outcome+' ~ ' + 'group' +'*random'  
# model = smf.glm(formula = formula, data=Y_RCT, family=f)
# result = model.fit()
# effect = result.params[3]
# p = result.pvalues[3]

# # Boxplots
# import seaborn as sns
# plt.figure()

# df_box = pd.DataFrame()
# for i in loop:
#     control = Y_RCT[(Y_RCT.group==i)&(Y_RCT.random==0)][outcome]
#     df_temp = pd.DataFrame()
#     df_temp['LOS'] = control.values
#     df_temp['Group'] = 'Control' 
#     df_temp[x_label] = x_ticks[i] 
#     df_box = pd.concat([df_box,df_temp],axis=0)
    
#     experiment = Y_RCT[(Y_RCT.group==i)&(Y_RCT.random==1)][outcome]
#     df_temp = pd.DataFrame()
#     df_temp['LOS'] = experiment.values
#     df_temp['Group'] = 'Steroids' 
#     df_temp[x_label] = x_ticks[i] 
#     df_box = pd.concat([df_box,df_temp],axis=0)
    


# sns.boxplot(x=x_label, y='LOS',hue='Group', data=df_box)
# plt.ylim((0,25))
# plt.ylabel('LOS (days)')
# plt.legend(loc='upper center')

# plt.title('HTE on '+outcome+ '\n P='+str(np.round(p,4)))

# # Relative difference in median
# plt.figure()

# for i in loop:
    
#     plt.plot(i+x_shift[0],R_diffs[i],'o',color = colors[0])
#     plt.plot([i+x_shift[0],i+x_shift[0]],[R_diff_low[i],R_diff_high[i]],color='k',Linewidth=0.6)
        
        
# plt.axhline(y=1,Linestyle='--',color='k',Linewidth=0.6)        
# plt.xlabel(x_label)
# plt.ylabel('Relative reduction in median LOS (%)')

# plt.xticks(loop, x_ticks)
# plt.title('HTE on '+outcome)

# #absolute difference in median
# plt.figure()

# for i in loop:
    
#     plt.plot(i+x_shift[0],A_diffs[i],'o',color = colors[0])
#     plt.plot([i+x_shift[0],i+x_shift[0]],[A_diff_low[i],A_diff_high[i]],color='k',Linewidth=0.6)
        

# for i in loop:
#     plt.axhline(y=A_diffs[i],Linestyle='--',color=colors[0],Linewidth=0.4)        
    


# plt.axhline(y=0,Linestyle='--',color='k',Linewidth=0.6)        
# plt.xlabel(x_label)
# plt.ylabel('Absolute reduction in median LOS (days)')

# plt.xticks(loop, x_ticks)
# plt.title('HTE on '+outcome+ '\n P='+str(np.round(p,4)))
