from statsmodels.iolib.smpickle import load_pickle

from pickle import load


load_directory = '../overall_pipeline_results/ALL_models/LASSO/effect_8_AUC_delta_benefit_IQR_Meduri/'


# load the scaler
scaler = load(open(load_directory +'scaler.pkl', 'rb'))
print(scaler.scale_)
print(scaler.mean_)
print(scaler.var_)
print(scaler.feature_names_in_)
print(scaler.n_samples_seen_)





# load the imputer
imputer = load(open(load_directory +'imputer.pkl', 'rb'))

# load the model
model = load_pickle(load_directory +"trained_model.pickle")
coefs = model.params

print('MODEL WEIGHTS:')
print(coefs)
