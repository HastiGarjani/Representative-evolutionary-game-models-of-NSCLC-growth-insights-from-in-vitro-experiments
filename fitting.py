import numpy as np
from lmfit import minimize, Parameters, report_fit
import pickle
from cancer_models.model_factory import get_model_by_name
from config_models.config import CONFIG_DICT
import pandas as pd
all_models = ['Logistic6']
Environments = ['DMSO', 'CAF', 'Drug', 'CAFandDrug']
t = np.linspace(12, 116, 27)
with open("6train_dataset.pickle", "rb") as f:
    all_data = pickle.load(f)
path = 'config_models'
K_all1 = pd.read_excel(path+"\K.xlsx")
K_all = K_all1.loc[:,~K_all1.columns.str.contains('^Unnamed')]

rho_all1 = pd.read_excel(path+"\/rho.xlsx")
rho_all = rho_all1.loc[:,~rho_all1.columns.str.contains('^Unnamed')]

lambda_all1 = pd.read_excel(path+"\Lambda.xlsx")
lambda_all = lambda_all1.loc[:,~lambda_all1.columns.str.contains('^Unnamed')]
# one K for sensitive
K1_median = np.median(np.concatenate((K_all['$K_1$-DMSO'], K_all['$K_1$-CAF'], K_all['$K_1$-Drug'], K_all['$K_1$-DrugandCAF'])))
# one K for resistant
K2_median = np.median(np.concatenate((K_all['$K_2$-DMSO'], K_all['$K_2$-CAF'], K_all['$K_2$-Drug'], K_all['$K_2$-DrugandCAF'])))
# four \rhos for sensitive
rho1_DMSO_median = np.median(rho_all['$\\rho_1$-DMSO'])
rho1_CAF_median = np.median(rho_all['$\\rho_1$-CAF'])
rho1_Drug_median =  np.median(rho_all['$\\rho_1$-Drug'])
rho1_DrugandCAF_median =  np.median(rho_all['$\\rho_1$-DrugandCAF'])
rho1_median = np.median(np.concatenate((rho_all['$\\rho_1$-DMSO'], rho_all['$\\rho_1$-CAF'], rho_all['$\\rho_1$-Drug'], rho_all['$\\rho_1$-DrugandCAF'])))

# one \rho for resistant
rho2_median = np.median(np.concatenate((rho_all['$\\rho_2$-DMSO'], rho_all['$\\rho_2$-CAF'], rho_all['$\\rho_2$-Drug'], rho_all['$\\rho_2$-DrugandCAF'])))
# two lambda in Drug/DeugandCAF
lambda_Drug = np.median(lambda_all['$\lambda_1$-Drug'])
lambda_DrugandCAF = np.median(lambda_all['$\lambda_1$-DrugandCAF'])
for Environment in Environments:
    for model_name in all_models:
        
        model = get_model_by_name(model_name)

        if Environment == 'DMSO':
            d_parental = all_data["P_train"]
            d_resistant = all_data["R_train"]
            C_all_drug = np.zeros(27)

        if Environment == 'CAF':
            d_parental = all_data["P_train_c"]
            d_resistant = all_data["R_train_c"]
            C_all_drug = np.zeros(27)

        if Environment == 'Drug':
            d_parental = all_data["P_train_d"]
            d_resistant = all_data["R_train_d"]
            C_all_drug = np.ones(27)
            C_all_drug[0] = 0
            C_all_drug[1] = 0

        if Environment == 'CAFandDrug':
            d_parental = all_data["P_train_c_d"]
            d_resistant = all_data["R_train_c_d"]
            C_all_drug = np.ones(27)
            C_all_drug[0] = 0
            C_all_drug[1] = 0

        params = Parameters()

        if model_name in ['Logistic6'] :
            if Environment == 'DMSO':
                 params.add('ro1', value = rho1_median, vary = False)
                 params.add('ro2', value = rho2_median, vary = False)
                 params.add('K1', value = K1_median, vary = False)
                 params.add('K2', value = K2_median, vary = False)
                 params.add('aSR', value = CONFIG_DICT[model_name]['aSR_initial'],\
                            min = CONFIG_DICT[model_name]['aSR_min']\
                                , max = CONFIG_DICT[model_name]['aSR_max'])
                 params.add('aRS', value = CONFIG_DICT[model_name]['aRS_initial'],\
                            min = CONFIG_DICT[model_name]['aRS_min']\
                            , max = CONFIG_DICT[model_name]['aRS_max'])
                 params.add('Lambda_m', value = 0, vary = False)

            if Environment == 'CAF':
                params.add('ro1', value = rho1_median, vary = False)
                params.add('ro2', value = rho2_median, vary = False)
                params.add('K1', value = K1_median, vary = False)
                params.add('K2', value = K2_median, vary = False)
                params.add('aSR', value = CONFIG_DICT[model_name]['aSR_initial'],\
                            min = 0.25\
                                , max = CONFIG_DICT[model_name]['aSR_max'])
                params.add('aRS', value = CONFIG_DICT[model_name]['aRS_initial'],\
                            min = CONFIG_DICT[model_name]['aRS_min']\
                            , max = CONFIG_DICT[model_name]['aRS_max'])
                params.add('Lambda_m', value = 0, vary = False)
            
            

            
            if Environment == 'Drug':
                params.add('ro1', value = rho1_median, vary = False)
                params.add('ro2', value = rho2_median, vary = False)
                params.add('K1', value = K1_median, vary = False)
                params.add('K2', value = K2_median, vary = False)
                params.add('aSR', value = CONFIG_DICT[model_name]['aSR_initial'],\
                            min = 0.2\
                                , max = CONFIG_DICT[model_name]['aSR_max'])
                params.add('aRS', value = 0.2,\
                            min = CONFIG_DICT[model_name]['aRS_min']\
                            , max = CONFIG_DICT[model_name]['aRS_max'])
                params.add('Lambda_m', value = lambda_Drug, vary = False)
                            # CONFIG_DICT[model_name]['lambda_initial'],\
                            # min = CONFIG_DICT[model_name]['lambda_min']\
                            #     , max = CONFIG_DICT[model_name]['lambda_max'])
          
            if Environment == "CAFandDrug":
                params.add('ro1', value = rho1_median, vary = False)
                params.add('ro2', value = rho2_median, vary = False)
                params.add('K1', value = K1_median, vary = False)
                params.add('K2', value = K2_median, vary = False)
                params.add('aSR', value = CONFIG_DICT[model_name]['aSR_initial'],\
                            min = CONFIG_DICT[model_name]['aSR_min']\
                                , max = CONFIG_DICT[model_name]['aSR_max'])
                params.add('aRS', value = CONFIG_DICT[model_name]['aRS_initial'],\
                            min = CONFIG_DICT[model_name]['aRS_min']\
                            , max = CONFIG_DICT[model_name]['aRS_max'])
                params.add('Lambda_m', value = lambda_DrugandCAF, vary = False)
                # \
                #             min = CONFIG_DICT[model_name]['lambda_min']\
                #                 , max = CONFIG_DICT[model_name]['lambda_max'])

        ################## fit each well ########################

        for prop in [3, 4, 5, 6, 7, 8]:
            for w in [0, 1, 2, 3, 4, 5]:
                x_measured = d_parental[3:30,w,prop] 
                y_measured = d_resistant[3:30,w,prop] 
                measured = np.array([x_measured, y_measured]).T
                
                result = minimize(model.residual, params, args=(t, measured, C_all_drug),\
                                    method='differential_evolution', nan_policy='omit')
                
                report_fit(result)
                print(Environment)
                print(model_name)
                dict = {
                    "fit_params" : result.params,
                    "parental" : d_parental[:,w,prop],
                    "resistant" : d_resistant[:,w,prop],
                    "error_chisqr" : result.chisqr,
                    "redchisqr" : result.redchi,
                    "bic" : result.bic,
                    "aic" : result.aic,
                    "residual" : result.residual,

                }
            
                with open(""+"fitting_storage/params/"+Environment+"/"+model_name+"/"+str(prop)+str(w)+".pickle", "wb") as handle:
                    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                