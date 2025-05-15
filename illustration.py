import numpy as np
from matplotlib import pyplot as plt
import copy as copy
import pickle
import pylab
from cancer_models.model_factory import get_model_by_name
from itertools import product
import pandas as pd
import seaborn as sns

pylab.rcParams['figure.figsize'] = (15.0, 12.0)

all_models = ['Logistic6']

Environments = ['DMSO', 'CAF', 'Drug', 'CAFandDrug']
all_prop = [3, 4, 5, 6, 7, 8]
all_w = [0, 1, 2, 3, 4, 5]
df_empty = pd.DataFrame({'A' : []})

with open("6train_dataset.pickle", "rb") as f:
    all_data = pickle.load(f)
t = np.linspace(12, 116, 27)
time = np.linspace(0, 120, 31)
for Environment, model_name in product(Environments, all_models):
    model = get_model_by_name(model_name)

    if Environment == 'DMSO':
        d_parental = all_data["P_train"]
        d_resistant = all_data["R_train"]
        C_all_drug = np.zeros(27)
        aSR_DMSO = []
        aRS_DMSO = []
        error_DMSO = []
        aic_DMSO = []

    if Environment == 'CAF':
        d_parental = all_data["P_train_c"]
        d_resistant = all_data["R_train_c"]
        C_all_drug = np.zeros(27)
        aSR_CAF = []
        aRS_CAF = []
        error_CAF = []
        aic_CAF = []

    if Environment == 'Drug':
        d_parental = all_data["P_train_d"]
        d_resistant = all_data["R_train_d"]
        C_all_drug = np.ones(27)
        C_all_drug[0] = 0
        C_all_drug[1] = 0
        aSR_Drug = []
        aRS_Drug = []
        L_Drug = []
        error_drug = []
        aic_drug = []

    if Environment == 'CAFandDrug':
        d_parental = all_data["P_train_c_d"]
        d_resistant = all_data["R_train_c_d"]
        C_all_drug = np.ones(27)
        C_all_drug[0] = 0
        C_all_drug[1] = 0
        aSR_CAFandDrug = []
        aRS_CAFandDrug = []
        L_CAFandDrug = []
        error_CAFandDrug = []
        aic_CAFandDrug = []


    Dict = []

    for prop,w in product(all_prop,all_w):
        
        with open(""+"fitting_storage/params/"+Environment+"/"+model_name+"/"+str(prop)+str(w)+".pickle", "rb") as f:
            Dict.append(pickle.load(f))

        x_measured = d_parental[3:30,w,prop] 
        y_measured = d_resistant[3:30,w,prop] 
        measured = np.array([x_measured, y_measured]).T
        params = Dict[-1]["fit_params"]
        name_param = params.keys()
        param_now = []
        if(Environment=='CAF'):
            aSR_CAF.append(round(params['aSR'].value, 5))
            aRS_CAF.append(round(params['aRS'].value, 5))
            error_CAF.append(round(Dict[-1]['error_chisqr'], 5))
            aic_CAF.append(round(Dict[-1]['aic'], 1))
        if(Environment=='DMSO'):
            aSR_DMSO.append(round(params['aSR'].value, 5))
            aRS_DMSO.append(round(params['aRS'].value, 5))
            error_DMSO.append(round(Dict[-1]['error_chisqr'], 5))
            aic_DMSO.append(round(Dict[-1]['aic'], 1))
        if(Environment=='CAFandDrug'):
            aSR_CAFandDrug.append(round(params['aSR'].value, 5))
            aRS_CAFandDrug.append(round(params['aRS'].value, 5))
            L_CAFandDrug.append(round(params['Lambda_m'].value, 5))
            error_CAFandDrug.append(round(Dict[-1]['error_chisqr'], 5))
            aic_CAFandDrug.append(round(Dict[-1]['aic'], 1))

        if(Environment=='Drug'):
            aSR_Drug.append(round(params['aSR'].value, 5))
            aRS_Drug.append(round(params['aRS'].value, 5))
            L_Drug.append(round(params['Lambda_m'].value, 5))
            error_drug.append(round(Dict[-1]['error_chisqr'], 5))
            aic_drug.append(round(Dict[-1]['aic'], 1))

        for key in name_param:
            param_now.append(params[key].value)
        final = np.array(model.eval_fit(t, [x_measured[0],y_measured[0]], C_all_drug, param_now))
    #     #check
    #     res = np.sum(abs((final[:,0]-measured[:,0])/np.max(measured[:,0]))**2 + \
    #    abs((final[:,1]-measured[:,1])/np.max(measured[:,1]))**2)
        fig, ax = plt.subplots(2)
        ax[0].plot(time, d_parental[:,w,prop], 'o')
        ax[0].plot(t, final[:,0], linewidth=2, label="Sensitive", c= 'red')
        ax[1].plot(time, d_resistant[:,w,prop], 'o')
        ax[0].set_ylabel("cell count", size=20)
        ax[1].plot(t, final[:,1], linewidth=2, label="Resistant", c= 'blue')
        fig.legend(prop={'size': 20})
        ax[1].set_xlabel("time", size=20)
        ax[1].set_ylabel("cell count", size=20)
        plt.savefig(""+"fitting_storage/figs/"+Environment+"/"+model_name+"/"+str(prop)+str(w))
        plt.close('all')
        # print(Environment)
        # print(model_name)
        # print(prop)
        # print(w)
    plt.close('all')
    print('NEWRUN')

pylab.rcParams['figure.figsize'] = (6.5, 6.5)

fig, axes = plt.subplots(2,2)
for Environment in Environments:
    if(Environment=='DMSO'):
        df_DMSO_competition_coef = pd.DataFrame({'$\\alpha_{SR}$': np.array(aSR_DMSO),
                    '$\\alpha_{RS}$': np.array(aRS_DMSO),
                    })
        with pd.ExcelWriter("df_DMSO_competition_coef.xlsx") as writer:
            df_DMSO_competition_coef.to_excel(writer)
        sns.boxplot(data = df_DMSO_competition_coef, showfliers=False, ax=axes[0,0])
        # plt.savefig("competition_coef_DMSO.pdf")
        # plt.close()
        print(Environment+ ', ratio: ' + str(np.round(np.array(aSR_DMSO)/np.array(aRS_DMSO), 2)))
        df_all_DMSO = pd.DataFrame({'$a_{SR}$': np.array(aSR_DMSO),
                    '$a_{RS}$': np.array(aRS_DMSO),\
                          'error': np.array(error_DMSO), 'aic': np.array(aic_DMSO)
                    })
        with pd.ExcelWriter("df_all_DMSO.xlsx") as writer:
            df_all_DMSO.to_excel(writer)  
    if(Environment=='Drug'):
        df_Drug_competition_coef = pd.DataFrame({'$\\alpha_{SR}$': np.array(aSR_Drug),
                    '$\\alpha_{RS}$': np.array(aRS_Drug),
                    })
        with pd.ExcelWriter("df_Drug_competition_coef.xlsx") as writer:
            df_Drug_competition_coef.to_excel(writer)
        df_all_Drug = pd.DataFrame({'$a_{SR}$': np.array(aSR_Drug),
                    '$a_{RS}$': np.array(aRS_Drug), 'L':np.array(L_Drug),\
                          'error': np.array(error_drug), 'aic': np.array(aic_drug)
                    })
        with pd.ExcelWriter("df_all_Drug.xlsx") as writer:
            df_all_Drug.to_excel(writer)  
        sns.boxplot(data = df_Drug_competition_coef, showfliers=False, ax=axes[0,1])
        # plt.savefig("competition_coef_Drug.pdf")
        # plt.close()
    if(Environment=='CAF'):
        df_CAF_competition_coef = pd.DataFrame({'$\\alpha_{SR}$': np.array(aSR_CAF),
                    '$\\alpha_{RS}$': np.array(aRS_CAF),
                    })
        with pd.ExcelWriter("df_CAF_competition_coef.xlsx") as writer:
            df_CAF_competition_coef.to_excel(writer)
        sns.boxplot(data = df_CAF_competition_coef, showfliers=False, ax=axes[1,0])
        # plt.savefig("competition_coef_CAF.pdf")
        df_all_CAF = pd.DataFrame({'$a_{SR}$': np.array(aSR_CAF),
                    '$a_{RS}$': np.array(aRS_CAF),\
                          'error': np.array(error_CAF), 'aic': np.array(aic_CAF)
                    })
        with pd.ExcelWriter("df_all_CAF.xlsx") as writer:
            df_all_CAF.to_excel(writer)  
    if(Environment=='CAFandDrug'):
        df_CAFandDrug_competition_coef = pd.DataFrame({'$\\alpha_{SR}$': np.array(aSR_CAFandDrug),
                    '$\\alpha_{RS}$': np.array(aRS_CAFandDrug),
                    })
        with pd.ExcelWriter("df_CAFandDrug_competition_coef.xlsx") as writer:
            df_CAFandDrug_competition_coef.to_excel(writer)
        sns.boxplot(data = df_CAFandDrug_competition_coef, showfliers=False, ax=axes[1,1])
        df_all_CAFandDrug = pd.DataFrame({'$a_{SR}$': np.array(aSR_CAFandDrug),
                    '$a_{RS}$': np.array(aRS_CAFandDrug),\
                          'error': np.array(error_CAFandDrug), 'aic': np.array(aic_CAFandDrug)
                    })
        with pd.ExcelWriter("df_all_CAFandDrug.xlsx") as writer:
            df_all_CAFandDrug.to_excel(writer)  
    axes[0,0].set_ylabel("Competition coefficient values", fontsize=12)
    axes[0,0].title.set_fontsize(13)  
    axes[0,0].title.set_text('Environment = DMSO')
    # axes[0,0].set_ylim([0,7.2])
    axes[0,1].title.set_fontsize(13)  
    axes[0,1].title.set_text('Environment = Drug')
    axes[0,1].set_ylim([0,7.2])
    axes[1,0].set_ylabel("Competition coefficient values", fontsize=12)
    axes[1,0].title.set_fontsize(13)  
    axes[1,0].title.set_text('Environment = CAF')
    # axes[1,0].set_ylim([0,7.2])
    axes[1,1].title.set_fontsize(13)  
    axes[1,1].title.set_text('Environment = CAF and Drug')
    # axes[1,1].set_ylim([0,7.2])

fig.tight_layout(h_pad=2, w_pad=1)

plt.savefig("competitionCoef.eps", format='eps', dpi=300)
        # The residual shape is similar to:
        # x_measured = d_normal_parental[3:30,w,prop] 
        # y_measured = d_normal_resistant[3:30,w,prop] 
        # measured = np.array([x_measured, y_measured]).T
plt.show()
from scipy.stats import f_oneway
f_oneway(df_Drug_competition_coef['$\\alpha_{SR}$'], df_DMSO_competition_coef['$\\alpha_{SR}$'])


f_oneway(df_CAF_competition_coef['$\\alpha_{SR}$'], df_DMSO_competition_coef['$\\alpha_{SR}$'])
f_oneway(df_Drug_competition_coef['$\\alpha_{SR}$'], df_CAFandDrug_competition_coef['$\\alpha_{SR}$'])
f_oneway(df_CAF_competition_coef['$\\alpha_{SR}$'], df_CAFandDrug_competition_coef['$\\alpha_{SR}$'])
f_oneway(df_DMSO_competition_coef['$\\alpha_{SR}$'], df_CAFandDrug_competition_coef['$\\alpha_{SR}$'])
f_oneway(df_CAF_competition_coef['$\\alpha_{SR}$'], df_Drug_competition_coef['$\\alpha_{SR}$'])

f_oneway(df_Drug_competition_coef['$\\alpha_{SR}$'], df_CAFandDrug_competition_coef['$\\alpha_{SR}$'], df_DMSO_competition_coef['$\\alpha_{SR}$'],df_CAF_competition_coef['$\\alpha_{SR}$'])


f_oneway(df_Drug_competition_coef['$\\alpha_{RS}$'], df_DMSO_competition_coef['$\\alpha_{RS}$'])
f_oneway(df_CAF_competition_coef['$\\alpha_{RS}$'], df_DMSO_competition_coef['$\\alpha_{RS}$'])
f_oneway(df_Drug_competition_coef['$\\alpha_{RS}$'], df_CAFandDrug_competition_coef['$\\alpha_{RS}$'])
f_oneway(df_CAF_competition_coef['$\\alpha_{RS}$'], df_CAFandDrug_competition_coef['$\\alpha_{RS}$'])
f_oneway(df_DMSO_competition_coef['$\\alpha_{RS}$'], df_CAFandDrug_competition_coef['$\\alpha_{RS}$'])
f_oneway(df_CAF_competition_coef['$\\alpha_{RS}$'], df_Drug_competition_coef['$\\alpha_{RS}$'])

f_oneway(df_Drug_competition_coef['$\\alpha_{RS}$'], df_CAFandDrug_competition_coef['$\\alpha_{RS}$'], df_DMSO_competition_coef['$\\alpha_{RS}$'],df_CAF_competition_coef['$\\alpha_{RS}$'])

f_oneway(df_DMSO_competition_coef['$\\alpha_{SR}$'], df_DMSO_competition_coef['$\\alpha_{RS}$'])
f_oneway(df_Drug_competition_coef['$\\alpha_{SR}$'], df_Drug_competition_coef['$\\alpha_{RS}$'])
f_oneway(df_CAF_competition_coef['$\\alpha_{SR}$'], df_CAF_competition_coef['$\\alpha_{RS}$'])
f_oneway(df_CAFandDrug_competition_coef['$\\alpha_{SR}$'], df_CAFandDrug_competition_coef['$\\alpha_{RS}$'])
