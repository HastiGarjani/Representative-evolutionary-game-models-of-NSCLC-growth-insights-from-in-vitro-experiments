from cancer_models.Diff_EQs import Logistic6


def get_model_by_name(model_name: str):
    if model_name=='Logistic6':
        return Logistic6()
    else:
        return None