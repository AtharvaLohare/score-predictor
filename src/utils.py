import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    """ This will save the serialised object to
        the path specified by argument 1.
    """
    try:
        dir_path = os.path.dirname(file_path)
    
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(models, param_grids, 
                    X_train, X_test, y_train, y_test):
    """
    This is the regression problem statement thus using r2 
    score to choose the suitable model
    
    models : dictionary datatype here : {model_name : modelObj()}.
    """
    try:
        r2_scores_report = {};

        for i in range(len(list(models))):

            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            params = param_grids[model_name]

            gridSCV = GridSearchCV(
                            estimator=model, 
                            param_grid=params,
                            n_jobs=-1,
                            error_score=np.nan
                            )
            
            gridSCV.fit(X_train, y_train)
            model_trained_best_i = gridSCV.best_estimator_

            y_test_pred = model_trained_best_i.predict(X_test)

            model_r2_score = r2_score(y_test, y_test_pred)

            r2_scores_report[model_name] = model_r2_score
            models[model_name] = model_trained_best_i   

        return r2_scores_report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


