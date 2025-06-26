import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trainer_model_obj_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr, test_arr):
        try:
            logging.info("Splitting the data to get the dependent and independent features.")

            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "GradientBoost Regressor" : GradientBoostingRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoost Regressor" : CatBoostRegressor(verbose=0)
            }

            params={
                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },

                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth': [None, 5, 10, 20] 
                },

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "GradientBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [2, 3, 4, 5],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                },
                
                "XGBRegressor": {
                    'learning_rate': [.01, .05, .1], 
                    'n_estimators': [100, 200]        
                },

                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }

            model_report : dict = evaluate_models(
                                                models=models, param_grids=params,
                                                X_train=X_train,
                                                X_test=X_test,
                                                y_train=y_train,
                                                y_test=y_test
                                            )
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]


            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trainer_model_obj_path,
                obj=best_model
            )
            
            return model_report[best_model_name]


        except Exception as e:
            raise CustomException(e, sys)
