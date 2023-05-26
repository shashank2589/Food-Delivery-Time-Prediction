import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import xgboost as xgb
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'Decision Tree':DecisionTreeRegressor(min_samples_leaf=6, max_depth=7, random_state=42),
            'Random Forest':RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_depth=10, random_state=42),
            'Bagging':BaggingRegressor(base_estimator=DecisionTreeRegressor(min_samples_leaf=6, max_depth=7, random_state=42), n_estimators=10, random_state=42),
            'XGBoost':xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1 ,objective='reg:squarederror', random_state=42)
        }
            
            train_model_report, test_model_report=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(train_model_report)
            print(test_model_report)
            print('\n', '='*50, '\n')
            logging.info(f'Train Model Report : {train_model_report}')
            logging.info(f'Test Model Report : {test_model_report}')

            # To get best model score from dictionary 
            best_model_score = max(test_model_report.values())

            best_model_name = list(test_model_report.keys())[
                list(test_model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score.round(2)} percent')
            print('\n', '='*50, '\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score.round(2)} percent')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
        return best_model_name, best_model_score, train_model_report, test_model_report