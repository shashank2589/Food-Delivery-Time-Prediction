import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        train_report = {}
        test_report = {}
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = models[model_name]
            
            # Train model
            model.fit(X_train, y_train)

            # Predict Training data
            y_train_pred = model.predict(X_train)
        
            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train data
            train_model_score = r2_score(y_train, y_train_pred) * 100

            # Get R2 scores for test data
            test_model_score = r2_score(y_test, y_test_pred) * 100

            train_report[model_name] = train_model_score
            test_report[model_name] = test_model_score

            print("Train report for", model_name)
            print('--------------------------------------------')
            print("R2 score on train data:", train_model_score)
            print("Test report for", model_name)
            print('--------------------------------------------')
            print("R2 score on test data:", test_model_score)

        return train_report, test_report
    
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)