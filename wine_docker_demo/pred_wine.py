import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load

import logging, os

def predict_model():
    # Setting up Logger
    logging.basicConfig(filename = 'train_model.log', encoding = 'UTF-8', 
        filemode='w', level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    model_path = 'model-wine'
    pred_path = 'pred_data.csv'
    #model_path = os.environ["MODEL_NAME"]
    #pred_path = os.environ["PRED_NAME"]

    logger.info(" > Loading prediction data < ")
    pred_data = pd.read_csv(pred_path)
    

    logger.info(" > Loading model to memory < ")
    model = load(model_path)

    logger.info(" > Predicting wine quality < ")
    pred_data['predicted_wine_grade'] = model.predict(pred_data)
    pred_data.to_csv('model_predictions.csv')

if __name__ == "__main__":
    predict_model()