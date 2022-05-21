import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load
import logging, os

def train_model():
    # Setting up Logger
    logging.basicConfig(filename = 'train_model.log', filemode='w', level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Setting file paths for our model
    model_name = 'model-wine'
    #model_name = os.environ["MODEL_NAME"]

    logger.info(" > Importing Data < ")
    dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(
        dataset_url, header='infer', na_values='?', sep=';')
    
    logger.info(" > Splitting in Training & Testing < ")
    X_train, X_test, y_train, y_test = train_test_split(data.drop('quality', axis = 1), data['quality'])

    logger.info(" > Training Random Forest Model < ")
    regressor = RandomForestRegressor(
            max_depth=None, n_estimators=30)
    regressor.fit(X_train, y_train)

    logger.info(" > Saving model to container < ")
    dump(regressor, model_name)

    logger.info("Model saved with name: " + model_name)
    logger.info(" > Evaluating the Model < ")

    y_predicted = regressor.predict(X_test)

    logger.info(" Mean Absolute Error on full test set: " + str(round(metrics.mean_absolute_error(y_test, y_predicted), 3)))

if __name__ == "__main__":
    train_model()