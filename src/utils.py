import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object will be saved.
        obj (any): The Python object to save.

    Raises:
        CustomException: If any error occurs during the process.
    """
    try:
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if any error occurs
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple machine learning models using grid search for hyperparameter tuning.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target variable.
        X_test (array-like): Testing features.
        y_test (array-like): Testing target variable.
        models (dict): Dictionary of model names and their corresponding instances.
        param (dict): Dictionary of hyperparameters for each model.

    Returns:
        dict: A report containing model names and their R-squared scores on the test set.

    Raises:
        CustomException: If an error occurs during model evaluation.
    """
    try:
        report = {}  # Dictionary to store the evaluation results

        # Iterate over each model and its corresponding hyperparameters
        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model instance
            para = param[list(models.keys())[i]]  # Get hyperparameters for the model

            # Perform grid search for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Update the model with the best parameters found by grid search
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train the model with the best parameters

            # Predict on the training set
            y_train_pred = model.predict(X_train)

            # Predict on the testing set
            y_test_pred = model.predict(X_test)

            # Calculate R-squared scores for training and testing data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a serialized object from a file.

    Args:
        file_path (str): Path to the file containing the serialized object.

    Returns:
        object: The deserialized object.

    Raises:
        CustomException: If an error occurs while loading the object.
    """
    try:
        # Open the file in binary read mode and load the object
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)