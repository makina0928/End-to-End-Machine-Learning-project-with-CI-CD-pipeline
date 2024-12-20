import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
import sys
from src.utils import save_object
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    Defines the path to save the preprocessor object.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    """
    Class to handle the data transformation process, 
    including creating preprocessing pipelines for numerical and categorical features.
    """
    def __init__(self):
        # Initialize configuration for data transformation
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create a preprocessing object for data transformation.

        This function constructs pipelines for numerical and categorical columns.
        Numerical columns: Impute missing values and scale features.
        Categorical columns: Impute missing values, encode categories, and scale features.

        Returns:
            ColumnTransformer: A preprocessing object combining pipelines for different data types.
        
        Raises:
            CustomException: If an error occurs while creating the preprocessing object.
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
                    ("scaler", StandardScaler())  # Standardize the data
                ]
            )

            # Pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle missing values
                    ("one_hot_encoder", OneHotEncoder()),  # Convert categories to one-hot encoding
                    ("scaler", StandardScaler(with_mean=False))  # Scale encoded values
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Perform data transformation on the train and test datasets.

        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the testing dataset.

        Returns:
            tuple: Transformed train and test arrays, and the path to the saved preprocessor object.
        
        Raises:
            CustomException: If an error occurs during the transformation process.
        """
        try:
            # Read train and test data into DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column name and numerical columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target variable for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and testing dataframes."
            )

            # Transform input features using the preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target variable for train and test sets
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")

            # Save the preprocessing object to the specified file path
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed data and the path to the preprocessor object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)
