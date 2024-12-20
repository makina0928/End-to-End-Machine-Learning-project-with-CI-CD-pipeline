import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from dataclasses import dataclass
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import sys

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion. 
    Defines the paths for raw, train, and test data files.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """
    Class to handle the data ingestion process, 
    including reading raw data, splitting it into train and test sets,
    and saving the results to specified file paths.
    """
    def __init__(self):
        # Initialize configuration for data ingestion
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads raw data from a source, splits it into train and test datasets,
        and saves these datasets to configured file paths.

        Returns:
            tuple: Paths to the train and test datasets.
        
        Raises:
            CustomException: If an error occurs during ingestion.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset into a pandas DataFrame
            df = pd.read_csv('notebook\\data\\stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create directories for saving the train, test, and raw data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the configured raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            # Split the dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train dataset to the configured train data path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the test dataset to the configured test data path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return the paths of the train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Create an instance of the DataIngestion class
    obj = DataIngestion()

    # Initiate the data ingestion process and get train and test data paths
    train_data, test_data = obj.initiate_data_ingestion()

    # Initiate the data transformation process
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Train and evaluate the model using the processed data
    
    # model_trainer = ModelTrainer()
    # print(model_trainer.initiate_model_trainer(train_arr, test_arr))
