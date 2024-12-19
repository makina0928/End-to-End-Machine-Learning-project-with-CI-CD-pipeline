import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self, data_path='notebook/data/stud.csv', output_dir='artifacts'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.train_data_path = os.path.join(output_dir, "train.csv")
        self.test_data_path = os.path.join(output_dir, "test.csv")
        self.raw_data_path = os.path.join(output_dir, "data.csv")

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process.")
        try:
            # Read dataset
            df = pd.read_csv(self.data_path)
            logging.info(f"Dataset loaded successfully from {self.data_path}.")

            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Save raw data
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.raw_data_path}.")

            # Split dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info(f"Train data saved to {self.train_data_path}.")
            logging.info(f"Test data saved to {self.test_data_path}.")

            return self.train_data_path, self.test_data_path
        except Exception as e:
            logging.error("An error occurred during data ingestion.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        ingestion.initiate_data_ingestion()
    except CustomException as e:
        logging.error(e)
