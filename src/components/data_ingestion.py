import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

# when we are just defining variables we can use @dataclass
@dataclass
class DataIngestionConfig:
    # all outputs are stored in the artifacts folder.
    train_data_path: str = os.path.join("artifacts", "train.csv")  
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # all variables from DataIngestionConfig class will be stored in this variable
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            # Here we read data from any source:
            df = pd.read_csv("E:/Career/Data Science/Machine Learning/Projects/Student_Perfomance_Prediction/data/stud.csv")
            logging.info("Read the dataset as dataframe.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # creating the folder and filepath  
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train_Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Train_Test Completed")
            logging.info("Ingestion of the data completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            ) # returding this file path of datasets for upcoming modules
        except Exception as e:
            raise CustomException(e)
        

