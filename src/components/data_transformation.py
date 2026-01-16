import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # all variables from trans

    def get_data_transformer_object(self):
        '''
            This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )

            logging.info("Numerical Column Scaling completed")
            
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("Encoding", OneHotEncoder())
                ]
            )

            logging.info("Categorical Column encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessing done")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Training and test data")
            logging.info("Importing preprocessing object")

            preprocessor = self.get_data_transformer_object()

            target_column = "math_score"
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education", "lunch", "test_preparation_course"]

            x_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            x_test = test_df.drop(columns=target_column, axis=1)
            y_test = test_df[target_column]

            logging.info("Applying preprocessing object on train and test dataframe")

            preprocessed_x_train = preprocessor.fit_transform(x_train)
            preprocessed_x_test = preprocessor.transform(x_test)

            train_arr = np.c_[preprocessed_x_train, np.array(y_train)]
            test_arr = np.c_[preprocessed_x_test, np.array(y_test)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessor
            )

            logging.info("Saved Preprocessing Object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )

        except Exception as e:
            raise CustomException(e)
            
           