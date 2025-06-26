
# imports to handle system configuration.
import sys
import os
from dataclasses import dataclass

# imports to handle the data transforamtion.
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# imports to handle the errors and help debugging.

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# setup the paths and the initial configurations
# to save the transformer in serialised manner.

@dataclass
class DataTransformationConfig:
    preprocessing_obj_filepath = os.path.join('artifacts', 'preprocessor.pkl')

# Implementation of pipelines and the transformers.

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_datatransformer_obj(self):
        """
        This method will build the data tranformer from scratch,
        using separate pipelines for numerical and categorical features
        and then combining them using column transformer.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            pipeline_num_features = Pipeline(
                [
                    ("Handle the missing datapoints.", SimpleImputer(strategy="median")),
                    ("Standardize the features.", StandardScaler())
                ]
            )
            logging.info("Created pipeline for numerical features.")

            pipeline_cat_features = Pipeline(
                [
                    ("Handle the missing datapoints.", SimpleImputer(strategy="most_frequent")),
                    ("Encoding the values.", OneHotEncoder())
                ]
            )
            logging.info("Created pipeline for categorical features.")

            preprocessor_obj = ColumnTransformer(
                [
                    ("Numerical features.", pipeline_num_features, numerical_columns),
                    ("Categorical features", pipeline_cat_features, categorical_columns)
                ]
            )
            logging.info("Created the preprocessor object using Column Transformer.")
            return preprocessor_obj


        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Does two things mainly;
        1) Takes the raw, train and test data, does data prepreprocessing
           on the them and returns the arrays with processed independent 
           features combined with the dependent features.
        2) Saves the preprocessor object into the artefact as pkl and returns .
        """
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Reading the train and test data complete.")

            logging.info("Getting the data preprocessing object.")

            preprecessor = self.get_datatransformer_obj()

            target_column = "math_score"

            input_features_train_df = train_data.drop(columns=target_column)
            target_feature_train_df = train_data[target_column]

            input_features_test_df = test_data.drop(columns=target_column)
            target_feature_test_df = test_data[target_column]

            logging.info(
                "Applying the preprocessor to the independent feature in the" \
                "train dataset and the test data set."
            )

            input_features_train_df = preprecessor.fit_transform(input_features_train_df)
            input_features_test_df = preprecessor.transform(input_features_test_df)

            # recombining the processed inoput and the target features.

            train_arr = np.c_[
                input_features_train_df, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_df, np.array(target_feature_test_df)
            ]

            logging.info("Saving the preprocessed data.")

            save_object(
                file_path= self.transformation_config.preprocessing_obj_filepath,
                obj=preprecessor
            )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessing_obj_filepath
            )


        except Exception as e:
            raise CustomException(e, sys)
