import os
import sys

import pandas as pd
from src.utils import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    """
    Contains methods that will be used for the 
    prediction of the independent feature using suitable model.
    """
    def __init__(self):
        pass

    def predict(self, input_features):
        """
        input_features is a pandas dataframe corresponding to
        the entered details, pd required as column transformer took
        dataframe.
        """

        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            logging.info("Loading the model and the preprocessor.")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            features_scaled = preprocessor.transform(input_features)
            prediction = model.predict(features_scaled)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    """
    Help create the dataframe that shud be passed to the
    preprocessor in the prediction pipeline.
    """
    def __init__(
            self,
            gender: str,
            race_ethnicity: str,
            parental_level_of_education,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int
        ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)