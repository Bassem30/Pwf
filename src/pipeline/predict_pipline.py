import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path="artifacts\model.pkl"
            preprocessor_path="artifacts\proprocessor.pkl"
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__( self, 
        ct_depth: float,
        ct_pressure: float,
        n2_rate: float):

        self.ct_depth = ct_depth

        self.ct_pressure = ct_pressure

        self.n2_rate = n2_rate

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "ct_depth": [self.ct_depth],
                "ct_pressure": [self.ct_pressure],
                "n2_rate": [self.n2_rate],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)