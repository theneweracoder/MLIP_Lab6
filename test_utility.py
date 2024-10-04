import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    return_tuple = data_split(*feature_target_sample)
    # TODO test if the length of return_tuple is 4
    #raise NotImplemented
    # Check if return_tuple has exactly 4 elements
    assert len(return_tuple) == 4, "Expected 4 elements in the return tuple (X_train, X_test, y_train, y_test)"
    X_train, X_test, y_train, y_test = return_tuple
    # Check if the sizes of X_train, X_test, y_train, and y_test are correct
    total_samples = len(feature_target_sample[0])  # Total number of samples before split
    assert len(X_train) + len(X_test) == total_samples, "The sum of train and test features should equal the total number of samples"
    assert len(y_train) + len(y_test) == total_samples, "The sum of train and test targets should equal the total number of samples"
    # Check if the number of rows in X_train matches the length of y_train
    assert len(X_train) == len(y_train), "X_train and y_train should have the same number of rows"
    # Check if the number of rows in X_test matches the length of y_test
    assert len(X_test) == len(y_test), "X_test and y_test should have the same number of rows"
