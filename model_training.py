from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from zenml.steps import step
from typing import Tuple, List
import pandas as pd
import numpy as np
@step
def model_training(merge_df: pd.DataFrame, arrays_question: List[np.ndarray], arrays_answer: List[np.ndarray]) -> Tuple[float, float]:
    X=np.hstack((arrays_question,arrays_answer))
    y=np.array(merge_df['wording']).ravel()
    train_X, val_X, train_y, val_y = train_test_split(X,y,test_size = 0.25, random_state = 14)
    Model_wordings=RandomForestRegressor()
    Model_wordings.fit(train_X,train_y)
    predicted_wordings = Model_wordings.predict(val_X)
    y=np.array(merge_df['content']).ravel()
    train_X, val_X, train_y, val_y = train_test_split(X,y,test_size = 0.25, random_state = 14)
    Model_content=RandomForestRegressor()
    Model_content.fit(train_X,train_y)
    predicted_content = Model_content.predict(val_X)
    mae_wordings = mean_absolute_error(val_y,predicted_wordings)
    mae_content = mean_absolute_error(val_y,predicted_content)
    return mae_wordings, mae_content