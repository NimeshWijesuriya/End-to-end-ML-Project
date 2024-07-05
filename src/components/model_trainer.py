import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splite training and testing data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],

            )
            models = {
                 "Linear Regression": LinearRegression(),
                 "Lasso": Lasso(),
                 "Ridge": Ridge(),
                 "K-Neighbors Regressor": KNeighborsRegressor(),
                 "Decision Tree": DecisionTreeRegressor(),
                 "Random Forest Regressor": RandomForestRegressor(),
                 "XGBRegressor": XGBRegressor(), 
                 "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {
                "Linear Regression": {},
                "Lasso": {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                    # Add other hyperparameters here if needed
                },
                "Ridge": {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                    # Add other hyperparameters here if needed
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                    # Add other hyperparameters here if needed
                },
                "Decision Tree": {
                    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                    # Add other hyperparameters here if needed
                },
                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                    # Add other hyperparameters here if needed
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                    # Add other hyperparameters here if needed
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                    # Add other hyperparameters here if needed
                }

}
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model= models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square= r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)