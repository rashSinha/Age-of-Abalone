# Abalone Age Prediction

This project builds three machine learning models using regression to predict the age of abalone from physical measurements.

## Directory Structure

The project is organized as follows:

- `config/`: contains the congifuration parameters for the models used to solve the regression problem of age prediction
- `data/`: contains the abalone dataset in CSV format
- `notebook/`: contains a Jupyter notebook for exploratory data analysis and model building
- `src/`: contains a Python script for data preprocessing, model building, and evaluation
- `results/`: contains output files and figures generated by the models
- `README.md`: this file

## Data

The abalone dataset was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Abalone). The dataset was preprocessed to remove duplicates and missing values (if present), and the 'Age' column was added as the target variable. Feature selection and outlier handling were performed but removed from the pipeline as it interferred with prediction analysis and reduced performance metrics. 

## Models

Three machine learning models were built to predict the age of abalone: Random Forest Regressor, Support Vector Regression, and Regression-based Deep Learning, KerasRegressor. Grid search cross-validation was used to tune the hyperparameters of each model and get the best estimators for each, which were then added to the config files. All the models are contained in a single file.

## Results

The Support Vector Regressor achieved the highest R^2 score of 0.56, followed by Random Forest Regressor with a score of 0.55, and KerasRegressor with a score of 0.54. The SVR model also had the lowest MSE.

## Reproducing the Results

To reproduce the results, you will need to install the following dependencies:

- Python 3.x
- NumPy 1.19.5
- Pandas
- Scikit-learn
- Keras 2.11.0
- Tensorflow 2.5.0
- Matplotlib
- Seaborn

The code can be run in a cloned environment as well. I have used my base conda environment to run the code and download the necessary packages.

You can run the models by executing the Python script in the `src/` directory. The output files and figures will be saved in the `results/` directory.