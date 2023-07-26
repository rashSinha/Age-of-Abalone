import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Load the dataset
df = pd.read_csv('../data/abalone.csv')

# Add 'Age' column as the target variable instead of 'Rings'
df['Age'] = df['Rings'] + 1.5

# Load the config files
with open('../config/random_forest.yml', 'r') as file:
    rf_config = yaml.safe_load(file)

with open('../config/svr.yml', 'r') as file:
    svr_config = yaml.safe_load(file)

with open('../config/dl.yml', 'r') as file:
    dl_config = yaml.safe_load(file)

print('Loaded all config files.')

# Define the preprocessing steps for numerical and categorical features
numerical_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
categorical_features = ['Sex']

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Prepare the data
X = df.drop(['Rings', 'Age'], axis=1)
y = df['Age']

print('Preprocessing and data prep done. Splitting the data set into train-test splits.')

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Starting Random Forest Regressor.')

# Define the random forest model pipeline
rf_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Define the hyperparameters to tune
param_grid_rf = {
    'regressor__n_estimators': rf_config['hyperparameters']['RandomForestRegressor']['n_estimators'],
    'regressor__max_depth': rf_config['hyperparameters']['RandomForestRegressor']['max_depth'],
    'regressor__min_samples_split': rf_config['hyperparameters']['RandomForestRegressor']['min_samples_split']
}

# Perform grid search cross validation
rf_model_cv = GridSearchCV(rf_model_pipeline, param_grid_rf, cv=5)
rf_model_cv.fit(X_train, y_train)
print('Random Forest Model Best Parameters:', rf_model_cv.best_params_)

# Use the best model to predict on the test set
rf_model_pred = rf_model_cv.predict(X_test)
rf_model_mse = mean_squared_error(y_test, rf_model_pred)
rf_model_r2 = r2_score(y_test, rf_model_pred)
print('Random Forest Model MSE:', rf_model_mse)
print('Random Forest Model R2 Score:', rf_model_r2)

# Save the random forest model and its predictions
joblib.dump(rf_model_cv.best_estimator_, '../results/rf_model.joblib')
np.savetxt('../results/rf_pred.csv', rf_model_pred)

# Plot the random forest model predictions
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=rf_model_pred)
plt.plot([0, 30], [0, 30], color='red')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title('Random Forest Model Predictions')
plt.savefig('../results/rf_predictions.png')
plt.show(block=False)

print('Starting SVR.')

# Define the SVR model pipeline
svr_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])

# Define the hyperparameters to tune
param_grid_svr = {
    'regressor__C': svr_config['hyperparameters']['SVR']['C'],
    'regressor__kernel': svr_config['hyperparameters']['SVR']['kernel'],
    'regressor__gamma': svr_config['hyperparameters']['SVR']['gamma']
}

# Perform grid search cross validation
svr_model_cv = GridSearchCV(svr_model_pipeline, param_grid_svr, cv=5, n_jobs=-1)
svr_model_cv.fit(X_train, y_train)
print('SVR Model Best Parameters:', svr_model_cv.best_params_)

# Use the best model to predict on the test set
svr_model_pred = svr_model_cv.predict(X_test)
svr_model_mse = mean_squared_error(y_test, svr_model_pred)
svr_model_r2 = r2_score(y_test, svr_model_pred)
print('SVR Model MSE:', svr_model_mse)
print('SVR Model R2 Score:', svr_model_r2)

# Save the SVR model and its predictions
joblib.dump(svr_model_cv.best_estimator_, '../results/svr_model.joblib')
np.savetxt('../results/svr_pred.csv', svr_model_pred)

# Plot the SVR model predictions
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=svr_model_pred)
plt.plot([0, 30], [0, 30], color='red')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title('SVR Model Predictions')
plt.savefig('../results/svr_predictions.png')
plt.show(block=False)

# Remove the 'Sex' column from the training set for DL model and convert the dataframe to a numpy array for ease of use
X_train = preprocessor.fit_transform(X_train, y_train)
X_test = preprocessor.transform(X_test)
X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')
X_train = np.delete(X_train, 3, axis=1) # remove the 'Sex' column
X_test = np.delete(X_test, 3, axis=1) # remove the 'Sex' column

print('Starting Keras Regressor.')

# Define a function to build the Keras regressor model, adding an input_shape that corresponds to the changes in the preprocessed
# data, including the removal of 'Sex' column to ensure the code runs smoothly
def build_regressor_model(optimizer=Adam(), hidden_layers=1, neurons=64, input_shape=None):
    model = Sequential()
    model.add(Dense(units=neurons, input_shape=input_shape, activation='relu'))
    for _ in range(hidden_layers - 1):
        model.add(Dense(units=neurons, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Create the KerasRegressor object
regressor = KerasRegressor(build_fn=build_regressor_model, epochs=100, batch_size=10, input_shape=(X_train.shape[1],))

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', regressor)])

# Define the parameter grid for the grid search
param_grid = {
    'optimizer': dl_config['hyperparameters']['KerasRegressor']['optimizer'],
    'hidden_layers': dl_config['hyperparameters']['KerasRegressor']['hidden_layers'],
    'neurons': dl_config['hyperparameters']['KerasRegressor']['neurons']
}

# Create the GridSearchCV object and perform GridSearch, print the best hyperparameters
dl_model_cv = GridSearchCV(estimator=regressor,
                           param_grid=param_grid,
                           cv=5,
                           verbose=2,
                           n_jobs=-1)
dl_model_cv.fit(X_train, y_train)
print("Best Hyperparameters: ", dl_model_cv.best_params_)

# Use the best model to predict on the test set
dl_model_pred = dl_model_cv.predict(X_test)
dl_model_mse = mean_squared_error(y_test, dl_model_pred)
dl_model_r2 = r2_score(y_test, dl_model_pred)
print('Keras Regressor Model MSE:', dl_model_mse)
print('Keras Regressor Model R2 Score:', dl_model_r2)

# Save the DL model and its predictions
dl_model_cv.best_estimator_.model.save('../results/dl_model.h5')
np.savetxt('../results/dl_pred.csv', dl_model_pred)

# Plot the dl model predictions
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=dl_model_pred)
plt.plot([0, 30], [0, 30], color='red')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title('DL Model Predictions')
plt.savefig('../results/dl_predictions.png')
plt.show(block=False)

# Fit the model and get the history object
history = pipeline['regressor'].fit(X_train, y_train, validation_split=0.2, verbose=0)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Epoch vs Loss')
plt.savefig('../results/dl_val.png')
plt.show(block=False)

# Load the true ages and the predicted ages from each model
true_ages = y_test
rf_pred = np.loadtxt('../results/rf_pred.csv')
svr_pred = np.loadtxt('../results/svr_pred.csv')
dl_pred = np.loadtxt('../results/dl_pred.csv')

# Compute the errors for each model
rf_error = true_ages - rf_pred
svr_error = true_ages - svr_pred
dl_error = true_ages - dl_pred

# Create a boxplot to compare the errors of each model
plt.figure(figsize=(10,6))
sns.boxplot(data=[rf_error, svr_error, dl_error], showmeans=True)
plt.xticks(np.arange(3), ['Random Forest', 'SVR', 'Keras Regressor'])
plt.xlabel('Model')
plt.ylabel('Error')
plt.title('Error Comparison')
plt.savefig('../results/model_comparison.png')
plt.show(block=False)

# Create a dictionary of results
results_dict = {
    'Model': ['Random Forest', 'SVR', 'KerasRegressor'],
    'Best Parameters': [rf_model_cv.best_params_, svr_model_cv.best_params_, dl_model_cv.best_params_],
    'R2 Score': [rf_model_r2, svr_model_r2, dl_model_r2],
    'MSE': [rf_model_mse, svr_model_mse, dl_model_mse]
}

# Convert the dictionary to a Pandas dataframe
results_df = pd.DataFrame.from_dict(results_dict)

# Export the dataframe to a CSV file
results_df.to_csv('../results/model_results.csv', index=False)