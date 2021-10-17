# Import libraries
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Load train and test datasets
# Hyperspectral reflectance values
x_train = pd.read_csv('x_train.csv')
x_test = pd.read_csv('x_test.csv')

# Target - Sugar content
y_train = pd.read_csv('y_train.csv')
y_train = y_train.Target.values.reshape(-1,1)

y_test = pd.read_csv('y_test.csv')
y_test = y_test.Target.values.reshape(-1,1)


# Define optimise PLSR function
def optimise_components(x_train, y_train, n_components):
    """
    Function to optimise number of components for PLSR
    :param x_train: Hyperspectral reflectance values for training, pandas dataframe
    :param y_train: Target values for training, numpy array (n, 1)
    :param n_components: maximum number of components
    :return: list with 5-fold cross validated R2 and corresponding number of components
    """
    results_list = []  # list to store results

    for i in np.arange(1, n_components+1):
        model = PLSRegression(n_components=i)
        y_predict = cross_val_predict(model, x_train, y_train, cv=5)  # 5-fold cross validation
        r2_cv = r2_score(y_train, y_predict)
        results_list.append(r2_cv)
    return results_list


# Optimise number of components for PLSR
optimise_PLSR = optimise_components(x_train, y_train, n_components=20)  # up to max 20 components

# Convert results to dataframe and save as csv
df_components = pd.DataFrame(optimise_PLSR, columns=['R2_score'])
df_components.insert(0, 'n_components', np.arange(1, df_components.size+1))
df_components.to_csv('Optimize_PLSR.csv', index=False)

# Get best n_components
best_n_components = int(df_components[df_components['R2_score']==df_components['R2_score'].max()]['n_components'].values)

# Train PLSR with best n_components
model = PLSRegression(n_components=best_n_components)
model.fit(x_train, y_train)

# Predict target values
y_predict = model.predict(x_test)

# Evaluation metrics
r2 = r2_score(y_test, y_predict)
rmse = mean_squared_error(y_test, y_predict, squared=False)
mae = mean_absolute_error(y_test, y_predict)
mape = mean_absolute_percentage_error(y_test, y_predict) * 100

# Print PLSR results
print(f'Best components: {best_n_components}\n'
      f'R2: {r2}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}')