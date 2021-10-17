"""
This script provides example source code implementation of the TPE optimizer in AutoVI.
Please visit https://optuna.readthedocs.io/en/stable/ for further information on the TPE implementation options.
"""

# Import libraries
import optuna  # open-source hyperparameter optimization package with default TPE optimizer
import pandas as pd
import Models  # custom Models class encoding all 33 index models as defined in Table S2

# Load csv files as pandas dataframes
x = pd.read_csv('x_train.csv')  # hyperspectral reflectance values in csv file
y = pd.read_csv('y_train.csv')  # trait values in csv file
nb_x = x.shape[1]  # total number of bands in x


# Define objective function
def objective(trial):
    # select model from list of 33 candidates - STEP1
    model_index = trial.suggest_int('model_index', 0, 32)
    model = Models(model_index)  # instantiate selected model based on model_index

    # get number of wavebands (nwb) and number of coefficients (ncf) - STEP2
    nwb, ncf = model.params

    # Select optimum wavebands - STEP3
    selected_bands = []  # create empty list to store selected bands
    b1_index = trial.suggest_int('b1_index', 0, nb_x - 1)
    b2_index = trial.suggest_int('b2_index', 0, nb_x - 2)
    selected_bands.append(b1_index, b2_index)
    if nwb >= 3:
        b3_index = trial.suggest_int('b3_index', 0, nb_x - 3)
        selected_bands.append(b3_index)
    if nwb >= 4:
        b4_index = trial.suggest_int('b4_index', 0, nb_x - 4)
        selected_bands.append(b4_index)
    if nwb >= 5:
        b5_index = trial.suggest_int('b5_index', 0, nb_x - 5)
        selected_bands.append(b5_index)
    if nwb >= 6:
        b6_index = trial.suggest_int('b6_index', 0, nb_x - 6)
        selected_bands.append(b6_index)

    # Select optimum coefficient values between 0.0 and 1.0 - STEP3
    cf_val = []  # create empty list to store coefficient values
    if ncf >= 1:
        alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
        cf_val.append(alpha)
    if ncf >= 2:
        beta = trial.suggest_uniform('beta', 0.0, 1.0)
        cf_val.append(beta)
    if ncf >= 3:
        gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
        cf_val.append(gamma)
    if ncf >= 4:
        delta = trial.suggest_uniform('delta', 0.0, 1.0)
        cf_val.append(delta)
    if ncf >= 5:
        epsilon = trial.suggest_uniform('epsilon', 0.0, 1.0)
        cf_val.append(epsilon)

    # Evaluate model - STEP4
    score = model.evaluate(x, y, selected_bands, cf_val)  # calculate R2 based on selected hyperparameters
    return score


# Initiate AutoVI optimization with TPE (default optimizer in optuna)
study = optuna.create_study(direction='maximize')  # increase objective function (R2 score)
study.optimize(objective, n_trials=20,000, n_jobs=-1)  # run optimization for 20,000 trials/iterations with parallel processing (n_jobs=-1)
study.best_params  # get the best model and associated hyperparameters