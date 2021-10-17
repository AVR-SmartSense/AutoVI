# Import libraries
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load train & test datasets
# AutoVI-indices
AutoVI_x_train = pd.read_csv('AutoVI_x_train.csv')
AutoVI_x_test = pd.read_csv('AutoVI_x_test.csv')

# Published VIs
VI_x_train = pd.read_csv('VI_x_train.csv')
VI_x_test = pd.read_csv('VI_x_test.csv')

# Target - Sugar content
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

y_train = y_train.Target.values.reshape(-1,1)
y_test = y_test.Target.values.reshape(-1,1)


# Define SMLR function
def SMLR (x_train, y_train, x_test, y_test, n_features):
    """
    Feature selection with stepwise multiple linear regression (SMLR)
    :param x_train: VI values for training, pandas dataframe
    :param y_train: Target values for testing, numpy array (n, 1)
    :param x_test: VI values for testing, pandas dataframe
    :param y_test: Target values for testing, numpy array (n, 1)
    :param n_features: max number of features to select, int
    :return: list with selected features, R2 score on test dataset
    """
    scores_list = []  # list to store results

    for i in range(n_features):
        model = LinearRegression()  # linear regression model
        feature_names = x_train.columns.values  # VI names

        # Perform stepwise forward selection
        sfs = SequentialFeatureSelector(model,
                                        n_features_to_select=i+1,
                                        cv=5,  # 5-fold cross validation
                                        direction='forward').fit(x_train, y_train)

        # Subset x_train and x_test with selected features
        selected_features = feature_names[sfs.get_support()]
        x_train_selected = x_train[selected_features]
        x_test_selected = x_test[selected_features]

        # Train model using selected features
        model.fit(x_train_selected, y_train)
        y_predict = model.predict(x_test_selected)  # predict target values

        # Evaluation metrics
        r2 = r2_score(y_test, y_predict)
        scores_list.append([selected_features, r2])
    return scores_list


# SMLR for published VIs
print('Performing SMLR on published VIs...')
smlr_VI = SMLR(VI_x_train, y_train, VI_x_test, y_test, n_features=20)  # up to max 20 features

# Convert results to dataframe and save as csv
df_smlr_VI = pd.DataFrame(smlr_VI, columns=['Features',
                                            'R2_score'])
df_smlr_VI.to_csv('SMLR_VI.csv', index=False)
print('Process completed.')

# SMLR for AutoVI-indices
print('Performing SMLR on AutoVI indices...')
smlr_AutoVI = SMLR(AutoVI_x_train, y_train, AutoVI_x_test, y_test, n_features=20)  # up to max 20 features

# Convert results to dataframe and save as csv
df_smlr_AutoVI = pd.DataFrame(smlr_AutoVI, columns=['Features',
                                                    'R2_score'])
df_smlr_AutoVI.to_csv('SMLR_AutoVI.csv', index=False)
print('Process completed.')