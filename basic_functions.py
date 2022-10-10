# We will use this file to define basic functions used in multiple notebooks
def get_data_from_csv():
    """df import with some alterations we discovered so far
    Parses dates, drops 'CountryCode' and 'CurrencyCode' columns, sets appropriate dtypes.
    drop (Boolean): Whether or not to drop unnecessary columns (import all if False)
    
    Returns:
        DataFrame: A dataframe with the imported data
    """

    import pandas as pd
    return pd.read_csv(
        'data/xente/training.csv', parse_dates=['TransactionStartTime'],  
        index_col='TransactionId').drop(
            ['BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'CountryCode', 
            'CurrencyCode', 'ProductId'], axis=1
        )


def feature_engineering(dataframe):
    """edits features for model learning

    Args:
        dataframe (pandas_df): data frame obtained from get_data_from_csv()
    Returns:
        DataFrame: A dataframe with the engineered data
    """
    import pandas as pd
    #######################
    # Feature engineering #
    #######################
    # editing columns for smotenc (it apparently does not take strings like "ChannelID_1")
    #dataframe["CustomerId"] = dataframe["CustomerId"].str.replace("CustomerId_", "")
    dataframe["ProviderId"] = dataframe["ProviderId"].str.replace("ProviderId_", "")
    #dataframe["ProductId"] = dataframe["ProductId"].str.replace("ProductId_", "")
    dataframe["ChannelId"] = dataframe["ChannelId"].str.replace("ChannelId_", "")

    # create column with 0 for negative values in "Amount" and 1 for positive values
    dataframe['InOut'] = dataframe['Amount']
    dataframe['InOut'][dataframe['Amount'] < 0 ] = 1
    dataframe['InOut'][dataframe['Amount'] >=0 ] = 0

    # create a column which is 0 if abs("Amount")=="Value" and 1 if not
    dataframe['difference'] = dataframe.eval("abs(Amount) - Value")
    dataframe['difference'][dataframe['difference'] != 0] = 1
    dataframe = dataframe.drop("Amount", axis = 1)

    # create weekday column
    dataframe['weekday'] = dataframe['TransactionStartTime'].dt.dayofweek

    # creating time of day column
    dataframe["time_of_day"] = dataframe["TransactionStartTime"].dt.second + dataframe["TransactionStartTime"].dt.minute * 60 + dataframe["TransactionStartTime"].dt.hour * 3600
    dataframe = dataframe.drop("TransactionStartTime", axis=1)
    dataframe[['PricingStrategy', 'weekday']] = dataframe[['PricingStrategy', 'weekday']].astype('object')
    return dataframe

def tts_custom(df, RSEED):
    """get training and test data from feature-engineered dataframe

    Args:
        df (dataframe): dataframe from feature engineering
    """
    from sklearn.model_selection import train_test_split
    ##############################
    # Get training and test data #
    ##############################
    # Define predictors
    X = df.drop('FraudResult', axis=1)

    # Define target variable
    y = df['FraudResult']

    # Split into train and test set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RSEED, stratify=y, test_size = .25)
    return X_train, X_test, y_train, y_test

def custom_smote(X, y, RSEED):
    """function to implement SMOTENC on the dataframe from tts_custom

    Args:
        X (pd.df): training feature table from tts_custom
        y (pd.df): training target from tts_custom
        RSEED (num): number for random state

    Returns:
        X_sm, y_sm: oversampled datasets
    """
    ##############################
    #     implement smotenc      #
    ##############################
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import SMOTENC

    # I think it's necessary to change to np.array for smote
    X = np.array(
        X[["ProviderId", "ProductCategory", "ChannelId", "PricingStrategy", 
        "weekday", "difference", "InOut", "Value", "time_of_day"]]
        )

    # initiate smotenc
    sm = SMOTENC(categorical_features=[0, 1, 2, 3, 4, 5, 6], random_state = RSEED)
    # run smotenc
    X_sm, y_sm = sm.fit_resample(X, y)

    # transform back into a data frame
    X_sm = pd.DataFrame(X_sm)
    X_sm.columns = ["ProviderId", "ProductCategory", "ChannelId", "PricingStrategy", 
        "weekday", "difference", "InOut", "Value", "time_of_day"]
    X_sm = X_sm.astype({"Value": "int32", "time_of_day": "int32"})
    return X_sm, y_sm


def custom_preprocess(X_tr, X_te, nf):
    """function to preprocess data

    Args:
        X_train (pd.df): training features, run through custom_smote()
        X_test (pd.df): test features, run through custom_smote()
        nf (list of strings): numerical features in X
        cf (list of strings): categorical features in X

    Returns:
        X_tr, X_te: preprocessed datasets
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    # Initiate scaler
    scaler = StandardScaler()
    # run scaler

    X_tr[nf] = scaler.fit_transform(X_tr[nf])
    X_te[nf] = scaler.transform(X_te[nf])

    return X_tr, X_te


def cust_dummies(X, cf):

    # get_dummies
    import pandas as pd
    dummies = pd.get_dummies(X[cf], drop_first=True)
    X = X.drop(cf, axis=1)
    X[dummies.columns] = dummies
    
    return X, dummies.columns

def custom_logreg(X_train, X_test, y_train, y_test):
    """function to run logistic regression

    Args:
        X_train (pd.df): training features, run through custom_smote()
        X_test (pd.df): test features
        y_train (pd.df): training target
        y_test (pd.df): test target

    Returns:
        y_train_pred, y_test_pred: predictions
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    # Initiate model
    logreg = LogisticRegression()
    y_train_pred = logreg.fit(X_train, y_train)
    y_test_pred = logreg.predict(X_test)

    return y_train_pred, y_test_pred

def custom_nb(X_train, X_test, y_train, y_test):
    """function to run logistic regression

    Args:
        X_train (pd.df): training features, run through custom_smote()
        X_test (pd.df): test features
        y_train (pd.df): training target
        y_test (pd.df): test target

    Returns:
        y_train_pred, y_test_pred: predictions
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_predict

    # Initiate model
    nb = GaussianNB()
    y_train_pred = nb.fit(X_train, y_train)
    y_test_pred = nb.predict(X_test)

    return y_train_pred, y_test_pred

def custom_rf(X_train, X_test, y_train, y_test):
    """_summary_

    Args:
        X_train (pd.df): training features, run through custom_smote()
        X_test (pd.df): test features
        y_train (pd.df): training target
        y_test (pd.df): test target

    Returns:
        y_train_pred, y_test_pred: predictions
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    # Initiate model
    rf = RandomForestClassifier()
    y_train_pred = rf.fit(X_train, y_train)
    y_test_pred = rf.predict(X_test)

    return y_train_pred, y_test_pred


def custom_knn(X_train, X_test, y_train, y_test):
    """_summary_

    Args:
        X_train (pd.df): training features, run through custom_smote()
        X_test (pd.df): test features
        y_train (pd.df): training target
        y_test (pd.df): test target

    Returns:
        y_train_pred, y_test_pred: predictions
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict

    # Initiate model
    knn = KNeighborsClassifier()
    y_train_pred = knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)

    return y_train_pred, y_test_pred


def custom_svc(X_train, X_test, y_train, y_test):
    """_summary_

    Args:
        X_train (pd.df): training features, run through custom_smote()
        X_test (pd.df): test features
        y_train (pd.df): training target
        y_test (pd.df): test target

    Returns:
        y_train_pred, y_test_pred: predictions
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_predict

    # Initiate model
    svc = SVC()
    y_train_pred = svc.fit(X_train, y_train)
    y_test_pred = svc.predict(X_test)

    return y_train_pred, y_test_pred


def custom_stack(X_train, y_train, X_test, y_test):
    """_summary_

    Args:
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    ###################
    #    Stacking     #
    ###################
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_predict


    models = [
        ('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('nb', GaussianNB()), 
        ('rf', RandomForestClassifier()), ('svm', SVC())
        ]
    stacking = StackingClassifier(estimators=models)

    # That's from https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
    #scores = cross_val_score(
    #    stacking, X, y, scoring='matthews_corrcoef', cv=5, n_jobs=-1, error_score='raise'
    #    )

    y_train_pred = stacking.fit(
        X_train, y_train
        )
    y_test_pred = stacking.predict(
        X_test
        )
    
    return y_train_pred, y_test_pred



