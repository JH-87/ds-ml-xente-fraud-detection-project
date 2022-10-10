#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# preprocess functions
from basic_functions import cust_dummies, custom_preprocess
from basic_functions import get_data_from_csv, feature_engineering, tts_custom, custom_smote
from basic_functions import custom_logreg, custom_knn, custom_nb, custom_svc, custom_rf, custom_stack
# for some reason the other functions won't load
from sklearn.metrics import confusion_matrix


# In[ ]:


df = get_data_from_csv()
df.head()


# In[ ]:


df = feature_engineering(df)


# In[ ]:


df.head()


# In[ ]:


X_train, X_test, y_train, y_test = tts_custom(df, RSEED = 42)


# In[ ]:


X_test.head()


# In[ ]:


X_train_sm, y_train_sm = custom_smote(X_train, y_train, 42)

# sort test to match train
X_test = X_test[["ProviderId", "ProductCategory", "ChannelId", "PricingStrategy", 
        "weekday", "difference", "InOut", "Value", "time_of_day"]]


# In[ ]:


# andrey and john: how are transactions "developing" within an Id
# number of customer ids associated with an account
# starting a grid search
# making sure functions are documented and everything is pushed to git hub
# Jannik: check the pattern?
# feature importance


# In[ ]:


print(X_train.shape)
X_train_sm.shape


# In[ ]:


cat_features = [
    "ProviderId", "ProductCategory", "ChannelId", "PricingStrategy", "InOut", "difference", "weekday"
    ]
num_features = ["Value", "time_of_day"]

X_train_sm_sc, X_test_sc = custom_preprocess(X_train_sm, X_test, nf=num_features)


# In[ ]:


# Make sure types are the same for train and test (can certainly be done more elegantly from the start)
X_test_sc = X_test_sc.astype({"difference": "object", "InOut": "object"})


# In[ ]:


X_train_sm_sc.dtypes


# In[ ]:


X_test_sc.dtypes


# In[ ]:


cat_features


# In[ ]:


X_train_sm_sc, cat_features_dummies = cust_dummies(X_train_sm_sc, cat_features)
X_test_sc, cat_features = cust_dummies(X_test_sc, cat_features)


# In[ ]:


X_test_sc.columns


# In[ ]:


X_train_sm_sc.head()


# In[ ]:


y_train_sm_lr, y_test_lr = custom_logreg(X_train_sm_sc, X_test_sc, y_train_sm, y_test)
confusion_matrix(y_test, y_test_lr)


# In[ ]:


y_train_sm_nb, y_test_nb = custom_nb(X_train_sm_sc, X_test_sc, y_train_sm, y_test)


# In[ ]:


confusion_matrix(y_test, y_test_nb)


# In[ ]:


y_train_sm_rf, y_test_rf = custom_rf(X_train_sm_sc, X_test_sc, y_train_sm, y_test)


# In[ ]:


confusion_matrix(y_test, y_test_rf)


# In[ ]:


y_train_sm_knn, y_test_knn = custom_knn(X_train_sm_sc, X_test_sc, y_train_sm, y_test)
confusion_matrix(y_test, y_test_knn)


# In[ ]:


y_train_sm_svc, y_test_svc = custom_svc(X_train_sm_sc, X_test_sc, y_train_sm, y_test)
confusion_matrix(y_test, y_test_svc)


# In[ ]:


y_train_sm_stack, y_test_stack = custom_knn(X_train_sm_sc, X_test_sc, y_train_sm, y_test)
confusion_matrix(y_test, y_test_stack)


# In[ ]:


from sklearn.metrics import f1_score, matthews_corrcoef
###########################
#  Predict on test data   #
###########################

# Calculating the accuracy for the LogisticRegression Classifier 
print('Cross validation scores Logistic Regression:')
print('-------------------------')
print("F1-score: {:.2f}".format(f1_score(y_test, y_test_lr)))
print("MCC: {:.2f}".format(matthews_corrcoef(y_test, y_test_lr)))

# Calculating the accuracy for the RandomForest Classifier 
print('Cross validation scores Random Forest:')
print('-------------------------')
print("F1-score: {:.2f}".format(f1_score(y_test, y_test_rf)))
print("MCC: {:.2f}".format(matthews_corrcoef(y_test, y_test_rf)))

# Calculating the accuracy for the KNN Classifier 
print('Cross validation scores KNN:')
print('-------------------------')
print("F1-score: {:.2f}".format(f1_score(y_test, y_test_knn)))
print("MCC: {:.2f}".format(matthews_corrcoef(y_test, y_test_knn)))

# Calculating the accuracy for the SVM Classifier 
print('Cross validation scores SVM:')
print('-------------------------')
print("F1-score: {:.2f}".format(f1_score(y_test, y_test_svc)))
print("MCC: {:.2f}".format(matthews_corrcoef(y_test, y_test_svc)))

# Calculating the accuracy for the Naive Bayes Classifier 
print('Cross validation scores Naive Bayes:')
print('-------------------------')
print("F1-score: {:.2f}".format(f1_score(y_test, y_test_nb)))
print("MCC: {:.2f}".format(matthews_corrcoef(y_test, y_test_nb)))

# Calculating the accuracy for the stacking Classifier 
print('Cross validation scores Stack:')
print('-------------------------')
print("F1-score: {:.2f}".format(f1_score(y_test, y_test_stack)))
print("MCC: {:.2f}".format(matthews_corrcoef(y_test, y_test_stack)))


# some old code which I may want to reactivate again
# 
# `# Initiate OneHotEncoder()  
# ohe = OneHotEncoder(handle_unknown='ignore')  
# # run ohe  
# X_train = ohe.fit_transform(X_train[cf])
# X_test[cf] = ohe.transform(X_test[cf])
#     
# # transfrom sparse to dense
# X_train = X_train.todense() 
# X_test = X_test.todense()`
# 

# ### As learning a model takes a lot of time, I think it would be useful to save them to file so that they can be loaded later on for prediction
# #### examplecode, save the model to disk
# from [here](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)  
# 
# `filename = 'finalized_model.sav'  
# 
# joblib.dump(model, filename)`  
#  
# #### some time later...
#  
# #### load the model from disk
# 
# `loaded_model = joblib.load(filename)  
# 
# result = loaded_model.score(X_test, Y_test)  
# 
# print(result)`  

# In[ ]:


def get_all_data_from_csv():
    """df import with some alterations we discovered so far
    Parses dates, drops 'CountryCode' and 'CurrencyCode' columns, sets appropriate dtypes.
    Returns:
        DataFrame: A dataframe with the imported data
    """

    import pandas as pd
    return pd.read_csv(
        'data/xente/training.csv', parse_dates=['TransactionStartTime'],  
        index_col='TransactionId')
import pandas as pd
tmpdf = get_all_data_from_csv()


# In[ ]:


tmpdf["mvg_avg_fr"] = df.FraudResult.rolling(14).mean()
tmpdf["mvg_avg_v"] = df.Value.rolling(14).mean()
import seaborn as sns

#sns.barplot(data=tmpdf, x="Start")
#tmpdf["day"]


# In[ ]:


import matplotlib.pyplot as plt
sns.scatterplot(data=tmpdf, x="TransactionStartTime", y="mvg_avg_v", hue = "FraudResult")
plt.xticks(rotation = 45);


# In[ ]:


sns.scatterplot(data=tmpdf, x="TransactionStartTime", y="mvg_avg_fr", hue = "FraudResult")
plt.xticks(rotation = 45);


# In[ ]:


sns.histplot(data=tmpdf, x="TransactionStartTime", bins=13)
plt.xticks(rotation = 45);


# In[ ]:


tmpdf.TransactionStartTime.dt.date.nunique()


# In[ ]:





# In[ ]:


tmpdf = get_all_data_from_csv()

import numpy as np
tmp = tmpdf[tmpdf['AccountId'].isin(list(tmpdf[tmpdf.FraudResult == 1].AccountId))].reset_index()

sel = np.array(tmp[tmp["FraudResult"] == 1].index)
sel1 = sel + 1
sel2 = sel + 2
sel3= sel + 3
sel4 = sel + 4
sel01 = sel -1
sel02 = sel -2
sel03 = sel -3
sel04 = sel -4
selfinal = np.array([sel04, sel03, sel02, sel01, sel, sel1, sel2, sel3, sel4]).reshape(-1).tolist()
tmp = tmp.iloc[sorted(selfinal), ]
# TODO: group number of CustomerIds for each account and assign new column
tmp.to_csv("data/xente/frauds.csv")


# In[ ]:


tmpdf.groupby("AccountId")["CustomerId"].count().sort_values(ascending=False).head(20)


# In[ ]:


tmp.groupby("AccountId")["CustomerId"].count().sort_values(ascending=False)

