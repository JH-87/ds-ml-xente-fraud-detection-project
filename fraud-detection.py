#!/usr/bin/env python
# coding: utf-8

# preprocess functions
from basic_functions import cust_dummies, custom_preprocess
from basic_functions import get_data_from_csv, feature_engineering
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef


# #### load the model from disk
# 
# `loaded_model = joblib.load(filename)  
# 
# result = loaded_model.score(X_test, Y_test)  

df = get_data_from_csv() # input needs to be reworked, at the moment it takes 'data/xente/training.csv' as input
df = feature_engineering(df)

# --> pass in X_test as csv 
# --> pass in model as default, via model.sav and its location 
# --> model.sav gives us y_pred = model.predict


#BONUS: if function in case labels/target are present in test.csv --> print predictions and print score else: save predictions


# sort test to match train
X_test = X_test[["ProviderId", "ProductCategory", "ChannelId", "PricingStrategy", 
        "weekday", "difference", "InOut", "Value", "time_of_day"]]

cat_features = [
    "ProviderId", "ProductCategory", "ChannelId", "PricingStrategy", "InOut", "difference", "weekday"
    ]
num_features = ["Value", "time_of_day"]

_, X_test_sc = custom_preprocess(X_train_sm, X_test, nf=num_features)

X_test_sc = X_test_sc.astype({"difference": "object", "InOut": "object"})


X_test_sc, cat_features = cust_dummies(X_test_sc, cat_features)

###########################
#  Predict on test data   #
###########################

# Calculating the accuracy for the RandomForest Classifier 
print('Cross validation scores Random Forest:')
print('-------------------------')
print("F1-score: {:.2f}".format(f1_score(y_test, y_test_rf)))
print("MCC: {:.2f}".format(matthews_corrcoef(y_test, y_test_rf)))

# Export results as CSV


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