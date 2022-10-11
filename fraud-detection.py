#!/usr/bin/env python
# coding: utf-8

# preprocess functions
from basic_functions import cust_dummies, custom_preprocess
from basic_functions import get_data_from_csv, feature_engineering
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef
import pandas as pd
import pickle
import argparse

# set default command-line arguments
test_data_path = 'data/xente/full_test_export.csv'
model_path = 'pickles/rf_model.p'
csv_output_path = 'output/predictions.csv'

# get command-line arguments (if any)
arg_parser = argparse.ArgumentParser(description='Take fraud test data and generate predictions in a CSV file, and output the F1 and Matthew\'s Correlation Coefficient for the results.')
arg_parser.add_argument('--test-data-path', help=f'The path to the labelled test data. Defaults to {test_data_path}')
arg_parser.add_argument('--model-path', help=f'The path to a pickle file with the pre-trained model. Defaults to {model_path}')
arg_parser.add_argument('--csv-output-path', help=f'The absolute or relative path to output the CSV file with predictions, including the filename. Defaults to {csv_output_path}')
args = arg_parser.parse_args()

test_data_path = args.test_data_path if args.test_data_path else test_data_path
model_path = args.model_path if args.model_path else model_path
csv_output_path = args.csv_output_path if args.csv_output_path else csv_output_path
df = get_data_from_csv(path=test_data_path) # input needs to be reworked, at the moment it takes 'data/xente/training.csv' as input
df = feature_engineering(df)

X = df
y = X.pop('FraudResult')

# --> pass in X as csv 
# --> pass in model as default, via model.sav and its location 
# --> model.sav gives us y_pred = model.predict

#BONUS: if function in case labels/target are present in test.csv --> print predictions and print score else: save predictions

# distinguish between numerical and categorical features
cat_features = [
    "ProviderId", "ProductCategory", "ChannelId", "PricingStrategy", "InOut", "difference", "weekday"
    ]
num_features = ["Value", "time_of_day"]

# import saved scaler so we can pass it into custom_preprocess
our_scaler = pickle.load(open("pickles/scaler.p", "rb"))
# for compatibility, we pass in X twice
_, X_sc = custom_preprocess(X, X, nf=num_features, custom_scaler=our_scaler)

# one hot encoding
X_sc, cat_features = cust_dummies(X_sc, cat_features)

###########################
#  Predict on test data   #
###########################

# import a model from disk, as specified in command-line argument
imported_model = pickle.load(open(model_path, 'rb'))
y_pred = imported_model.predict(X_sc)

# Calculating the accuracy for the classifier
print('Cross validation scores:')
print('-------------------------')
print("F1-score: {:.2f}".format(f1_score(y, y_pred)))
print("MCC: {:.2f}".format(matthews_corrcoef(y, y_pred)))

# Export results as CSV
pd.DataFrame(y_pred).to_csv(csv_output_path)
