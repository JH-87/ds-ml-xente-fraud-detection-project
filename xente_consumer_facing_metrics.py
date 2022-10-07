import numpy as np
def get_money_saved(y_pred, y_true, x_values):
    
    # get boolean mask for the predicted true values
    y_pred_fraud_mask = (y_pred == 1)
    # get boolean mask for the actual true values
    y_true_fraud_mask = (y_true == 1)

    # get boolean mask for true positives
    y_saved_mask = (y_pred_fraud_mask & y_true_fraud_mask)

    # find the value (in money) for each of the true positives
    saved_values = x_values[y_saved_mask]

    # return the sum of the monetary value of the true positives
    return x_values[y_saved_mask].sum()

def get_money_left_on_table(y_pred, y_true, x_values):
    # get boolean mask for the predicted false values
    y_pred_fraud_mask = (y_pred == 0)
    # get boolean mask for the actual true values
    y_true_fraud_mask = (y_true == 1)

    # get boolean mask for true positives
    y_missed_mask = (y_pred_fraud_mask & y_true_fraud_mask)

    # find the value (in money) for each of the false negatives
    missed_values = x_values[y_missed_mask]

    # return the sum of the monetary value of the true positives
    return x_values[y_missed_mask].sum()

def get_annoyance_index(y_pred, y_true, coefficient=1):
    
    # get indices the predicted true values
    y_pred_fraud_mask = (y_pred == 1)
    # get the indices for the actual false values
    y_non_fraud_mask = (y_true == 0)

    # get boolean mask for false positives
    y_fp_mask = (y_pred_fraud_mask & y_non_fraud_mask)

    # count the false positives and multiply by coefficient
    annoyance_index = y_fp_mask.sum() * coefficient

    # return the sum of the value of the true positives
    return annoyance_index