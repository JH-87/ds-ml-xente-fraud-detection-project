# We will use this file to define basic functions used in multiple notebooks

def get_data_from_csv(drop=True):
    """df import with some alterations we discovered so far
    Parses dates, drops 'CountryCode' and 'CurrencyCode' columns, sets appropriate dtypes.
    drop (Boolean): Whether or not to drop unnecessary columns (import all if False)
    
    Returns:
        DataFrame: A dataframe with the imported data
    """

    import pandas as pd

    df = pd.read_csv('data/xente/training.csv', parse_dates=['TransactionStartTime'], dtype={'ProductId': 'category','ProductCategory': 'category','ChannelId': 'category','PricingStrategy': 'category'}, index_col='TransactionId')
    if drop: df.drop(['CountryCode', 'CurrencyCode', 'BatchId', 'SubscriptionId', 'AccountId', 'ProductCategory'], axis=1, inplace=True)

    return df