# We will use this file to define basic functions used in multiple notebooks

def get_data_from_csv():
    """df import with some alterations we discovered so far
    Parses dates, drops 'CountryCode' and 'CurrencyCode' columns, sets appropriate dtypes.
    Returns:
        DataFrame: A dataframe with the imported data
    """

    import pandas as pd
    return pd.read_csv('data/xente/training.csv', parse_dates=['TransactionStartTime'], dtype={'ProductId': 'category','ProductCategory': 'category','ChannelId': 'category','PricingStrategy': 'category'}, index_col='TransactionId').drop(['CountryCode', 'CurrencyCode'], axis=1)