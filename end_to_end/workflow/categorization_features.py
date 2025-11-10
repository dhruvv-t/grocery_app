# Functions for categorization of products based on their sales pattern

# Imports
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


# Formulas

def formula_autocorr_lag1(x):
    
    x = np.array(x, dtype=float)
    mean_x = np.mean(x)
    
    numerator = np.sum((x[1:] - mean_x)*(x[:-1] - mean_x))
    denominator = np.sum((x - mean_x)**2)
    
    return numerator/denominator if denominator != 0 else np.nan


def formula_seasonal_strength(ts): # Input is the return of calculate_monthly_sales function
    
    if len(ts) < 12:
        return np.nan
    
    stl = STL(ts, period=12, robust=True)
    res = stl.fit()

    total_var = np.var(ts) 
    # seasonal_var = np.var(res.seasonal)
    resid_var = np.var(res.resid)

    return 1 - (resid_var / total_var)


def formula_normalized_peakiness(ts):
    
    if len(ts) < 6 or ts.sum() == 0:
        return pd.Series({'peakiness_norm': np.nan})
    
    peakiness_norm = np.max(ts) / np.sum(ts)
    
    return pd.Series({'peakiness_norm': peakiness_norm})


def formula_spike_fraction(ts):
    if len(ts) < 6 or ts.sum() == 0:
        return pd.Series({'spike_fraction': np.nan})

    mu, sigma = np.mean(ts), np.std(ts)
    k = 1.5
    spike_fraction = np.sum(ts > mu + k * sigma) / len(ts)

    return pd.Series({'spike_fraction': spike_fraction})


def calculate_monthly_sales(df):
    
    return df.groupby(['product_id', pd.Grouper(key='sales_date', freq='M')])['quantity'].sum().reset_index(name='total_sales')


def calculate_number_of_months(df):
    
    return len(calculate_monthly_sales(df)['sales_date'].unique())


def calculate_mean(df): 
    
    return ((df.groupby(['product_id'])['quantity'].sum() / calculate_number_of_months(df)).reset_index(name='mean_sales'))


# Implementations


# Returns Total Mean Sales and Standard Deviation
def calculate_std_dev(df): # Input is -> Processed Sales Data <-
    
    monthlyTotSales = calculate_monthly_sales(df)
    monthlyTotSales['product_sales_avg'] = (
    monthlyTotSales['product_id'].map(calculate_mean(df).set_index('product_id')['mean_sales']))
    monthlyTotSales['std_dev_buffer_value'] = (monthlyTotSales['total_sales'] - monthlyTotSales['product_sales_avg'])**2
    monthlyTotSales.drop(columns=['total_sales', 'product_sales_avg', 'month'], inplace=True)   
    
    feature_vector = calculate_mean(df)
    feature_vector['std_dev'] = feature_vector['product_id'].map(
        np.sqrt(
            monthlyTotSales.groupby('product_id')['std_dev_buffer_value'].sum() * (1/(calculate_number_of_months(df)-1))
        ).round(0)
        .reset_index(name='std_dev')
        .set_index('product_id')['std_dev']
    )

    return feature_vector


def calculate_coeff_of_variation(df): # Input is the Return of -> calculate_std_dev() <- function 
    
    df['coeff_variation'] = df['std_dev'] / df['mean_sales']
    
    return df


# Input is the -> Processed Sales Data ( df ) <- and the return of -> calculate_coeff_of_variation()  ( match ) <- function
def calculate_acf1(df, match): 

    buffer_df = calculate_monthly_sales(df)
    buffer_df['product_sales_avg'] = (buffer_df['product_id'].map(calculate_mean(df).set_index('product_id')['mean_sales']))
    buffer_df = buffer_df.sort_values(by=['product_id', 'month'])
    
    result = (
        buffer_df.groupby('product_id')['total_sales']
        .apply(formula_autocorr_lag1)
        .reset_index(name='autocorr_lag1')
    )
    result = match.merge(result, on='product_id', how='left')
    
    return result


def calculate_seasonal_strength(df): # Input is the return of -> calculate_monthly_sales() <- function
    
    return df.groupby('product_id').apply(lambda x: formula_seasonal_strength(x.set_index('sales_date')['total_sales'])).reset_index(name='seasonal_strength')


def calculate_peakiness_and_spike_index(df): # Input is return of -> calculate_monthly_sales() <- function
    
    peakiness = df.groupby('product_id').apply(lambda x: formula_normalized_peakiness(x.set_index('sales_date')['total_sales'])).reset_index()
    spike_fraction = df.groupby('product_id').apply(lambda x: formula_spike_fraction(x.set_index('sales_date')['total_sales'])).reset_index()

    spike_index = peakiness.merge(spike_fraction, on='product_id', how='left')

    return spike_index