# Stable, Trending and Volatile categorization of products
import pandas as pd
import numpy as np

# Formulas

def formula_autocorr_lag1(x):
    x = np.array(x, dtype=float)
    mean_x = np.mean(x)
    numerator = np.sum((x[1:] - mean_x)*(x[:-1] - mean_x))
    denominator = np.sum((x - mean_x)**2)
    
    return numerator/denominator if denominator != 0 else np.nan


def calculate_monthly_sales(df):
    return df.groupby(['product_id', 'month', 'year'])['quantity'].sum().reset_index(name='total_sales')


def calculate_number_of_months(df):
    return len(calculate_monthly_sales(df)[['year', 'month']].drop_duplicates())


def calculate_mean(df): 
    return ((df.groupby(['product_id'])['quantity'].sum() / calculate_number_of_months(df)).reset_index(name='mean_sales'))


# Implementations


# Returns Total Mean Sales and Standard Deviation
def calculate_std_dev(df): # Input is -> Processed Sales Data <-
    monthlyTotSales = calculate_monthly_sales(df)
    monthlyTotSales['product_sales_avg'] = (
    monthlyTotSales['product_id'].map(
        calculate_mean(df).set_index('product_id')['mean_sales']
        ))
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


# Input is the -> Processed Sales Data <- and the return of -> calculate_coeff_of_variation() <- function
def calculate_acf1(df, match): 

    buffer_df = calculate_monthly_sales(df)
    buffer_df['product_sales_avg'] = (
        buffer_df['product_id'].map(
            calculate_mean(df).set_index('product_id')['mean_sales']
        ))

    buffer_df = buffer_df.sort_values(by=['product_id', 'month'])
    result = (
        buffer_df.groupby('product_id')['total_sales']
        .apply(formula_autocorr_lag1)
        .reset_index(name='autocorr_lag1')
    )
    result = match.merge(result, on='product_id', how='left')
    return result



def calculate_seasonal_strength(df): # Input is the return of calculate_monthly_sales function
    print('Returns the seasonal strength for each product')
