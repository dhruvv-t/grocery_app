# Functions for categorization of products based on their sales pattern

# Imports
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.stats import linregress


# Formulas

def formula_autocorr_lag1(x):
    
    x = np.array(x, dtype=float)
    mean_x = np.mean(x)
    
    numerator = np.sum((x[1:] - mean_x)*(x[:-1] - mean_x))
    denominator = np.sum((x - mean_x)**2)
    
    return numerator/denominator if denominator != 0 else np.nan


def formula_seasonal_strength(ts):
    
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


def formula_trend_slope(df):
    if len(df) < 3:
        return np.nan  # not enough data points
    slope, _, r_value, _, _ = linregress(df['month_number'], df['total_sales'])
    return pd.Series({'trend_slope': slope, 'r_square': r_value**2})


def formula_entropy(df):
    
    p = df['p_t'].dropna()
    p = p[p > 0]
    
    return -np.sum(p * np.log(p))


def calculate_monthly_sales(df):
    
    return df.groupby(['product_id', pd.Grouper(key='sales_date', freq='ME')])['quantity'].sum().reset_index(name='total_sales')


def calculate_number_of_months(df):
    
    return len(calculate_monthly_sales(df)['sales_date'].unique())


def calculate_mean(df): 
    
    return ((df.groupby(['product_id'])['quantity'].sum() / calculate_number_of_months(df)).reset_index(name='mean_sales'))


# Implementations

# Input is -> Processed Sales Data <-
def calculate_std_dev(df): 
    
    print("Calculating Product Wise Total Monthly Sales     ", end='\r')  # Comment out at time of deployment
    monthlyTotSales = calculate_monthly_sales(df)
    monthlyTotSales['product_sales_avg'] = (
    monthlyTotSales['product_id'].map(calculate_mean(df).set_index('product_id')['mean_sales']))
    monthlyTotSales['std_dev_buffer_value'] = (monthlyTotSales['total_sales'] - monthlyTotSales['product_sales_avg'])**2
    monthlyTotSales.drop(columns=['total_sales', 'product_sales_avg'], inplace=True)
    print("Calculating Product Wise Average Sales           ", end='\r')  # Comment out at time of deployment
    feature_vector = calculate_mean(df)
    print("Calculating Product Wise Standard Deviation      ", end='\r')  # Comment out at time of deployment
    feature_vector['std_dev'] = feature_vector['product_id'].map(
        np.sqrt(
            monthlyTotSales.groupby('product_id')['std_dev_buffer_value'].sum() * (1/(calculate_number_of_months(df)-1))
        ).round(0)
        .reset_index(name='std_dev')
        .set_index('product_id')['std_dev']
    )

    return feature_vector
# Returns a DataFrame with following features:
# Mean Sales, Standard Deviation


# Input is the Return of -> calculate_std_dev() <- function 
def calculate_coeff_of_variation(df):
    
    print("Calculating Product Wise Coefficient of Variation", end='\r')  # Comment out at time of deployment
    df['coeff_variation'] = df['std_dev'] / df['mean_sales']
    
    return df
# Returns a DataFrame with following features: 
# Mean Sales, Standard Deviation, Coefficient of Variation


# Input is the -> Processed Sales Data ( df ) <- and 
# the return of -> calculate_coeff_of_variation()  ( match ) <- function
def calculate_acf1(df, match): 

    buffer_df = calculate_monthly_sales(df)
    buffer_df['product_sales_avg'] = (buffer_df['product_id'].map(calculate_mean(df).set_index('product_id')['mean_sales']))
    buffer_df = buffer_df.sort_values(by=['product_id'])
    print("Calculating Product Wise Autocorrelation Lag1    ", end='\r')  # Comment out at time of deployment
    result = (
        buffer_df.groupby('product_id')['total_sales']
        .apply(formula_autocorr_lag1)
        .reset_index(name='autocorr_lag1')
    )
    
    return match.merge(result, on='product_id', how='left')
# Returns a DataFrame with the following features: 
# Mean Sales, Standard Deviation, Coefficient of Variation, Autocorrelation Lag1


# Input is the return of -> calculate_monthly_sales()  ( df ) <- function
# and the return of -> calculate_acf1()  ( match ) <- function
def calculate_seasonal_strength(df, match): 
    
    print("Calculating Product Wise Seasonal Strength       ", end='\r')  # Comment out at time of deployment
    seasonal_strength = df.groupby('product_id', group_keys=False).apply(lambda x: formula_seasonal_strength(x.set_index('sales_date')['total_sales']), include_groups=False).reset_index(name='seasonal_strength')

    return match.merge(seasonal_strength, on='product_id', how='left')
# Returns a DataFrame with the following features:
# Mean Sales, Standard Deviation, Coefficient of Variation, Autocorrelation Lag1, Seasonal Strength


# Input is the return of -> calculate_monthly_sales()  ( df ) <- function
# and the return of -> calculate_seasonal_strength()  ( match ) <- function
def calculate_peakiness_and_spike_index(df, match): 
    print("Calculating Product Wise Normalized Peakiness    ", end='\r')  # Comment out at time of deployment
    peakiness = df.groupby('product_id', group_keys=False).apply(lambda x: formula_normalized_peakiness(x.set_index('sales_date')['total_sales']), include_groups=False).reset_index()
    print("Calculating Product Wise Spike Fraction          ", end='\r')  # Comment out at time of deployment
    spike_fraction = df.groupby('product_id', group_keys=False).apply(lambda x: formula_spike_fraction(x.set_index('sales_date')['total_sales']), include_groups=False).reset_index()
    peakX = peakiness.merge(spike_fraction, on='product_id', how='left')

    return match.merge(peakX, on='product_id', how='left')
# Returns a DataFrame with the following features:
# Mean Sales, Standard Deviation, Coefficient of Variation, Autocorrelation Lag1, Seasonal Strength, 
# Normalized Peakiness, Spike Fraction


# Input is -> Processed Sales Data  ( df ) <- 
# and the return of -> calculate_peakiness_and_spike_index()  ( match ) <- function
def calculate_zero_fraction(df, match):

    print("Calculating Product Wise Zero Fraction           ", end='\r')  # Comment out at time of deployment
    all_months = pd.date_range(
        start=df['sales_date'].min().to_period('M').start_time,
        end=df['sales_date'].max().to_period('M').end_time,
        freq='ME'
    )
    
    product_month_index = pd.MultiIndex.from_product(
        [df['product_id'].unique(), all_months],
        names=['product_id', 'sales_date']
    )

    monthly_sales_full = (
        calculate_monthly_sales(df).set_index(['product_id', 'sales_date'])
        .reindex(product_month_index, fill_value=0)
        .reset_index()
    )

    zero_fraction = monthly_sales_full.groupby('product_id')['total_sales'].apply(lambda x: (x == 0).sum() / len(x)).reset_index(name='zero_fraction')

    return match.merge(zero_fraction, on='product_id', how='left')
# Returns a DataFrame with the following features:
# Mean Sales, Standard Deviation, Coefficient of Variation, Autocorrelation Lag1, Seasonal Strength, 
# Normalized Peakiness, Spike Fraction, Zero Fraction


# Input is the return of -> calculate_monthly_sales()  ( df ) <- function
# and the return of -> calculate_zero_fraction()  ( match ) <- function
def calculate_trend_slope(df, match):

    print("Calculating Product Wise Trend Slope             ", end='\r')  # Comment out at time of deployment
    df['month_number'] = (
        df.groupby('product_id')['sales_date'].rank(method='dense').astype(int)
    )
    trend_slope = df.groupby('product_id').apply(formula_trend_slope, include_groups=False).reset_index()
    print("Calculating Product Wise R-Square                ", end='\r')  # Comment out at time of deployment

    return match.merge(trend_slope, on='product_id', how='left')
# Returns a DataFrame with the following features:
# Mean Sales, Standard Deviation, Coefficient of Variation, Autocorrelation Lag1, Seasonal Strength, 
# Normalized Peakiness, Spike Fraction, Zero Fraction, Trend Slope, R-Square


# Input is the return of -> calculate_monthly_sales()  ( df ) <-
# and the return of -> calculate_trend_slope()  ( match ) <- 
def calculate_entropy(df, match):
    df['p_t'] = df.groupby('product_id')['total_sales'].transform(lambda x: x / x.sum())
    print("Calculating Product Wise Entropy                 ", end='\r')  # Comment out at time of deployment
    entropy = df.groupby('product_id').apply(formula_entropy, include_groups=False).reset_index(name='entropy')
    entropy['entropy_norm'] = entropy['entropy'] / np.log(df['sales_date'].nunique())
    
    return match.merge(entropy[['product_id', 'entropy_norm']], on='product_id', how='left')
# Returns a DataFrame with the following features:
# Mean Sales, Standard Deviation, Coefficient of Variation, Autocorrelation Lag1, Seasonal Strength, 
# Normalized Peakiness, Spike Fraction, Zero Fraction, Trend Slope, R-Square, Entropy


