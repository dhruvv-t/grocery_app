import pandas as pd
import numpy as np
import calendar
import math

from end_to_end.workflow.file_paths import paths


# Returns 2 DataFrames
def load_dataset():

    print("Loading Datasets to work on.                             ", end='\r')
    sales = pd.read_csv(paths['sales_pro'])
    products = pd.read_csv(paths['products'])
    sales['sales_date'] = pd.to_datetime(sales['sales_date'])
    sales['date'] = sales['sales_date'].dt.day
    return sales, products


# Returns 2 DataFrames
def calculate_safety_stock():

    sales, products = load_dataset()

    one_year_stats = (
        (
            sales[sales['sales_date'].dt.year == 2023]
        ).groupby(['product_id', 'month', 'week_start', 'date'])['quantity']
        .sum().reset_index(name='daily_sales')
    ).groupby(['product_id'])['daily_sales'].agg(avg_daily_sales='mean', max_daily_sales='max').reset_index()

    print("Calculating Safety Stock for each product                ", end='\r')
    one_year_stats['safety_stock'] = (
        (one_year_stats['max_daily_sales'] - one_year_stats['avg_daily_sales']) * products['lead_time_days']
    ).round(0)

    products = (products.merge(one_year_stats[['product_id', 'safety_stock']], on='product_id', how='left'))[['product_id', 'category_id', 'product_name', 'vitality_days', 'lead_time_days', 'safety_stock']]
    return products, sales


# Doesn't Return anything, Saves a .csv file with Reorder Level and Date to Recalculate it on.
def initialize_reorder_level():

    categorized_products = pd.read_csv(paths['catzd_products'])
    products, sales = calculate_safety_stock()

    sales_to_analyze = sales[sales['sales_date'].dt.year == 2023].merge(
        categorized_products,
        on='product_id',
        how='left'
    )
    products = products.merge(categorized_products, on='product_id', how='left')

    print("Calculating Reorder Level for Stable products            ", end='\r') 
    stable_reorder_initialize = (
        sales_to_analyze[sales_to_analyze['category'] == 'Stable']
        .groupby('product_id')['quantity'].sum() / 364
    ).reset_index()
    stable_reorder_initialize['recalculate_on'] = (
        (
            sales_to_analyze[sales_to_analyze['category'] == 'Stable']
            .groupby('product_id')['sales_date'].max() + pd.DateOffset(months=1, days=1)
        ).dt.date
    ).reset_index()['sales_date']

    print("Calculating Reorder Level for Trending products          ", end='\r') 
    trending_reorder_initialize = (sales_to_analyze[
        (sales_to_analyze['category'] == 'Trending') 
        & (sales_to_analyze['sales_date'].dt.month 
           == sales_to_analyze['sales_date'].max().month)
    ].groupby('product_id')['quantity'].sum() / calendar.monthrange(
        sales_to_analyze['sales_date'].max().year, sales_to_analyze['sales_date'].max().month
    )[1]).reset_index()
    trending_reorder_initialize['recalculate_on'] = (
        (
            sales_to_analyze[sales_to_analyze['category'] == 'Stable']
            .groupby('product_id')['sales_date'].max() + pd.Timedelta(weeks=2)
        ).dt.date
    ).reset_index()['sales_date']

    print("Calculating Reorder Level for Volatile products          ", end='\r') 
    volatile_reorder_initialize = (sales_to_analyze[
        (sales_to_analyze['category'] == 'Volatile') 
        & (sales_to_analyze['sales_date'] 
        >= (sales_to_analyze['sales_date'].max() - pd.Timedelta(days=14)))
    ].groupby('product_id')['quantity'].sum() / 14).reset_index()
    volatile_reorder_initialize['recalculate_on'] = (
        (
            sales_to_analyze[sales_to_analyze['category'] == 'Stable']
            .groupby('product_id')['sales_date'].max() + pd.DateOffset(days=3)
        ).dt.date
    ).reset_index()['sales_date']

    products = products.merge(
        pd.concat([
            stable_reorder_initialize, 
            trending_reorder_initialize, 
            volatile_reorder_initialize
        ], ignore_index=True), on='product_id', how='left'
    ).rename(columns={'quantity': 'avg_daily_demand'})

    print("Handling NA values in the because of missing values      ", end='\r') 
    products['avg_daily_demand'] = products['avg_daily_demand'].fillna(0)
    products['recalculate_on'] = products['recalculate_on'].fillna(
        products[products['category'] == "Trending"]['recalculate_on'].value_counts().index[0]
    )

    products['avg_daily_demand'] = products['avg_daily_demand'].apply(math.ceil)
    print("Compiling Reorder Levels for all Three Categories        ", end='\r') 
    products['reorder_level'] = (products['avg_daily_demand'] * products['lead_time_days']) + products['safety_stock']
    print("Calculating Perishability Factor for each Product        ", end='\r') 
    products['perishability_factor'] = (products['vitality_days'] / 10).clip(upper=1)
    print("Calculating Category Factor for each Category            ", end='\r') 
    products['category_factor'] = products['category_id'].map((products.groupby('category_id')['avg_daily_demand'].mean() / products.groupby('category_id')['avg_daily_demand'].std()) + 1)
    print("Calculating Reorder Quantity for each Products           ", end='\r') 
    products['reorder_quantity'] = (
        products['avg_daily_demand'] * np.minimum(
                (products['vitality_days'] * products['perishability_factor']),
                (products['lead_time_days'] * products['category_factor'])
            )
    )

    print(f'Saving calculated features in as a CSV file: {paths['reorder_specifics']}') 
    products[['product_id', 'category', 'reorder_level', 'reorder_quantity', 'recalculate_on']].to_csv(paths['reorder_specifics'], index=False)


# Saves the New Reorder Level and the Next Date to Recalculate it on, in the pre-existing .csv file.
def calculate_reorder_level():

    recalculate = pd.read_csv(paths['reorder_levels'])
    print('Recalculate Reorder Levels')


