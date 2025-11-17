import pandas as pd
import numpy as np
import calendar
import math

from end_to_end.workflow.file_paths import paths


def load_dataset():

    sales = pd.read_csv(paths['sales_pro'])
    products = pd.read_csv(paths['products'])
    sales['sales_date'] = pd.to_datetime(sales['sales_date'])
    sales['date'] = sales['sales_date'].dt.day
    return sales, products


def calculate_safety_stock():

    sales, products = load_dataset()

    one_year_stats = (
        (
            sales[sales['sales_date'].dt.year == 2023]
        ).groupby(['product_id', 'month', 'week_start', 'date'])['quantity']
        .sum().reset_index(name='daily_sales')
    ).groupby(['product_id'])['daily_sales'].agg(avg_daily_sales='mean', max_daily_sales='max').rest_index()

    one_year_stats['safety_stock'] = (
        (one_year_stats['max_daily_sales'] - one_year_stats['avg_daily_sales']) * products['lead_time_days']
    ).round(0)

    products = (products.merge(one_year_stats[['products_id', 'safety_stock']], on='product_id', how='left'))[['product_id', 'product_name', 'vitality_days', 'lead_time_days', 'safety_stock']]
    return products, sales


def initialize_reorder_level():

    categorized_products = pd.read_csv(paths['catzd_products'])
    products, sales = calculate_safety_stock()

    sales_to_analyze = sales[sales['sales_date'].dt.year == 2023].merge(
        categorized_products, on='product_id', how='left'
    )
    products = products.merge(categorized_products, on='product_id', how='left')

    stable_reorder_initialize = (sales_to_analyze[sales_to_analyze['category'] == 'Stable'].groupby('product_id')['quantity'].sum() / 364).reset_index()
    stable_reorder_initialize['recalculate_on'] = sales_to_analyze
    trending_reorder_initialize = (sales_to_analyze[
        (sales_to_analyze['category'] == 'Trending') & (sales_to_analyze['sales_date'].dt.month == sales_to_analyze['sales_date'].max().month)
    ].groupby('product_id')['quantity'].sum() / calendar.monthrange(sales_to_analyze['sales_date'].max().year, sales_to_analyze['sales_date'].max().month)[1]).reset_index()
    volatile_reorder_initialize = (sales_to_analyze[
        (sales_to_analyze['category'] == 'Volatile') & (sales_to_analyze['sales_date'] >= (sales_to_analyze['sales_date'].max() - pd.Timedelta(days=14)))
    ].groupby('product_id')['quantity'].sum() / 14).reset_index()

    products = products.merge(
        pd.concat([stable_reorder_initialize, trending_reorder_initialize, volatile_reorder_initialize], ignore_index=True), on='product_id', how='left'
    ).rename(columns={'quantity': 'avg_daily_demand'}).fillna(0)

    products['avg_daily_demand'] = products['avg_daily_demand'].apply(math.ceil)
    products['reorder_level'] = (products['avg_daily_demand'] * products['lead_time_days']) + products['safety_stock']
    
    return products

def calculate_reorder_level():
    recalculate = pd.read_csv(paths['reorder_levels'])

