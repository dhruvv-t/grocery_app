import pandas as pd
import numpy as np

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
        sales[sales['sales_date'].dt.year == 2023].groupby(['product_id', 'month', 'week_start', 'date'])['quantity'].sum().reset_index(name='daily_sales')
    ).groupby(['product_id'])['daily_sales'].agg(avg_daily_sales='mean', max_daily_sales='max').rest_index()

    one_year_stats['safety_stock'] = ((one_year_stats['max_daily_sales'] - one_year_stats['avg_daily_sales']) * products['lead_time_days']).round(0)

    return one_year_stats[['product_id', 'safety_stock']]


def calculate_reorder_level():
    print('calculate Reorder Level.')