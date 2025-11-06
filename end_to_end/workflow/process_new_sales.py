import pandas as pd
from end_to_end.workflow.file_paths import paths


# Function to set a week start tag for every week in the sales data
def week_start_date(dt_series):
    weekday = dt_series.dt.weekday
    week_start = (dt_series - pd.to_timedelta(weekday, unit='d')).dt.normalize().dt.date
    return week_start

# Function to set month start tag for every month in the sales data
def month_start_date(dt_series):
    month_start = dt_series.dt.to_period('M').dt.to_timestamp().dt.date
    return month_start


# Process new Sales Data
def process_sales():
    
    # Read the .csv files
    sales = pd.read_csv(paths['sales'])
    products =  pd.read_csv(paths['products'])
    empty_df = pd.DataFrame(columns=sales.columns)

    # Make the date column timezone aware
    sales['sales_date'] = pd.to_datetime(sales['sales_date'])

    sales['week_start'] = week_start_date(sales['sales_date'])
    sales['month_start'] = month_start_date(sales['sales_date'])
    
    # Adding the Total Selling Price Column
    temp = sales[['sales_id', 'product_id', 'quantity']].merge(products[['product_id', 'selling_price']], on='product_id', how='left')
    sales['total_price'] = temp['quantity'] * temp['selling_price']

    # Save as final csv
    sales.to_csv(paths['sales_pro'], index=False)

    # Reset original file for new data
    empty_df.to_csv(paths['sales'], index=False)

