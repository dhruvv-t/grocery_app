import pandas as pd
import os
import csv
from end_to_end.workflow.file_paths import paths


# Function to set a week start tag for every week in the sales data
def week_start_date(dt_series):
    weekday = dt_series.dt.weekday
    week_start = (dt_series - pd.to_timedelta(weekday, unit='d')).dt.normalize().dt.date
    return week_start

# Function to set month start tag for every month in the sales data
def month_number(dt_series):
    month_number = dt_series.dt.month
    year = dt_series.dt.year
    return month_number, year

# Process new Sales Data
def process_sales():
    
    # Read the .csv files
    print("Reading sales.csv file.")                          # Checkpoints -> Remove them in production
    sales = pd.read_csv(paths['sales'])
    products =  pd.read_csv(paths['products'])
    empty_df = pd.DataFrame(columns=sales.columns)

    # Make the date column timezone aware
    sales['sales_id'] = range(1, len(sales) + 1) 
    sales['sales_date'] = range(1, len(sales['sales_date']))
    print("Changing sales_date to timezone aware format.")                          # Checkpoints -> Remove them in production
    sales['sales_date'] = pd.to_datetime(sales['sales_date'])
    print("Creating new columns for week and month start dates.")                          # Checkpoints -> Remove them in production
    sales['week_start'] = week_start_date(sales['sales_date'])
    sales['month'], sales['year'] = month_number(sales['sales_date'])
    
    # Adding the Total Selling Price Column
    print("Calculating total_price for each sale.")                          # Checkpoints -> Remove them in production
    temp = sales[['sales_id', 'product_id', 'quantity']].merge(products[['product_id', 'selling_price']], on='product_id', how='left')
    sales['total_price'] = temp['quantity'] * temp['selling_price']

    # Save as final csvs
    print("Adding Processed Sales data to the pre-existing data.")                          # Checkpoints -> Remove them in production
    if os.path.exists(paths['sales_pro']):
        sales.to_csv(paths['sales_pro'], mode='a', index=False, header=False)
    else:
        sales.to_csv(paths['sales_pro'], index=False)

    # Reset original file for new data
    print("Emptying the recent sales data for the next day.")                          # Checkpoints -> Remove them in production
    empty_df.to_csv(paths['sales'], index=False)
    

