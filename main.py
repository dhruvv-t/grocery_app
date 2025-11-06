import os
from standalone_scripts.generate_sales_6m import generate_sales_data


# Check if sales data has already been generated.
sales_path = "end_to_end/datasets/sales_6m.csv"

if not os.path.exists(sales_path):
    generate_sales_data()

