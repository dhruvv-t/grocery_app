import os
from standalone_scripts.generate_sales_6m import generate_sales_data
from end_to_end.workflow.file_paths import paths
from end_to_end.workflow.process_new_sales import process_sales

if not os.path.exists(paths['sales']):
    generate_sales_data()

process_sales()


