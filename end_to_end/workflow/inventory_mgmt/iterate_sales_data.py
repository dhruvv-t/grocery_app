import pandas as pd

from end_to_end.workflow.inventory_mgmt import calculate_reorder_specifications



def iterate_over_sales_data(sales):
    reorder_sp, sales = calculate_reorder_specifications.initialize_reorder_level()
    
