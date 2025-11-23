# Imports
import os

from standalone_scripts import generate_directory_tree, realistic_dataset_generator

from end_to_end.workflow import process_new_sales, file_paths
from end_to_end.workflow.product_categorization import categorize_products
from end_to_end.workflow.inventory_mgmt import calculate_reorder_specifications



# Displaying the Directory Tree
print("Here is the project's File Structure: ")
root_dir = os.getcwd()
generate_directory_tree.print_tree(root_dir)

# # Generating the dataset if it doesn't exits
# if not os.path.exists(file_paths.paths['sales']):
#     realistic_dataset_generator.generate_sales_data()

# # Pre-processing the sales data
# process_new_sales.process_sales()

# # Categorize products into categories based on their sales patterns
# categorize_products.categorize_products_on_sales()

# # Calculate Reorder Levels and Quantities for each product
# calculate_reorder_specifications.initialize_reorder_level()