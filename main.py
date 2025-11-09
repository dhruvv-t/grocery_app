import os
from standalone_scripts.realistic_dataset_generator import generate_sales_data
from end_to_end.workflow.file_paths import paths
from end_to_end.workflow.process_new_sales import process_sales
from standalone_scripts.generate_directory_tree import print_tree

print("Here is the project's File Structure: ")
root_dir = os.getcwd()
print_tree(root_dir)

if not os.path.exists(paths['sales']):
    generate_sales_data()

process_sales()


