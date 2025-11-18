import os

def print_tree(startpath, prefix=""):
    items = sorted(os.listdir(startpath))
    items = [i for i in items if i != ".git"]  # Ignore .git folder

    for index, item in enumerate(items):
        path = os.path.join(startpath, item)
        connector = "├─" if index < len(items) - 1 else "└─"
        if os.path.isdir(path):
            print(f"{prefix}{connector} Dir: {item}")
            # Continue deeper with updated prefix
            extension = "│  " if index < len(items) - 1 else "   "
            print_tree(path, prefix + extension)
        else:
            print(f"{prefix}{connector} File: {item}")

