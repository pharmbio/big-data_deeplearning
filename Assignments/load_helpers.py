import os
from importlib.machinery import SourceFileLoader

def load_helpers(base_path):
    """
    Loads the helpers, cnn_helper, and plot_helper modules from SharedUtils.

    If the base_path ends with 'Assignments', 'Labs', or 'LabsAndAssignments',
    that part will be removed to find the root path where SharedUtils lives.

    Args:
        base_path (str): Absolute path pointing to the current notebook folder.

    Returns:
        tuple: (helpers, cnn_helper, plot_helper) loaded as modules.
    """
    base_path = base_path.rstrip('/')
    subfolders_to_strip = ['Assignments', 'Labs', 'LabsAndAssignments']
    
    if any(base_path.endswith(folder) for folder in subfolders_to_strip):
        base_path = os.path.dirname(base_path)

    shared_utils_path = os.path.join(base_path, 'SharedUtils')

    def load_module(module_name):
        path = os.path.join(shared_utils_path, f"{module_name}.py")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {module_name}.py at: {path}")
        return SourceFileLoader(module_name, path).load_module()

    return (
        load_module("helpers"),
        load_module("cnn_helper"),
        load_module("plot_helper")
    )
