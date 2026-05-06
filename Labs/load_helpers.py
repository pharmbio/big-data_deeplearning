import os
from importlib.machinery import SourceFileLoader

def load_helpers(base_path, load_cnn_helper=True, load_plot_helper=True):
    """
    Loads selected helper modules from SharedUtils.

    Args:
        base_path (str): Absolute path pointing to the current notebook folder.
        load_cnn_helper (bool): Whether to load cnn_helper.
        load_plot_helper (bool): Whether to load plot_helper.

    Returns:
        tuple: (helpers, cnn_helper, plot_helper) loaded as modules or None.
    """
    base_path = base_path.rstrip('/')
    subfolders_to_strip = ['Assignments', 'Labs', 'LabsAndAssignments']
    
    if any(base_path.endswith(folder) for folder in subfolders_to_strip):
        base_path = os.path.dirname(base_path)

    shared_utils_path = os.path.join(base_path, 'SharedUtils')
    print("Fetching helpers from: " + str(shared_utils_path))

    def load_module(module_name):
        path = os.path.join(shared_utils_path, f"{module_name}.py")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {module_name}.py at: {path}")
        return SourceFileLoader(module_name, path).load_module()

    helpers = load_module("helpers")
    cnn_helper = load_module("cnn_helper") if load_cnn_helper else None
    plot_helper = load_module("plot_helper") if load_plot_helper else None

    return helpers, cnn_helper, plot_helper