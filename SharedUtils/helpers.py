# Helper functions for the labs
# NOTE: students do *not* have to understand these functions nor should they study them in detail

from datetime import datetime
import numpy as np
import cv2

def start_time():
    """
    Prints the current datetime to indicate the start of a run.
    """
    print("Starting run at: " + str(datetime.now()))

def end_time():
    """
    Prints the current datetime to indicate the end of a run.
    """
    print("Run finished at: " + str(datetime.now()))

def get_image_data_flat_from_file(data_directory, image_paths):
    """
    Loads and flattens image data from the specified paths.

    Args:
        data_directory (str): Base directory where image files are stored.
        image_paths (pd.DataFrame): DataFrame with file names in a column (flattened to list).

    Returns:
        np.ndarray: Flattened image data of shape (num_images, height * width * channels).
    """
    file_names = image_paths.values.flatten()
    image_data = np.array([
        np.array(cv2.imread(data_directory + file_name))
        for file_name in file_names
    ])
    flattened_image_data = image_data.reshape(image_data.shape[0], -1)
    return flattened_image_data