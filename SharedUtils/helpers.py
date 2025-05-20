# Helper functions for the labs

from datetime import datetime
import numpy as np
import cv2

import os
from skimage import io, img_as_uint
import numpy as np
import os
import csv
from collections import Counter

import itertools
import matplotlib.pyplot as plt

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

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def show_random_mask_examples(df, file_column, class_column, image_dir, n=6, seed=None):
    """
    Show a grid of randomly selected grayscale images from a dataframe of mask files.

    Args:
        df (pd.DataFrame): DataFrame with image filenames and class labels.
        file_column (str): Column containing filenames of images.
        class_column (str): Column containing class labels.
        image_dir (str): Path to the folder where the image files are located.
        n (int): Number of images to display. Must be a multiple of number of columns in the grid.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        None
    """
    if seed is not None:
        df = df.sample(n=n, random_state=seed)
    else:
        df = df.sample(n=n)

    print("Classes in sample:", df[class_column].unique())

    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Examples of mask images", fontsize=20)
    axes = ax.ravel()

    for i in range(n):
        row = df.iloc[i]
        img = io.imread(image_dir + row[file_column])
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(row[class_column], fontsize=12, pad=10)
        axes[i].set_axis_off()
        print(f"Image {i+1} – max: {img.max()}, min: {img.min()}")

    # Turn off any unused axes
    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    plt.close()

def apply_masks_to_images(images_path, masks_path, output_path, verbose=True):
    """
    Apply binary masks to grayscale images and save the masked output.
    
    Args:
        images_path (str): Path to folder containing original .tif images.
        masks_path (str): Path to folder containing corresponding .tif masks.
        output_path (str): Directory to save the masked output images.
        verbose (bool): Whether to print status messages.

    Returns:
        list: Filenames that were skipped due to empty (black) masks.
    """
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.tif')])
    mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith('.tif')])

    imgs = [io.imread(os.path.join(images_path, f)) for f in image_files]
    masks = [io.imread(os.path.join(masks_path, f)) for f in mask_files]

    os.makedirs(output_path, exist_ok=True)
    only_black_masks = []

    for i in range(len(image_files)):
        img_name = image_files[i]
        base_name = os.path.splitext(img_name)[0]

        imgnp = np.array(imgs[i])
        masknp = np.array(masks[i])

        if imgnp.shape != masknp.shape:
            if verbose:
                print(f"Could not merge {img_name} due to shape mismatch.")
            continue

        if masknp.max() == 0:
            only_black_masks.append(img_name)
            continue

        masked_np = imgnp * masknp
        masked_np = masked_np / masked_np.max()
        masked_np = img_as_uint(masked_np)

        outfile = os.path.join(output_path, base_name + '.tif')
        io.imsave(outfile, masked_np)

    if verbose:
        print("Skipped due to empty masks (no segmentation):")
        for name in only_black_masks:
            print("-", name)

    return only_black_masks


def filenames_to_csv(folder_path, output_csv, column_names=None):
    """
    Extracts filenames and their parts from a specified folder and writes to a CSV file.
    
    Parameters:
        folder_path (str): The path to the directory containing files.
        output_csv (str): The path to the output CSV file.
        column_names (list of str): Optional list of column names for the CSV. If not provided, 
                                   generic names will be used.
    """
    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Prepare data for CSV
    data = []
    for filename in files:
        parts = filename.split('_')
        row = [filename] + parts
        data.append(row)
    
    # Determine the maximum number of columns needed
    max_columns = max(len(row) for row in data)
    
    # If column names are not specified, generate generic column names
    if not column_names or len(column_names) < max_columns:
        column_names = ['filename'] + [f'part_{i}' for i in range(1, max_columns)]
    
    # Write to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names[:max_columns])
        for row in data:
            writer.writerow(row[:max_columns])