import numpy as np
import os
import csv
from collections import Counter

def getClassSizes(generator):
    counter = Counter(generator.classes)
    max_val = float(max(counter.values()))
    class_sizes = {class_id: num_images for class_id, num_images in counter.items()}
    return class_sizes

def getClassWeights(generator):
    counter = Counter(generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
    return class_weights

import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [c for v,c in sorted(classes.items())], rotation=90)
    plt.yticks(tick_marks, [c for v,c in sorted(classes.items())])
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.0f} %'.format(100*cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_from_generator (model, generator):
    y_true = generator.classes
    predictions = model.predict(generator)
    y_predict = np.argmax(predictions, axis=1)
    acc = sum(1 for x,y in zip(y_predict,y_true) if x == y) / len(y_true)
    print ("Accuracy:", acc)
    conf_mat = confusion_matrix(y_true, y_predict)
    plot_confusion_matrix(conf_mat, {v: k for k, v in generator.class_indices.items()})


from tensorflow.keras.callbacks import Callback
from matplotlib.colors import ListedColormap

def plot_prediction (model, test_batch, num_plot):
    cMap = ListedColormap(['red', 'lime', 'blue'])
    a, b = test_batch
    pred = model.predict(a)
    fig, axs = plt.subplots(num_plot, 4,figsize=(16,num_plot*4), dpi=45, squeeze=False)
    for i in range(num_plot):
        axs[i,0].imshow(a[i])
        axs[i,0].axis('off')
        axs[i,1].imshow(b[i,:,:,0],cmap=cMap, vmax=3 - 0.5, vmin=-0.5)
        axs[i,1].axis('off')
        axs[i,2].imshow(pred[i])
        axs[i,2].axis('off')
        axs[i,3].imshow(np.argmax(pred[i,:,:,:], axis=2),cmap=cMap, vmax=3 - 0.5, vmin=-0.5)
        axs[i,3].axis('off')
    axs[0,0].set_title('Image')
    axs[0,1].set_title('Ground Truth')
    axs[0,2].set_title('Prediction')
    axs[0,3].set_title('Argmax of Prediction')
        
    plt.tight_layout()
    plt.show()
    
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
    return output_csv

class PlottingKerasCallback(Callback):
    def __init__(self, test_batch, num_plot):
        self.test_batch = test_batch
        self.num_plot = num_plot
    
    def on_epoch_end(self, epoch, logs=None):
        plot_prediction(self.model, self.test_batch, self.num_plot)
        

