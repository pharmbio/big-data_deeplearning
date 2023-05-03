import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import itertools
import math


def plot_history(model_history, model_name):
    # Code copied from Phil Harrissons pbseq lab and then adapted
    fig = plt.figure(figsize=(15, 5), facecolor='w')
    ax = fig.add_subplot(131)
    ax.plot(model_history.history['loss'])
    ax.plot(model_history.history['val_loss'])
    ax.set(title = model_name + ': Model loss', ylabel='Loss', xlabel='Epoch')
    ax.legend(['train', 'valid'], loc='upper right')
    
    # ax = fig.add_subplot(132)
    # ax.plot(np.log(model_history.history['loss'])) # THIS IS wrong
    # ax.plot(np.log(model_history.history['val_loss']))
    # ax.set(title=model_name + ': Log model loss', ylabel='Log loss', xlabel='Epoch')
    # ax.legend(['Train', 'Test'], loc='upper right')    

    ax = fig.add_subplot(133)
    ax.plot(model_history.history['accuracy'])
    ax.plot(model_history.history['val_accuracy'])
    ax.set(title=model_name + ': Model accuracy', ylabel='Accuracy', xlabel='Epoch')
    ax.legend(['train', 'valid'], loc='upper right')
    plt.show()
    plt.close()
    
    
    
def show_random_images(data_directory, df_labels, file_column):
    figure, ax = plt.subplots(2, 3, figsize=(14, 10))
    figure.suptitle("Examples of images", fontsize=20)
    axes = ax.ravel()

    df_images_to_show = df_labels.sample(8)


    for i in range(len(axes)):
        row = df_images_to_show.iloc[[i]]
        random_image = Image.open(data_directory + row["Filenames"].values[0])
        axes[i].set_title(row["Class"].values[0], fontsize=14) 
        axes[i].imshow(random_image)
        axes[i].set_axis_off()

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    plt.close()
    
    
def plot_confusion_matrix(confusion_matrix, classes, title='Confusion matrix', save = False, cmap=plt.cm.Blues, scale_colours = False):
    thresh = 0.5
    vmin=0
    vmax=1
    if scale_colours:
        thresh = confusion_matrix.max() / 2.
        vmin = confusion_matrix.min()
        vmax = confusion_matrix.max()

    plt.close('all')
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    for row, column in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        value = confusion_matrix[row, column]
        plt.text(column, row, '{:.0f} %'.format(100*value), horizontalalignment="center", color="white" if value > thresh else "black")
        # plt.txt wants x then y coordinates

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save == True:
        plt.savefig(title + '.png')
    
    plt.show()
    

def plot_2d_layer(weights, plot_title= "Layer visualized in 2D", number_of_rows = 2):
    # weights shape: (weights_in_a_node, number_of_nodes)
    # code based on:https://thispointer.com/python-convert-a-1d-array-to-a-2d-numpy-array-or-matrix/
    number_of_nodes = int(weights.shape[1])
    side_of_image = int(math.sqrt(weights.shape[0]))
    image_stack = np.reshape(weights, (side_of_image, side_of_image,number_of_nodes))

    number_of_rows = 2
    number_of_columns = (number_of_nodes // number_of_rows) + (1 if number_of_nodes % number_of_rows > 0 else 0)

    figure, ax = plt.subplots(number_of_rows,number_of_columns , figsize=(14, 10))
    figure.suptitle(plot_title, fontsize=20)
    axes = ax.ravel()

    for i in range(0,number_of_nodes):
        image = image_stack[:,:,i:i+1]
        image = image[:,:,0]
        axes[i].set_title("Node" + str(i+1), fontsize=14) 
        axes[i].imshow(image)
        axes[i].set_axis_off()
        
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def plot_1d_layer(weights, plot_title= "Layer visualized in 1d", number_of_rows = 2):
    # weights shape: (weights_in_a_node, number_of_nodes)
    number_of_nodes = int(weights.shape[1])
    image_stack = weights

    number_of_columns = (number_of_nodes // number_of_rows) + (1 if number_of_nodes % number_of_rows > 0 else 0)

    figure, ax = plt.subplots(number_of_rows,number_of_columns , figsize=(14, 10))
    figure.suptitle(plot_title, fontsize=20)
    axes = ax.ravel()

    for i in range(0,number_of_nodes):
        image = image_stack[:,i:i+1]
        image = image[:,0]
        axes[i].set_title("Node" + str(i+1), fontsize=14) 
        axes[i].plot(image)
        axes[i].set_axis_off()
        
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
