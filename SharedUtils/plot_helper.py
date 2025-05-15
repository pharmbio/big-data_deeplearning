import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import itertools
import math
import os
import glob


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
    
    
    
def show_random_images(data_directory, df_labels, file_column = "Filenames", class_column = "Class"):
    figure, ax = plt.subplots(2, 3, figsize=(14, 10))
    figure.suptitle("Examples of images", fontsize=20)
    axes = ax.ravel()

    df_images_to_show = df_labels.sample(8)


    for i in range(len(axes)):
        row = df_images_to_show.iloc[[i]]
        print (data_directory + row[file_column].values[0])
        random_image = Image.open(data_directory + row[file_column].values[0])
        axes[i].set_title(row[class_column].values[0], fontsize=14) 
        axes[i].imshow(random_image)
        axes[i].set_axis_off()

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    plt.close()
    
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_random_images_hpv_specific(data_directory, number_of_images=5):
    image_files = glob.glob(os.path.join(data_directory, '*'))
    n_images = len(image_files)

    if n_images < number_of_images:
        print("Not enough images available to display the requested number.")
        return

    selected_indices = np.random.choice(n_images, number_of_images, replace=False)
    selected_images = [image_files[i] for i in selected_indices]

    path_parts = os.path.normpath(data_directory).split(os.sep)
    if len(path_parts) >= 2:
        folder_names = " ".join(path_parts[-2:])
    else:
        folder_names = path_parts[-1]

    first_image = Image.open(selected_images[0])
    aspect_ratio = first_image.size[0] / first_image.size[1]
    fig, axes = plt.subplots(1, number_of_images, figsize=(number_of_images * aspect_ratio, 2 * aspect_ratio))

    plt.subplots_adjust(left=0, right=1, top=0.5, wspace=0.04, hspace= aspect_ratio* 0.2)
    fig.suptitle(f"Random images from {folder_names}", fontsize=20, y=0.92)  # Adjust y-position of title

    for ax, image_path in zip(axes, selected_images):
        img = Image.open(image_path)
        img = np.array(img)
        ax.imshow(img, cmap='gray')
        ax.set_title(os.path.basename(image_path), fontsize=10)
        ax.set_axis_off()

    plt.show()
    
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
    
def plot_layer_weights(weights, plot_title="Layer Weights Visualization", number_of_rows=2, visualize_in_2d=False, weight_in_2d = False):
    number_of_nodes = weights.shape[-1]
    number_of_columns = math.ceil(number_of_nodes / number_of_rows)
    fig_width = 4 * number_of_columns
    fig_height = 4 * number_of_rows if visualize_in_2d else 2 * number_of_rows
    figure, axes_array = plt.subplots(number_of_rows, number_of_columns, figsize=(fig_width, fig_height))
    figure.suptitle(plot_title, fontsize=20, y=1.05)
    
    axes_list = axes_array.flatten() if number_of_nodes > 1 else [axes_array]
    
    for idx, ax in enumerate(axes_list):
        if idx < number_of_nodes:
            ax.set_title(f"Node {idx + 1}", fontsize=14)
            ax.axis('off')
            if visualize_in_2d:
                if weight_in_2d:
                    filters = weights[:, :, :, idx]
                    image = filters[:,:,0]
                else:    
                    side_of_image = int(math.sqrt(weights.shape[0]))
                    image = np.reshape(weights[:, idx], (side_of_image, side_of_image))
                im = ax.imshow(image, cmap='jet')
            else:
                node_weights = weights[:, idx]
                ax.imshow(node_weights[np.newaxis, :], cmap='jet', aspect='auto')
        else:
            ax.set_visible(False)
    
    if visualize_in_2d:
        cbar_ax = figure.add_axes([0.92, 0.15, 0.02, 0.7])
        figure.colorbar(im, cax=cbar_ax)
    else:
        cbar_ax = figure.add_axes([0.1, 0.05, 0.8, 0.02])
        norm = plt.Normalize(np.min(weights), np.max(weights))
        sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        figure.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    plt.subplots_adjust(right=0.9 if visualize_in_2d else 1.0, wspace=0.05, hspace=0.3)
    plt.show()
