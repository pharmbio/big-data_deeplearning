import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from collections import Counter
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import Callback
from matplotlib.colors import ListedColormap
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_model_predictions(model, generator):
    """
    Gets predicted labels and true labels from a Keras model and generator.
    Works for binary and multiclass classification.
    """
    check_generator_not_shuffled(generator)

    if hasattr(generator, "reset"):
        generator.reset()

    y_pred_prob = model.predict(generator)
    y_true = generator.classes

    if y_pred_prob.ndim == 1 or y_pred_prob.shape[1] == 1:
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)

    return y_true, y_pred


def evaluate_and_plot_model_performance(
    model,
    generator,
    dataset_name="validation data",
    show_confusion_matrix=True,
    show_percentage=True,
    show_metrics=True,
    class_names=None
):
    """
    Runs a full evaluation pipeline with confusion matrices, metrics, and a classification report.
    """
    if class_names is None:
        class_names = get_class_names_from_generator(generator)

    y_true, y_pred = get_model_predictions(model, generator)

    print(f"\n--- Evaluation on {dataset_name} ---\n")

    if show_confusion_matrix:
        plot_confusion_matrix_from_predictions(
            y_true,
            y_pred,
            class_names=class_names,
            percentage=False,
            title=f'Confusion matrix for {dataset_name}'
        )

    if show_percentage:
        plot_confusion_matrix_from_predictions(
            y_true,
            y_pred,
            class_names=class_names,
            percentage=True,
            title=f'Confusion matrix (%) for {dataset_name}'
        )

    if show_metrics:
        print_model_metrics_from_predictions(
            y_true,
            y_pred,
            class_names=class_names
        )


def print_model_metrics_from_predictions(y_true, y_pred, class_names=None):
    """
    Prints accuracy, balanced accuracy, F1 scores, and a classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Balanced accuracy: {balanced_accuracy:.4f}')
    print(f'Macro F1 score: {macro_f1:.4f}')
    print(f'Weighted F1 score: {weighted_f1:.4f}')
    print('')

    if class_names is not None:
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))


def evaluate_model_metrics(model, generator, class_names=None):
    """
    Gets model predictions and prints model metrics.
    """
    y_true, y_pred = get_model_predictions(model, generator)

    print_model_metrics_from_predictions(
        y_true,
        y_pred,
        class_names=class_names
    )


def plot_confusion_matrix_from_predictions(
    y_true,
    y_pred,
    class_names=None,
    percentage=True,
    title='Confusion matrix',
    cmap=plt.cm.Blues
):
    """
    Plots a confusion matrix from true and predicted labels.
    """
    if percentage:
        normalize = 'true'
        values_format = '.0%'
    else:
        normalize = None
        values_format = 'd'

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize=normalize,
        values_format=values_format,
        cmap=cmap
    )

    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_model_confusion_matrix(
    model,
    generator,
    class_names=None,
    percentage=True,
    title='Confusion matrix',
    cmap=plt.cm.Blues
):
    """
    Gets model predictions and plots a confusion matrix.
    """
    if class_names is None:
        class_names = get_class_names_from_generator(generator)

    y_true, y_pred = get_model_predictions(model, generator)

    plot_confusion_matrix_from_predictions(
        y_true,
        y_pred,
        class_names=class_names,
        percentage=percentage,
        title=title,
        cmap=cmap
    )




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


def plot_prediction(model, test_batch, num_plot):
    cMap = ListedColormap(['red', 'lime', 'blue'])
    a, b = test_batch
    pred = model.predict(a)
    fig, axs = plt.subplots(num_plot, 4, figsize=(16, num_plot*4), dpi=45, squeeze=False)
    for i in range(num_plot):
        axs[i,0].imshow(a[i])
        axs[i,0].axis('off')
        axs[i,1].imshow(b[i,:,:,0], cmap=cMap, vmax=3 - 0.5, vmin=-0.5)
        axs[i,1].axis('off')
        axs[i,2].imshow(pred[i])
        axs[i,2].axis('off')
        axs[i,3].imshow(np.argmax(pred[i,:,:,:], axis=2), cmap=cMap, vmax=3 - 0.5, vmin=-0.5)
        axs[i,3].axis('off')
    axs[0,0].set_title('Image')
    axs[0,1].set_title('Ground Truth')
    axs[0,2].set_title('Prediction')
    axs[0,3].set_title('Argmax of Prediction')
        
    plt.tight_layout()
    plt.show()


class PlottingKerasCallback(Callback):
    def __init__(self, test_batch, num_plot):
        self.test_batch = test_batch
        self.num_plot = num_plot
    
    def on_epoch_end(self, epoch, logs=None):
        plot_prediction(self.model, self.test_batch, self.num_plot)


def show_last_layers(model, n=8):
    """
    Prints the last n layers of a model and their output shapes.
    """
    for layer in model.layers[-n:]:
        try:
            shape = layer.output.shape
        except:
            shape = "Unknown"
        print(f"{layer.name:30} {shape}")

def get_class_names_from_generator(generator):
    """
    Gets class names from a Keras generator.
    """
    return list(generator.class_indices.keys())


def check_generator_not_shuffled(generator):
    """
    Warns if a generator is shuffled during evaluation.
    """
    if getattr(generator, "shuffle", False):
        warnings.warn(
            "This generator has shuffle=True. For evaluation, prediction order may not match generator.classes. "
            "Use shuffle=False for validation or test generators.",
            UserWarning
        )

def print_generator_summary(generator):
    """
    Prints a short summary of a Keras image generator.
    """
    print(f"Number of images: {generator.samples}")
    print(f"Number of classes: {generator.num_classes}")
    print(f"Batch size: {generator.batch_size}")
    print(f"Shuffle: {getattr(generator, 'shuffle', 'Unknown')}")
    print("")
    print("Classes:")

    for class_name, class_index in generator.class_indices.items():
        print(f"{class_index}: {class_name}")


def setup_image_generators(
    df_train, df_validation, df_test,
    image_shape, data_directory,
    filename_column, class_column,
    batch_size=8, color_mode='grayscale',
    train_augmentations=None,
    valid_augmentations=None,
    test_augmentations=None
):
    """
    Creates train/validation/test generators using Keras' ImageDataGenerator.

    Args:
        df_train, df_validation, df_test (pd.DataFrame): DataFrames with filenames and class labels.
        image_shape (tuple): Target image size as (height, width).
        data_directory (str): Path to image folder.
        filename_column (str): Name of the column containing filenames.
        class_column (str): Name of the column containing class labels.
        batch_size (int): Batch size for the generators.
        color_mode (str): 'grayscale' or 'rgb'.
        train_augmentations (dict): Extra args for ImageDataGenerator for training.
        valid_augmentations (dict): Extra args for validation generator.
        test_augmentations (dict): Extra args for test generator.

    Returns:
        dict: {
            'train_generator': train_gen,
            'valid_generator': valid_gen,
            'test_generator': test_gen,
            'train_steps': int,
            'valid_steps': int
        }
    """
    
    base_config = dict(
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True
    )

    train_config = {**base_config, 'horizontal_flip': True, 'vertical_flip': True}
    if train_augmentations:
        train_config.update(train_augmentations)

    valid_config = base_config.copy()
    if valid_augmentations:
        valid_config.update(valid_augmentations)

    test_config = base_config.copy()
    if test_augmentations:
        test_config.update(test_augmentations)

    train_datagen = ImageDataGenerator(**train_config)
    valid_datagen = ImageDataGenerator(**valid_config)
    test_datagen = ImageDataGenerator(**test_config)

    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        directory=data_directory,
        x_col=filename_column,
        y_col=class_column,
        class_mode='categorical',
        batch_size=batch_size,
        target_size=image_shape,
        color_mode=color_mode,
        shuffle=True
    )

    valid_gen = valid_datagen.flow_from_dataframe(
        df_validation,
        directory=data_directory,
        x_col=filename_column,
        y_col=class_column,
        class_mode='categorical',
        batch_size=batch_size,
        target_size=image_shape,
        color_mode=color_mode,
        shuffle=False
    )

    test_gen = test_datagen.flow_from_dataframe(
        df_test,
        directory=data_directory,
        x_col=filename_column,
        y_col=class_column,
        class_mode='categorical',
        batch_size=batch_size,
        target_size=image_shape,
        color_mode=color_mode,
        shuffle=False
    )

    train_steps = max(train_gen.n // batch_size, 1)
    valid_steps = max(valid_gen.n // batch_size, 1)

    return {
        'train_generator': train_gen,
        'valid_generator': valid_gen,
        'test_generator': test_gen,
        'train_steps': train_steps,
        'valid_steps': valid_steps
    }