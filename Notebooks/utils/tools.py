import cv2
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import ImageOps, Image
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.optimizers import Adam,Adagrad,Adadelta,RMSprop
from tensorflow.keras.regularizers import L1,L2


def preprocess_img(paths: list,
                   width: int,
                   height: int,
                   resize: bool = False,
                   interpolation=None,
                   normalization: bool = True,
                   color=None,
                   alpha: float = None,
                   beta: float = None,
                   pil_autocontrast: bool = False) -> np.array:

    """
    Preprocess images

    This function makes the preprocessing of images of a given path, for the
    given parameters.

    Parameters
    ----------
        -paths : list [str]
            List of strings of paths to the images.

        -width : int
            New width of the image.

        -height : int
            New height of the image.

        -resize : bool
            If True, resize the image based on the given width and height.

        -interpolation : module
            Interpolation method

        -normalization : bool
            If True, normalize the image, diving the pixels by 255.

        -color : module
            Color space conversion method

        -alpha : float - [1.0 ; 3.0]
            Needs to be in the interval [1.0 ; 3.0].
            Changes the contrast of images.

        -beta : float - [0 ; 100]
            Needs to be in the interval [0 ; 100].
            Changes the brightness of images.

        -pil_autocontrast : bool
            If True, applies autocontrast to the image of the PIL library,
            of ImageOps library.

    Returns
    -------
        -np.array
            Array of the image preprocessed.
    """

    img_vector = []

    for image in tqdm(paths):

        # https://docs.opencv.org/4.x/

        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if resize:
            img = cv2.resize(img, (width, height), interpolation=interpolation)

        # alpha: Contrast Control [1.0-3.0], beta: Brightness Control [0-100]
        if alpha or beta:
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if color:
            img = cv2.cvtColor(img, color)

        if pil_autocontrast:
            img = Image.fromarray(img)
            img = ImageOps.autocontrast(img)
            img = np.array(img)

        if normalization:
            img = img/255

        img_vector.append(img)

    return np.array(img_vector, dtype=np.float32)


def evaluate_model_train(history_, label2: str = 'F1Score'):

    """
    Plotting.

    This function plots the loss and other wanted metric of the model during
    training.

    Parameters
    ----------
        -history_ : fit of the model
            The history of the model during training.

        -label2 : str
            The name of the metric to plot.

    Returns
    -------
        None
    """

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history_.history['loss'])
    ax[0].plot(history_.history['val_loss'])
    ax[0].set_title('Model Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend(['train', 'validation'])

    ax[1].plot(np.mean(history_.history[label2], axis=1))
    ax[1].plot(np.mean(history_.history['val_' + label2], axis=1))
    ax[1].set_title('Model F1 Score')
    ax[1].set_ylabel('Score')
    ax[1].set_xlabel('Epochs')
    ax[1].legend(['train', 'validation'])
    plt.show()


def evaluate_model_predictions(model, X_array: np.array,
                               y_array_1: np.array,
                               y_label_1: np.array,
                               problem_type: str = None,
                               y_array_2: np.array = None,
                               y_label_2: np.array = None):

    """
    Plot and predict.

    This function makes predictions of a given model, prints the
    classification report and then plots a heatmap based on the confusion
    matrix.


    Parameters
    ----------
        -model :
            The history of the model during training.

        -X_array : np.array
            The input array of the model.

        -y_array_1 : np.array
            Array of the true labels.

        -y_label_1 : list [str]
            Name of the labels to be displayed in the plots.

        -problem_type : str
            Problem type, either "single" or "both". "both" refers to having
            binary and multiclass classification together.

        -y_array_2 : np.array
            Only used if problem_type is "both". Array of the true labels.

        -y_label_2 : list [str]
            Only used if problem_type is "both".
            Name of the labels to be displayed in the plots.

    Returns
    -------
        None

    """

    if problem_type.lower() == 'single':
        y_predicted_mul = model.predict(X_array)
        y_predicted_mul = np.argmax(y_predicted_mul, axis=1)
        y_true_mul = np.argmax(y_array_1, axis=1)
        print(classification_report(y_true_mul, y_predicted_mul,
                                    target_names=np.unique(y_label_1)))

        conf_matrix = confusion_matrix(y_true_mul, y_predicted_mul)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_label_1),
                    yticklabels=np.unique(y_label_1))
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.title('Confusion Matrix')
        plt.show()

    else:
        y_predicted_bi, y_predicted_mul = model.predict(X_array)

        # binary
        y_predicted_bi = np.argmax(y_predicted_bi, axis=1)
        y_true_bi = np.argmax(y_array_1, axis=1)
        print('Classification Report: Multiclass')
        print(classification_report(y_true_bi, y_predicted_bi,
                                    target_names=np.unique(y_label_1)))

        conf_matrix = confusion_matrix(y_true_bi, y_predicted_bi)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_label_1),
                    yticklabels=np.unique(y_label_1))
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # multiclass
        y_predicted_mul = np.argmax(y_predicted_mul, axis=1)
        y_true_mul = np.argmax(y_array_2, axis=1)
        print('-----------------------------------------------------------')
        print('Classification Report: Multiclass')
        print(classification_report(y_true_mul, y_predicted_mul,
                                    target_names=np.unique(y_label_2)))

        conf_matrix = confusion_matrix(y_true_mul, y_predicted_mul)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_label_2),
                    yticklabels=np.unique(y_label_2))
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.title('Confusion Matrix')
        plt.show()


# Note: The two functions below, to be used for gridsearching
# regularizers and optimizers are directly taken from
# https://github.com/keras-team/keras-tuner/issues/266
# The reg_wrapper is a direct copy and
# the opt_wrapper uses the same logic,
# but was adapted by us for its purpose.


def reg_wrapper(type: str, value: float):

    """
    This function is a wrapper for regularization methods.

    Parameters
    ----------
        -type : str
            Type of regularization passed as string.

        -value : float
            Value of the regularization.

    Returns
    -------
        -regularization
            Regularization method.

    """
    if type == 'l2':
        return L2(value)
    if type == 'l1':
        return L1(value)


def opt_wrapper(type: str, lr: float):

    """
    This function is a wrapper for optimization methods.

    Parameters
    ----------
        -type : str
            Optimizer passed as string.

        -value : float
            Value of the learning rate.

    Returns
    -------
        -Optimizer
            Optimizer method.


    """

    if type == 'RMSprop':
        return RMSprop(learning_rate=lr)
    if type == 'Adam':
        return Adam(learning_rate=lr)
    if type == 'Adadelta':
        return Adadelta(learning_rate=lr)
    if type == 'Adagrad':
        return Adagrad(learning_rate=lr)


# The following inception_module was initially taken
# from https://maelfabien.github.io/deeplearning/inception/#in-keras
# but was then adapted by us


def inception_module(input_tensor, n_f_1x1: int,
                     n_f_3x3_r: int, n_f_3x3: int,
                     n_f_5x5_r: int, n_f_5x5: int,
                     n_f_1x1_end: int, name: str = None):

    """
    Function that creates an inception module architecture.

    Parameters
    ----------
    -input_tensor : tensor
        Tensor to be passed as input to convulutional layers.

    -n_f_1x1 : int
        Number of filters in the convolutional layer with 1x1 filter.

    -n_f_3x3_r : int
        Number of filters in the convolutional layer with 1x1 filter to reduce
        before the 3x3 convolution.

    -n_f_3x3 : int
        Number of filters in the convolutional layer with 3x3 filter.

    -n_f_5x5_r : int
        Number of filters in the convolutional layer with 1x1 filter to reduce
        before the 5x5 convolution.

    -n_f_5x5 : int
        Number of filters in the convolutional layer with 5x5 filter.

    -n_f_1x1_end : int
        Number of filters in the final convolutional layer with 1x1 filter.

    -name : str
        Name of the inception module we want to give.

    Returns
    -------
    -concatenation : tensor
        Tensor with the output concatenated of all the layers in module.

    """

    # 1x1 Convolution
    conv_1x1 = Conv2D(n_f_1x1, (1, 1), activation='relu',
                      name=f'{name}_Conv_1x1')(input_tensor)

    # 1x1 - 3x3 Convolution
    conv_3x3_reduce = Conv2D(n_f_3x3_r, (1, 1), activation='relu',
                             name=f'{name}_Conv_3x3_r')(input_tensor)
    conv_3x3 = Conv2D(n_f_3x3, (3, 3), padding='same', activation='relu',
                      name=f'{name}_Conv_3x3')(conv_3x3_reduce)

    # 1x1 - 5x5 Convolution
    conv_5x5_reduce = Conv2D(n_f_5x5_r, (1, 1), activation='relu',
                             name=f'{name}_Conv_5x5_r')(input_tensor)
    conv_5x5 = Conv2D(n_f_5x5, (5, 5), padding='same', activation='relu',
                      name=f'{name}_Conv_5x5')(conv_5x5_reduce)

    # Max Pooling - 1x1 Convolution
    # padding is same and strides=(1,1) in the max pooling to have all
    # features maps with the same dimentions at the concatenation step
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                            name=f'{name}_Pool_3x3')(input_tensor)
    conv_1x1_end = Conv2D(n_f_1x1_end, (1, 1), padding='same',
                          activation='relu',
                          name=f'{name}_Conv_1x1_end')(max_pool)

    # Concatenate all branches
    concatenation = concatenate([conv_1x1, conv_3x3, conv_5x5, conv_1x1_end],
                                axis=-1)
    return concatenation
