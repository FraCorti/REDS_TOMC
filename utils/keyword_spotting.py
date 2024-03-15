import os

import numpy as np
import tensorflow as tf

from utils.keyword_spotting_data import prepare_words_list
from utils.keyword_spotting_models import create_dnn_model, create_cnn_model, create_ds_cnn_model


def get_model_size_info_dnn(model_size):
    """
    Given the size of the model return the number of hidden units used in each layer
    """
    if model_size == "l":
        return [436, 436, 436]
    if model_size == "m":
        return [256, 256, 256]
    if model_size == "s":
        return [144, 144, 144]


def get_model_size_info_cnn(model_size):
    """
    Given the size of the model return the number of filters used in the convolution part followed by the number
    of hidden units used in each fully connected layer
    """
    if model_size == "l":
        return [60, 76], [4864, 58, 128, 12]
    if model_size == "m":
        return [64, 48], [3072, 16, 128, 12]
    if model_size == "s":
        return [28, 30], [1920, 16, 128, 12]


def get_model_size_info_ds_cnn(model_size):
    model_info = None
    if model_size == 's':
        model_info = [5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]
    elif model_size == 'm':
        model_info = [5, 172, 10, 4, 2, 1, 172, 3, 3, 2, 2, 172, 3, 3, 1, 1, 172, 3, 3, 1, 1, 172, 3, 3, 1, 1]
    elif model_size == 'l':
        model_info = [6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3,
                      3, 1, 1]
    return model_info


def load_pre_trained_kws_model(args, model_name, model_size,
                               pretrained_model_path="{}/models/pretrained_arm_kws_models/{}/{}_{}/ckpt/"):
    model_settings = prepare_model_settings(len(prepare_words_list(args.wanted_words.split(','))),
                                            args.sample_rate, args.clip_duration_ms, args.window_size_ms,
                                            args.window_stride_ms, args.dct_coefficient_count)

    if model_name == "dnn":
        model_size_info = get_model_size_info_dnn(model_size=model_size)
        model = create_dnn_model(model_settings=model_settings, model_size_info=model_size_info)
        latest_checkpoint = tf.train.latest_checkpoint(
            pretrained_model_path.format(os.getcwd(), model_name, model_name, model_size))
        model.load_weights(latest_checkpoint).expect_partial()
        return model, model_settings, model_size_info

    if model_name == "cnn":
        model_size_info_convolution, model_size_info_dense = get_model_size_info_cnn(model_size=model_size)
        model = create_cnn_model(model_settings=model_settings, model_size_info_convolution=model_size_info_convolution,
                                 model_size_info_dense=model_size_info_dense)
        latest_checkpoint = tf.train.latest_checkpoint(
            pretrained_model_path.format(os.getcwd(), model_name, model_name, model_size))
        model.load_weights(latest_checkpoint).expect_partial()
        return model, model_settings, model_size_info_convolution, model_size_info_dense
    if model_name == "ds_cnn":
        model_size_info = get_model_size_info_ds_cnn(model_size=model_size)
        model = create_ds_cnn_model(model_settings=model_settings, model_size_info=model_size_info)
        latest_checkpoint = tf.train.latest_checkpoint(
            pretrained_model_path.format(os.getcwd(), model_name, model_name, model_size))
        model.load_weights(latest_checkpoint).expect_partial()
        return model, model_settings, model_size_info

    return


def calculate_accuracy(predicted_indices, expected_indices):
    """Calculates and returns accuracy.

    Args:
        predicted_indices: List of predicted integer indices.
        expected_indices: List of expected integer indices.

    Returns:
        Accuracy value between 0 and 1.
    """
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def compute_accuracy_train_validation(model, model_settings, train_data, val_data, message="", confusion_matrix=False):
    expected_indices = np.concatenate([y for x, y in train_data])

    predictions = model.predict(train_data)
    predicted_indices = tf.argmax(predictions, axis=1)

    val_accuracy = calculate_accuracy(predicted_indices, expected_indices)

    if confusion_matrix:
        confusion_matrix = tf.math.confusion_matrix(expected_indices, predicted_indices,
                                                    num_classes=model_settings['label_count'])
        print(confusion_matrix.numpy())
    print(f'Train accuracy = {val_accuracy * 100:.2f}%')

    expected_indices = np.concatenate([y for x, y in val_data])

    predictions = model.predict(val_data)
    predicted_indices = tf.argmax(predictions, axis=1)

    val_accuracy = calculate_accuracy(predicted_indices, expected_indices)

    if confusion_matrix:
        confusion_matrix = tf.math.confusion_matrix(expected_indices, predicted_indices,
                                                    num_classes=model_settings['label_count'])
        print(confusion_matrix.numpy())

    print(f'{message} Validation accuracy = {val_accuracy * 100:.2f}%')

def compute_accuracy_test_mobilenet(model, test_data, message="", confusion_matrix=False):
    # Evaluate on testing set.
    #expected_indices = np.concatenate([y for x, y in test_data])

    #test_input_data = np.concatenate(([x for x, y in test_data]))
    #print(test_input_data.shape)
    test_accuracy = []
    for x, y in test_data:

        predictions = model.predict(tf.keras.applications.mobilenet.preprocess_input(x), verbose=0)
        predicted_indices = tf.argmax(predictions, axis=1)

        accuracy_batch = calculate_accuracy(predicted_indices, tf.cast(y, dtype=tf.int64))

        if confusion_matrix:
            confusion_matrix = tf.math.confusion_matrix(tf.cast(y, dtype=tf.int64), predicted_indices,
                                                        num_classes=1000)
            print(confusion_matrix.numpy())

        test_accuracy.append(accuracy_batch * 100)


    print(f'{message} Test accuracy = {tf.reduce_mean(test_accuracy):.2f}%')
    return test_accuracy

def compute_accuracy_test(model, test_data, model_settings=None, message="", confusion_matrix=False):
    # Evaluate on testing set.
    expected_indices = np.concatenate([y for x, y in test_data])

    predictions = model.predict(test_data)
    predicted_indices = tf.argmax(predictions, axis=1)

    test_accuracy = calculate_accuracy(predicted_indices, expected_indices)

    test_accuracy = test_accuracy * 100
    print(f'{message} Test accuracy = {test_accuracy:.2f}%')
    return test_accuracy


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    """Calculates common settings needed for all models.

    Args:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second.
        clip_duration_ms: Length of each audio clip to be analyzed.
        window_size_ms: Duration of frequency analysis window.
        window_stride_ms: How far to move in time between frequency windows.
        dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
        Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length

    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }
