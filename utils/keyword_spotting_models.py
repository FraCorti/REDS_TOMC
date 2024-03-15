import math

import tensorflow as tf


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


def create_single_fc_model(model_settings):
    """Builds a model with a single fully-connected layer.

    For details see https://arxiv.org/abs/1711.07128.

    Args:
        model_settings: Dict of different settings for model training.

    Returns:
        tf.keras Model of the 'SINGLE_FC' architecture.
    """
    inputs = tf.keras.Input(shape=(model_settings['fingerprint_size'],), name='input')
    # Fully connected layer
    output = tf.keras.layers.Dense(units=model_settings['label_count'], activation='softmax')(inputs)

    return tf.keras.Model(inputs, output)


def create_dnn_model_gpu(model_settings, model_size_info, classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=model_size_info[0], activation="relu", use_bias=True,
                              input_shape=(model_settings['fingerprint_size'],)),
        tf.keras.layers.Dense(units=model_size_info[1], activation="relu", use_bias=True),
        tf.keras.layers.Dense(units=model_size_info[2], activation="relu", use_bias=True),
        tf.keras.layers.Dense(units=classes, use_bias=True),
    ])
    return model


def create_dnn_model_gpu_vision(model_size_info, classes=10, input_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(units=model_size_info[0], activation="relu", use_bias=True),
        tf.keras.layers.Dense(units=model_size_info[1], activation="relu", use_bias=True),
        tf.keras.layers.Dense(units=model_size_info[2], activation="relu", use_bias=True),
        tf.keras.layers.Dense(units=classes, use_bias=True),
    ])
    return model


def create_dnn_model(model_settings, model_size_info):
    """Builds a model with multiple hidden fully-connected layers.

    For details see https://arxiv.org/abs/1711.07128.

    Args:
        model_settings: Dict of different settings for model training.
        model_size_info: Length of the array defines the number of hidden-layers and
            each element in the array represent the number of neurons in that layer.

    Returns:
        tf.keras Model of the 'DNN' architecture.
    """

    model = tf.keras.Sequential(
        tf.keras.layers.Dense(units=model_size_info[0], activation="relu",
                              input_shape=(250,))
    )
    for i in range(1, len(model_size_info)):
        model.add(tf.keras.layers.Dense(units=model_size_info[i], activation='relu'))

    model.add(tf.keras.layers.Dense(units=model_settings['label_count']))
    return model


def create_cnn_model_gpu(model_size_info_convolution, model_size_info_dense):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=model_size_info_convolution[0],
                               kernel_size=(10, 4),
                               strides=(1, 1),
                               padding='valid', input_shape=(49, 10, 1)),
        tf.keras.layers.BatchNormalization(fused=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=model_size_info_convolution[1],
                               kernel_size=(10, 4),
                               strides=(2, 1),
                               padding='valid'),
        tf.keras.layers.BatchNormalization(fused=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=model_size_info_dense[1]),
        tf.keras.layers.Dense(units=model_size_info_dense[2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(units=model_size_info_dense[3])]
    )
    return model


def create_cnn_model(model_settings, model_size_info_convolution, model_size_info_dense):
    """Builds a model with 2 convolution layers followed by a linear layer and a hidden fully-connected layer.

    For details see https://arxiv.org/abs/1711.07128.

    Args:
        model_settings: Dict of different settings for model training.
        model_size_info: Defines the first and second convolution parameters in
            {number of conv features, conv filter height, width, stride in y,x dir.},
            followed by linear layer size and fully-connected layer size.

    Returns:
        tf.keras Model of the 'CNN' architecture.
    """

    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((-1, input_time_size, input_frequency_size, 1),
                                input_shape=(model_settings['fingerprint_size'], 1)),
        tf.keras.layers.Conv2D(filters=model_size_info_convolution[0],
                               kernel_size=(10, 4),
                               strides=(1, 1),
                               padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=model_size_info_convolution[1],
                               kernel_size=(10, 4),
                               strides=(2, 1),
                               padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=model_size_info_dense[1]),
        tf.keras.layers.Dense(units=model_size_info_dense[2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(units=model_size_info_dense[3])]
    )
    return model


def create_ds_cnn_model_parameters(model_settings, model_size_info, input_shape=(49, 10, 1)):
    """Builds a model with convolutional & depthwise separable convolutional layers.

    For more details see https://arxiv.org/abs/1711.07128.

    Args:
        model_settings: Dict of different settings for model training.
        model_size_info: Defines number of layers, followed by the DS-Conv layer
            parameters in the order {number of conv features, conv filter height,
            width and stride in y,x dir.} for each of the layers.

    Returns:
        tf.keras Model of the 'DS-CNN' architecture.
    """

    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info.
    num_layers = model_size_info[0]
    conv_feat = [None] * num_layers
    conv_kt = [None] * num_layers
    conv_kf = [None] * num_layers
    conv_st = [None] * num_layers
    conv_sf = [None] * num_layers

    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    model = tf.keras.Sequential([])

    for layer_no in range(0, num_layers):
        if layer_no == 0:
            # First convolution.
            model.add(tf.keras.layers.Conv2D(filters=conv_feat[0],
                                             kernel_size=(conv_kt[0], conv_kf[0]),
                                             strides=(conv_st[0], conv_sf[0]),
                                             padding='same', input_shape=input_shape))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
        else:
            # Depthwise convolution.
            model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                                      strides=(conv_sf[layer_no], conv_st[layer_no]),
                                                      padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

            # Pointwise convolution.
            model.add(tf.keras.layers.Conv2D(filters=conv_feat[layer_no], kernel_size=(1, 1)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

        t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
        f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

    model.add(tf.keras.layers.AveragePooling2D(pool_size=(t_dim, f_dim), strides=1))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=label_count))
    return model


def create_ds_cnn_model_gpu(model_settings, model_size_info, input_shape=(49, 10, 1), labels=None, pooling=True,
                            pointwise_padding="valid", pool_size=(12, 5)):
    """Builds a model with convolutional & depthwise separable convolutional layers.

        For more details see https://arxiv.org/abs/1711.07128.

        Args:
            model_settings: Dict of different settings for model training.
            model_size_info: Defines number of layers, followed by the DS-Conv layer
                parameters in the order {number of conv features, conv filter height,
                width and stride in y,x dir.} for each of the layers.

        Returns:
            tf.keras Model of the 'DS-CNN' architecture.
        """

    labels = model_settings['label_count'] if labels is None else labels
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info.
    num_layers = model_size_info[0]
    conv_feat = [None] * num_layers
    conv_kt = [None] * num_layers
    conv_kf = [None] * num_layers
    conv_st = [None] * num_layers
    conv_sf = [None] * num_layers

    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    model = tf.keras.Sequential([])

    for layer_no in range(0, num_layers):
        if layer_no == 0:
            # First convolution.
            model.add(tf.keras.layers.Conv2D(filters=conv_feat[0],
                                             kernel_size=(3, 3),
                                             strides=(conv_st[0], conv_sf[0]),
                                             padding='same', input_shape=input_shape))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())
        else:
            # Depthwise convolution.
            model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                                      strides=(conv_sf[layer_no], conv_st[layer_no]),
                                                      padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())

            # Pointwise convolution.
            model.add(
                tf.keras.layers.Conv2D(filters=conv_feat[layer_no], kernel_size=(1, 1), padding=pointwise_padding))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())

        t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
        f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

    if pooling:
        model.add(tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=1))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=labels))

    return model


def create_ds_cnn_model(model_settings, model_size_info):
    """Builds a model with convolutional & depthwise separable convolutional layers.

    For more details see https://arxiv.org/abs/1711.07128.

    Args:
        model_settings: Dict of different settings for model training.
        model_size_info: Defines number of layers, followed by the DS-Conv layer
            parameters in the order {number of conv features, conv filter height,
            width and stride in y,x dir.} for each of the layers.

    Returns:
        tf.keras Model of the 'DS-CNN' architecture.
    """

    label_count = model_settings['label_count']
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    t_dim = input_time_size
    f_dim = input_frequency_size

    # Extract model dimensions from model_size_info.
    num_layers = model_size_info[0]
    conv_feat = [None] * num_layers
    conv_kt = [None] * num_layers
    conv_kf = [None] * num_layers
    conv_st = [None] * num_layers
    conv_sf = [None] * num_layers

    i = 1
    for layer_no in range(0, num_layers):
        conv_feat[layer_no] = model_size_info[i]
        i += 1
        conv_kt[layer_no] = model_size_info[i]
        i += 1
        conv_kf[layer_no] = model_size_info[i]
        i += 1
        conv_st[layer_no] = model_size_info[i]
        i += 1
        conv_sf[layer_no] = model_size_info[i]
        i += 1

    model = tf.keras.Sequential([])
    model.add(tf.keras.layers.Reshape((input_time_size, input_frequency_size, 1),
                                      input_shape=(model_settings['fingerprint_size'], 1)))

    for layer_no in range(0, num_layers):
        if layer_no == 0:
            # First convolution.
            model.add(tf.keras.layers.Conv2D(filters=conv_feat[0],
                                             kernel_size=(conv_kt[0], conv_kf[0]),
                                             strides=(conv_st[0], conv_sf[0]),
                                             padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
        else:
            # Depthwise convolution.
            model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                                                      strides=(conv_sf[layer_no], conv_st[layer_no]),
                                                      padding='same'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

            # Pointwise convolution.
            model.add(tf.keras.layers.Conv2D(filters=conv_feat[layer_no], kernel_size=(1, 1)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())

        t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
        f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

    model.add(tf.keras.layers.AveragePooling2D(pool_size=(t_dim, f_dim), strides=1))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=label_count))
    return model
