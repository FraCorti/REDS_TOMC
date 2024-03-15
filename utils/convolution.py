import time

import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
import numpy as np

from utils.batch_normalization import Reds_BatchNormalizationBase
from utils.linear import Linear, Linear_Adaptive


def get_reds_cnn_architecture(architecture_name, model_size_info_convolution, model_settings, model_size_info_dense,
                              classes=10, debug=False, use_bias=True,
                              subnetworks_number=4):
    if architecture_name == "cnn":
        return Reds_Cnn_Wake_Model(classes=classes, model_size_info_convolution=model_size_info_convolution,
                                   model_size_info_dense=model_size_info_dense, use_bias=use_bias,
                                   subnetworks_number=subnetworks_number,
                                   model_settings=model_settings, debug=debug)


def get_model_convolutional_layers_number(model):
    convolutional_layer_number = 0

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Conv2D):
            convolutional_layer_number += 1

    return convolutional_layer_number


class Reds_Cnn_Wake_Model(tf.keras.Model):

    def __init__(self, classes, model_size_info_convolution, model_size_info_dense, subnetworks_number, model_settings,
                 batch_dimensions=4,
                 use_bias=True,
                 debug=False):

        super(Reds_Cnn_Wake_Model, self).__init__()
        self.model_settings = model_settings

        self.reshape = tf.keras.layers.Reshape(
            (self.model_settings['spectrogram_length'], self.model_settings['dct_coefficient_count'], 1),
            input_shape=(self.model_settings['fingerprint_size'], 1))
        self.conv1 = Reds_2DConvolution_Standard(in_channels=1, out_channels=model_size_info_convolution[0],
                                                 kernel_size=(10, 4),
                                                 batch_dimensions=batch_dimensions,
                                                 use_bias=use_bias, strides=(1, 1),
                                                 debug=debug, padding='valid')
        self.batch_norm1 = Reds_BatchNormalizationBase(fused=False)
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = Reds_2DConvolution_Standard(in_channels=model_size_info_convolution[0],
                                                 out_channels=model_size_info_convolution[1], kernel_size=(10, 4),
                                                 use_bias=use_bias, strides=(2, 1),
                                                 batch_dimensions=batch_dimensions,
                                                 debug=debug, padding='valid')
        self.batch_norm2 = Reds_BatchNormalizationBase(fused=False)
        self.relu2 = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = Linear_Adaptive(in_features=model_size_info_dense[0], out_features=model_size_info_dense[1],
                                   debug=debug, use_bias=use_bias)
        self.fc2 = Linear(in_features=model_size_info_dense[1], out_features=model_size_info_dense[2], debug=debug,
                          use_bias=use_bias)
        self.batch_norm3 = Reds_BatchNormalizationBase(fused=False)

        self.relu3 = tf.keras.layers.ReLU()
        self.fc3 = Linear(in_features=model_size_info_dense[2], out_features=model_size_info_dense[3], debug=debug,
                          use_bias=use_bias)

        self.subnetworks_number = subnetworks_number
        self.print_hidden_feature = False
        self.debug = debug
        self.classes = classes
        self.use_bias = use_bias
        self.dense_model_trainable_weights = 0

    def get_subnetwork_parameters_percentage(self, subnetwork_index):
        """
        If the input feature are not registered perform a forward pass on the adaptive layers to compute them, return the
        number of trainable parameters used by the subnetwork
        @param subnetwork_index:
        @param input_sample:
        @return:
        """

        if self.dense_model_trainable_weights == 0:
            self.dense_model_trainable_weights = sum(
                [np.prod(tensor.shape) if not isinstance(tensor, Reds_BatchNormalizationBase) else 0 for tensor in
                 self.trainable_weights])

        subnetwork_trainable_parameters = 0
        for layer in self.layers:

            if isinstance(layer, Reds_BatchNormalizationBase) or isinstance(layer,
                                                                            tf.keras.layers.Reshape) or isinstance(
                layer, tf.keras.layers.ReLU) or isinstance(layer, tf.keras.layers.Flatten):
                continue
            elif isinstance(layer, Linear_Adaptive):
                subnetwork_trainable_parameters += layer.get_trainable_parameters_number()
            else:
                subnetwork_trainable_parameters += layer.get_trainable_parameters_number(
                    subnetwork_index=subnetwork_index)

        # iterate over model layers and count the weights of each layer specific for the subnetwork
        return subnetwork_trainable_parameters / self.dense_model_trainable_weights

    def set_subnetworks_number(self, subnetworks_number):
        self.subnetworks_number = subnetworks_number

    def finetune_batch_normalization(self):
        """
        Used to finetune the Batch Normalization layers while freezing all
        the other layers
        """
        for layer in self.layers:
            if isinstance(layer, Reds_BatchNormalizationBase):
                layer.trainable = True
            else:
                layer.trainable = False

    def set_subnetwork_indexes(self, subnetworks_filters_indexes):

        for subnetwork_filters_indexes in subnetworks_filters_indexes:
            self.conv1.add_splitting_filters_indexes(subnetwork_filters_indexes[0][0] + 1)
            self.conv2.add_splitting_filters_indexes(subnetwork_filters_indexes[0][1] + 1)

    def set_print_hidden_feature(self, print=True):
        self.print_hidden_feature = print
        self.debug = print

    def set_debug(self, debug=True):
        self.debug = debug

    def get_model_name(self):
        return "Wake_Word_CNN"

    def compute_lookup_table(self, train_data):
        """
        Compute the lookup table for the model given the training data set. A batch of the data is passed as input
        inside the model and the linear operation associated to the layer is performed for average_run times. Then the
        MACs for each layers's filters are computed and the memory size of each filter is computed too.
        @return: List[List] containing the filters forward times for each layer
        List[List] containing the filters macs for each layer
        """

        layers_filters_macs = []
        layers_filters_byte = []

        inputs = tf.ones((1, 49, 10, 1), dtype=tf.float32)

        inputs, macs, filters_byte_memory = self.conv1.compute_layer_lookup_table(inputs=inputs)
        layers_filters_macs.append(macs)
        layers_filters_byte.append(filters_byte_memory)

        inputs = self.batch_norm1(inputs)
        inputs = self.relu1(inputs)

        inputs, macs, filters_byte_memory = self.conv2.compute_layer_lookup_table(
            inputs=inputs)
        layers_filters_macs.append(macs)
        layers_filters_byte.append(filters_byte_memory)

        return layers_filters_macs, layers_filters_byte

    def build(self, input_shape):
        """
        Initialize model's layers
        """

        init_input = tf.ones(
            input_shape,
            dtype=tf.dtypes.float32,
        )

        init_input = self.reshape(init_input)
        init_input = [init_input for _ in range(1)]
        init_input = self.conv1(init_input)
        _ = self.batch_norm1.build(init_input[0].shape)

        init_input = self.conv2(init_input)
        _ = self.batch_norm2.build(init_input[0].shape)
        init_input = [tf.keras.layers.Flatten()(input) for input in init_input]

        init_input = self.fc1(init_input)
        init_input = self.fc2(init_input)
        _ = self.batch_norm3.build(init_input[0].shape)
        _ = self.fc3(init_input)

        self.built = True

    def call(self, inputs, training=None, mask=None):

        inputs = self.reshape(inputs)

        # copy the input once for each subnetwork
        inputs = [inputs for _ in range(self.subnetworks_number)]

        inputs = self.conv1(inputs)
        inputs = self.batch_norm1(inputs)
        inputs = [self.relu1(input) for input in inputs]
        inputs = self.conv2(inputs)
        inputs = self.batch_norm2(inputs)
        inputs = [self.relu2(input) for input in inputs]
        inputs = [tf.keras.layers.Flatten()(input) for input in inputs]
        inputs = self.fc1(inputs)
        inputs = self.fc2(inputs)
        inputs = self.batch_norm3(inputs)
        inputs = [self.relu3(input) for input in inputs]
        inputs = self.fc3(inputs)

        return inputs

    def get_model(self):

        model = tf.keras.Sequential()

        reshape_layer = tf.keras.layers.Reshape(
            (-1, self.model_settings['spectrogram_length'], self.model_settings['dct_coefficient_count'], 1),
            input_shape=(self.model_settings['fingerprint_size'], 1))

        model.add(reshape_layer)

        conv_first_layer = tf.keras.layers.Conv2D(filters=self.conv1.filters, use_bias=self.use_bias,
                                                  kernel_size=self.conv1.kernel_size, padding=self.conv1.padding,
                                                  strides=self.conv1.strides, input_shape=(1, 1, 40, 10, 1))

        conv_first_layer(tf.ones((1, 1, 40, 10, self.conv1.input_dimension)))
        conv_first_layer.set_weights(weights=self.conv1.get_parameters())
        model.add(conv_first_layer)

        batch_norm1 = tf.keras.layers.BatchNormalization()
        batch_norm1(tf.ones((1, 1, 40, 10, self.conv1.filters)))
        batch_norm1.set_weights(weights=self.batch_norm1.get_weights())
        model.add(batch_norm1)

        model.add(tf.keras.layers.ReLU())

        conv_second_layer = tf.keras.layers.Conv2D(filters=self.conv2.filters, use_bias=self.use_bias,
                                                   kernel_size=self.conv2.kernel_size, strides=self.conv2.strides,
                                                   padding=self.conv2.padding)

        conv_second_layer(tf.ones((1, 1, 40, 7, self.conv2.input_dimension)))
        conv_second_layer.set_weights(weights=self.conv2.get_parameters())
        model.add(conv_second_layer)

        batch_norm2 = tf.keras.layers.BatchNormalization()
        batch_norm2(tf.ones((1, 1, 40, 7, self.conv2.filters)))
        batch_norm2.set_weights(weights=self.batch_norm2.get_weights())
        model.add(batch_norm2)
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Flatten())

        dense_first_layer = tf.keras.layers.Dense(units=self.fc1.out_features, use_bias=self.use_bias,
                                                  input_shape=(1, self.fc1.in_feature))

        dense_first_layer(tf.ones((1, self.fc1.in_feature)))
        dense_first_layer.set_weights(weights=self.fc1.get_parameters())
        model.add(dense_first_layer)

        dense_second_layer = tf.keras.layers.Dense(units=self.fc2.out_features, use_bias=self.use_bias,
                                                   input_shape=(1, self.fc2.in_features))

        dense_second_layer(tf.ones((1, self.fc2.in_features)))
        dense_second_layer.set_weights(weights=self.fc2.get_parameters())
        model.add(dense_second_layer)

        batch_norm3 = tf.keras.layers.BatchNormalization()
        batch_norm3(tf.ones((1, self.fc2.out_features)))
        batch_norm3.set_weights(weights=self.batch_norm3.get_weights())
        model.add(batch_norm3)

        model.add(tf.keras.layers.ReLU())

        dense_third_layer = tf.keras.layers.Dense(units=self.fc3.out_features, use_bias=self.use_bias,
                                                  input_shape=(1, self.fc3.in_features))
        dense_third_layer(tf.ones((1, self.fc3.in_features)))
        dense_third_layer.set_weights(weights=self.fc3.get_parameters())
        model.add(dense_third_layer)

        return model


class Reds_2DConvolution_Standard(tf.keras.layers.Layer):

    def __init__(self, in_channels=1, out_channels=32, kernel_size=(3, 3),
                 kernel_initializer='glorot_uniform',
                 bias_initializer='glorot_uniform',
                 activation="relu",
                 groups=1,
                 strides=(1, 1),
                 rank=2,
                 use_bias=False,
                 batch_dimensions=4,
                 data_format="channels_first",  # channels_last
                 padding='valid',
                 trainable=True,
                 dilation_rate=1, debug=False):

        super(Reds_2DConvolution_Standard, self).__init__()
        self.kernel_size = kernel_size
        self.filters = out_channels
        self.input_dimension = in_channels
        self.groups = groups or 1
        self.rank = rank
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.padding = conv_utils.normalize_padding(padding)
        self.strides = conv_utils.normalize_tuple(
            strides, rank, "strides"
        )
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )

        self.data_format = data_format
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2
        )

        kernel_shape = self.kernel_size + (
            self.input_dimension // self.groups,
            self.filters,
        )

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      trainable=trainable,
                                      dtype=self.dtype)
        if use_bias:
            self.b = self.add_weight(shape=(self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     trainable=trainable,
                                     dtype=self.dtype)
        self.debug = debug
        self.use_bias = use_bias
        self.batch_dimensions = batch_dimensions

        self.filters_splittings = tf.Variable(initial_value=tf.constant([self.kernel.shape[3]], dtype=tf.int32),
                                              trainable=False)
        self.subnetwork_kernels_number = tf.Variable(initial_value=tf.constant([self.kernel.shape[2]], dtype=tf.int32),
                                                     trainable=False)
        self.registered_subnetworks_kernels_number = False
        self.built = True

    def registered_subnetworks_kernels_number(self, status=True):
        self.registered_subnetworks_kernels_number = status

    def get_trainable_parameters_number(self, subnetwork_index):
        """
        Get the number of trainable filters and bias parameters of the subnetwork
        @param subnetwork_index: index of the subnetowork to obtain the number of trainable parameters
        """

        # pointwise convolution layer
        if self.kernel.shape[0] == 1:
            return np.prod(self.kernel[:, :, :, 0:int(self.filters_splittings[subnetwork_index])].shape) + np.prod(
                self.b[:int(self.filters_splittings[subnetwork_index])].shape) if self.use_bias else 0
        else:
            # first convolution layer, to change if standard cnn are used
            return np.prod(self.kernel[:, :, :,
                           0:int(self.filters_splittings[subnetwork_index])].shape) + np.prod(
                self.b[:int(self.filters_splittings[subnetwork_index])].shape) if self.use_bias else 0

    def add_splitting_filters_indexes(self, filters_indexes):
        self.filters_splittings = tf.concat([self.filters_splittings, tf.constant([filters_indexes], dtype=tf.int32)],
                                            axis=0)

    def compute_layer_lookup_table(self, inputs):
        """
        Compute the lookup table for each layer, retrieve each filter of the layer and compute the forward time by
        applying the convolution operator on the input and average the forward time over the number of runs.
        @param inputs:
        @param average_run: how many runs to average the forward time
        @return: the lookup table for the layer and the transformed input
        """

        layer_filters = self.kernel.shape[3]

        activation_map = tf.nn.convolution(
            input=inputs,
            filters=self.kernel[:, :, :, :],
            strides=list(self.strides),
            padding=self.padding.upper()
        )

        if self.use_bias:
            activation_map = tf.nn.bias_add(activation_map, self.b[:])

        kernel_height, kernel_width, input_channels, output_channels = self.kernel.shape
        samples_number, output_height, output_width, _ = activation_map.shape
        filter_macs = (
                              samples_number * kernel_height * kernel_width * input_channels * output_channels * output_height * output_width) / layer_filters

        macs = np.full(layer_filters, filter_macs, dtype=float)

        filters_parameters_number = tf.size(self.kernel).numpy() / self.kernel.shape[3]
        element_size = self.kernel.dtype.size
        filters_byte_memory = np.full(layer_filters, filters_parameters_number * element_size, dtype=float)

        return activation_map, macs, filters_byte_memory

    def call(self, inputs, **kwargs):

        for subnet_number in range(len(inputs)):

            inputs[subnet_number] = tf.nn.convolution(
                input=inputs[subnet_number],
                filters=self.kernel[:, :, 0:inputs[subnet_number].shape[self.batch_dimensions - 1],
                        0:int(self.filters_splittings[subnet_number])],
                strides=list(self.strides),
                padding=self.padding.upper()
            )

            if self.use_bias:
                inputs[subnet_number] = tf.nn.bias_add(inputs[subnet_number],
                                                       self.b[:int(self.filters_splittings[subnet_number])])

        if self.registered_subnetworks_kernels_number is False and len(inputs) > 1:
            self.registered_subnetworks_kernels_number = True

        return inputs

    def get_parameters(self):
        return self.get_weights()
