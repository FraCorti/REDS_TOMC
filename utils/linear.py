import time

import numpy as np
import tensorflow as tf

from utils.logs import log_print, print_intermediate_activations


class Linear_Adaptive(tf.keras.layers.Layer):
    def __init__(self, in_features=32, out_features=32, use_bias=False, debug=False):
        super(Linear_Adaptive, self).__init__()
        self.w = self.add_weight(
            shape=(in_features, out_features), initializer="glorot_uniform", trainable=True
        )
        if use_bias:
            self.b = self.add_weight(shape=(out_features,), initializer="glorot_uniform", trainable=True)
        self.debug = debug
        self.in_feature = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.subnetworks_input_feature = tf.Variable(initial_value=tf.constant([], dtype=tf.int32),
                                                     trainable=False)
        self.registered_subnetworks_input_feature_number = False

    def call(self, inputs, **kwargs):

        for subnet_number in range(len(inputs)):

            if self.registered_subnetworks_input_feature_number is False and len(inputs) > 1:
                self.subnetworks_input_feature = tf.concat(
                    [self.subnetworks_input_feature,
                     tf.constant([inputs[subnet_number].shape[1]], dtype=tf.int32)],
                    axis=0)

            if self.debug:
                log_print(
                    f"Dense weights cuttings: {self.w[0:inputs[subnet_number].shape[1], :].shape}")

            inputs[subnet_number] = tf.matmul(inputs[subnet_number],
                                              self.w[0:inputs[subnet_number].shape[1], :])

            if self.use_bias:
                inputs[subnet_number] = tf.nn.bias_add(inputs[subnet_number], self.b)

        if self.registered_subnetworks_input_feature_number is False and len(inputs) > 1:
            self.registered_subnetworks_input_feature_number = True

        return inputs

    def get_trainable_parameters_number(self, subnetwork_index=0):
        if self.registered_subnetworks_input_feature_number == True:
            return np.prod(self.w[:,0:self.subnetworks_input_feature[subnetwork_index]].shape) + np.prod(
                self.b.shape)
        return np.prod(self.w) + np.prod(self.b.shape)

    def get_parameters(self):
        return self.get_weights()


def get_reds_dnn_architecture(model_settings, classes=12, hidden_units=144, subnetworks_number=4,
                              use_bias=True):
    return Reds_Dnn_Wake_Model(classes=classes, hidden_units=hidden_units, subnetworks_number=subnetworks_number,
                               use_bias=use_bias, model_settings=model_settings)


class Reds_Dnn_Wake_Model(tf.keras.Model):

    def __init__(self, subnetworks_number, model_settings, debug=False, use_bias=False,
                 hidden_units=30, classes=12):
        super(Reds_Dnn_Wake_Model, self).__init__()

        self.model_settings = model_settings
        self.fc1 = Reds_Linear(in_features=self.model_settings['fingerprint_size'], out_features=hidden_units,
                               initializer="glorot_uniform",
                               use_bias=use_bias)
        self.relu1 = tf.keras.layers.ReLU()
        self.fc2 = Reds_Linear(in_features=hidden_units, out_features=hidden_units, initializer="glorot_uniform",
                               use_bias=use_bias)
        self.relu2 = tf.keras.layers.ReLU()
        self.fc3 = Reds_Linear(in_features=hidden_units, out_features=hidden_units, initializer="glorot_uniform",
                               use_bias=use_bias)
        self.relu3 = tf.keras.layers.ReLU()
        self.fc4 = Linear_Adaptive(in_features=hidden_units, out_features=classes,
                                   use_bias=use_bias)

        self.subnetworks_number = subnetworks_number
        self.classes = classes
        self.use_bias = use_bias
        self.debug = debug
        self.print_hidden_feature = False
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
                [np.prod(tensor.shape) for tensor in
                 self.trainable_weights])

        subnetwork_trainable_parameters = 0
        for layer in self.layers:

            if isinstance(
                    layer, tf.keras.layers.ReLU) or isinstance(layer, tf.keras.layers.Flatten):
                continue
            elif isinstance(layer, Linear_Adaptive):
                subnetwork_trainable_parameters += layer.get_trainable_parameters_number()
            else:
                subnetwork_trainable_parameters += layer.get_trainable_parameters_number(
                    subnetwork_index=subnetwork_index)

        # iterate over model layers and count the weights of each layer specific for the subnetwork
        return subnetwork_trainable_parameters / self.dense_model_trainable_weights

    def set_subnetwork_indexes(self, subnetworks_neurons_indexes):

        for subnetwork_filters_indexes in subnetworks_neurons_indexes:
            self.fc1.add_splitting_neurons_indexes(subnetwork_filters_indexes[0][0] + 1)
            self.fc2.add_splitting_neurons_indexes(subnetwork_filters_indexes[0][1] + 1)
            self.fc3.add_splitting_neurons_indexes(subnetwork_filters_indexes[0][2] + 1)

    def set_print_hidden_feature(self, print=True):
        self.print_hidden_feature = print
        self.debug = print

    def set_debug(self, debug=True):
        self.debug = debug

    def get_model_name(self):
        return "reds_dnn_wake_model_{}hidden_four_linear{}".format(self.fc1.in_features,
                                                                   "_bias" if self.use_bias else "_no_bias")

    def set_subnetworks_number(self, subnetworks_number):
        self.subnetworks_number = subnetworks_number

    def compute_lookup_table(self, train_data):
        """
        Compute the number of MACs and the number of bytes of each layer contained inside the model
        """
        inputs = tf.ones((1, 250), dtype=tf.float32)

        layers_filters_macs = []
        layers_filters_byte = []

        for layer in self.layers:

            if isinstance(layer, Reds_Linear):
                inputs, macs, filters_byte_memory = layer.compute_layer_lookup_table(
                    inputs=inputs)

                layers_filters_macs.append(macs)
                layers_filters_byte.append(filters_byte_memory)

        return layers_filters_macs, layers_filters_byte

    def call(self, inputs, training=None, mask=None):
        inputs = [inputs for _ in range(self.subnetworks_number)]
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="Inputs shape") if self.debug else None

        inputs = self.fc1(inputs)
        inputs = [self.relu1(input) for input in inputs]
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="FC1 Layer") if self.debug else None

        inputs = self.fc2(inputs)
        inputs = [self.relu2(input) for input in inputs]
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="FC2 Layer") if self.debug else None

        inputs = self.fc3(inputs)
        inputs = [self.relu3(input) for input in inputs]
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="FC3 Layer") if self.debug else None

        inputs = self.fc4(inputs)
        print_intermediate_activations(inputs=inputs, print_hidden_feature=self.print_hidden_feature,
                                       message="LAST output vector") if self.debug else None

        return inputs

    def get_model(self):
        """
        Iterate over the REDS pretrained layers of the model and for each extract the weight and bias, instanciate
        a Linear layer with that followed by a ReLU function if the layer is not the last one.
        @return:
        """
        model = tf.keras.Sequential()

        first_layer = tf.keras.layers.Dense(units=self.fc1.out_features, use_bias=self.use_bias, activation='relu',
                                            input_shape=(self.fc1.in_features,))
        first_layer(tf.ones((1, self.fc1.in_features)))
        first_layer.set_weights(weights=self.fc1.get_parameters())
        model.add(first_layer)

        second_layer = tf.keras.layers.Dense(units=self.fc2.out_features, use_bias=self.use_bias, activation='relu')
        second_layer(tf.ones((1, self.fc2.in_features)))
        second_layer.set_weights(weights=self.fc2.get_parameters())
        model.add(second_layer)

        third_layer = tf.keras.layers.Dense(units=self.fc3.out_features, use_bias=self.use_bias, activation='relu')
        third_layer(tf.ones((1, self.fc3.in_features)))
        third_layer.set_weights(weights=self.fc3.get_parameters())
        model.add(third_layer)

        fourth_layer = tf.keras.layers.Dense(units=self.fc4.out_features, use_bias=self.use_bias, activation='relu')
        fourth_layer(tf.ones((1, self.fc4.in_features)))
        fourth_layer.set_weights(weights=self.fc4.get_parameters())
        model.add(fourth_layer)

        return model


class Reds_Linear(tf.keras.layers.Layer):

    def __init__(self, in_features=64, out_features=10, use_bias=True,
                 initializer="glorot_uniform",
                 debug=False):
        super(Reds_Linear, self).__init__()
        self.w = self.add_weight(
            shape=(in_features, out_features), initializer=initializer, trainable=True
        )

        if use_bias:
            self.b = self.add_weight(shape=(out_features,), initializer=initializer, trainable=True)

        self.out_features = out_features
        self.in_features = in_features
        self.debug = debug
        self.use_bias = use_bias
        self.neurons_splittings = tf.Variable(initial_value=tf.constant([self.w.shape[1]], dtype=tf.int32),
                                              trainable=False)

    def add_splitting_neurons_indexes(self, neurons_indexes):
        self.neurons_splittings = tf.concat([self.neurons_splittings, tf.constant([neurons_indexes], dtype=tf.int32)],
                                            axis=0)

    def get_trainable_parameters_number(self, subnetwork_index=0):
        return np.prod(self.w[:, 0:int(self.neurons_splittings[subnetwork_index])].shape) + np.prod(
            self.b[:int(self.neurons_splittings[subnetwork_index])].shape)

    def call(self, inputs, **kwargs):
        """
        Forward the inputs into each subnetwork by applying the cuttings to them, the inputs are stored in the
        same order of the subnetworks. They input is passed into a ReLU non-linear function if specified in the constructor.

        @param inputs:
        @param kwargs:
        @return:
        """

        for subnet_number in range(len(inputs)):

            inputs[subnet_number] = tf.matmul(inputs[subnet_number],
                                              self.w[0:inputs[subnet_number].shape[1],
                                              0:self.neurons_splittings[subnet_number]])
            if self.use_bias:
                inputs[subnet_number] = tf.add(inputs[subnet_number], self.b[0:self.neurons_splittings[subnet_number]])

        return inputs

    def get_parameters(self):
        return self.get_weights()

    def compute_layer_lookup_table(self, inputs):
        """
        Compute the lookup table for each layer, retrieve each filter of the layer and compute the forward time by
        applying the convolution operator on the input and average the forward time over the number of runs.
        @param inputs:
        @param average_run: how many runs to average the forward time
        @return: the lookup table for the layer and the transformed input
        """

        layer_neurons = self.w.shape[1]
        activation_map = tf.matmul(a=inputs, b=self.w[:, :])

        neuron_macs = (self.w.shape[0] * self.w.shape[1]) / layer_neurons

        macs = np.full(layer_neurons, neuron_macs, dtype=float)

        neurons_parameters_number = tf.size(self.w).numpy() / layer_neurons
        element_size = self.w.dtype.size
        filters_byte_memory = np.full(layer_neurons, neurons_parameters_number * element_size, dtype=float)

        return activation_map, macs, filters_byte_memory


class Linear(tf.keras.layers.Layer):

    def __init__(self, in_features=64, out_features=10, debug=False, use_bias=True, initializer="glorot_uniform"):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(in_features, out_features), initializer=initializer, trainable=True
        )

        if use_bias:
            self.b = self.add_weight(shape=(out_features,), initializer=initializer, trainable=True)
        self.out_features = out_features
        self.in_features = in_features
        self.use_bias = use_bias
        self.debug = debug

    def call(self, inputs, **kwargs):

        for subnet_number in range(len(inputs)):

            inputs[subnet_number] = tf.matmul(inputs[subnet_number], self.w)

            if self.use_bias:
                inputs[subnet_number] = tf.nn.bias_add(inputs[subnet_number], self.b)

        return inputs

    def get_trainable_parameters_number(self, subnetwork_index=0):
        return self.w.shape[0] * self.w.shape[1] + self.b.shape[0] if self.use_bias else 0

    def get_parameters(self):
        return self.get_weights()
