import tensorflow as tf

from utils.convolution import get_model_convolutional_layers_number, Reds_2DConvolution_Standard
from utils.ds_convolution import get_pointwise_convolutions_layers_weights


def get_feature_extraction_layers_gradient_indexes(gradients):
    """
    Iterate over the computed gradients and store the position of the convolution weight and
    bias gradients
    @param gradients:
    @return:
    """
    convolution_gradients_indexes = []

    for gradient_layer_index in range(len(gradients)):

        if gradients[gradient_layer_index].shape.ndims >= 4 and gradients[gradient_layer_index].shape[1] != 1:
            convolution_gradients_indexes.append(gradient_layer_index)

    return convolution_gradients_indexes


def get_fully_connected_layers_gradient_indexes(gradients):
    """
    Iterate over the computed gradients and store the position of the fully connected weight and
    bias gradients
    @param gradients:
    @return:
    """
    fully_connected_gradients_indexes = []

    for gradient_layer_index in range(len(gradients)):

        if gradients[gradient_layer_index].shape.ndims == 2 and gradient_layer_index < len(gradients) - 2:
            fully_connected_gradients_indexes.append(gradient_layer_index)

    return fully_connected_gradients_indexes


def get_pointwise_layers_gradients_indexes(gradients):
    """
    Iterate over the computed gradients and store the position of the convolution weight and
    bias gradients
    @param gradients:
    @return:
    """
    convolution_gradients_indexes = []

    for gradient_layer_index in range(len(gradients)):

        if gradients[gradient_layer_index].shape.ndims >= 4 and gradients[gradient_layer_index].shape[1] == 1 and \
                gradients[gradient_layer_index].shape[3] != 1000:
            convolution_gradients_indexes.append(gradient_layer_index)

    return convolution_gradients_indexes


def compute_accumulated_gradients_pointwise_layers_debug(model, train_data, loss_fn, args, debug=True):
    """
    Compute the gradients accumulated for the pointwise layers
    """
    gradients_accumulation = []
    first_gradients = True
    mini_batches_number = args.minibatch_number
    convolution_parameters_gradients_indexes = None

    # compute gradients and accumulate it
    with tf.GradientTape() as tape:
        predictions = model(
            tf.keras.applications.mobilenet.preprocess_input(tf.convert_to_tensor(tf.random.normal([64, 224, 224, 3]))),
            training=True)

        loss = loss_fn(tf.random.uniform([64], minval=1, maxval=1001, dtype=tf.int32), predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    if convolution_parameters_gradients_indexes == None:
        convolution_parameters_gradients_indexes = get_pointwise_layers_gradients_indexes(gradients=gradients)

    convolution_gradients = [gradients[index] for index in convolution_parameters_gradients_indexes]

    if first_gradients:
        [gradients_accumulation.append(gradient) for gradient in convolution_gradients]
        first_gradients = False
    else:
        for gradient_index in range(len(gradients_accumulation)):
            gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                 convolution_gradients[gradient_index])

    return gradients_accumulation


def compute_accumulated_gradients_ds_cnn_layers(model, train_data, loss_fn, args):
    """
        Compute the gradients accumulated for the pointwise layers
        """
    gradients_accumulation = []
    first_gradients = True
    mini_batches_number = args.minibatch_number
    convolution_parameters_gradients_indexes = None

    for images, labels in train_data:

        if mini_batches_number <= 0:
            break

        # compute gradients and accumulate it
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        if convolution_parameters_gradients_indexes == None:
            convolution_parameters_gradients_indexes = get_pointwise_layers_gradients_indexes(gradients=gradients)

        convolution_gradients = [gradients[index] for index in convolution_parameters_gradients_indexes]

        if first_gradients:
            [gradients_accumulation.append(gradient) for gradient in convolution_gradients]
            first_gradients = False
        else:
            for gradient_index in range(len(gradients_accumulation)):
                gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                     convolution_gradients[gradient_index])
        mini_batches_number -= 1

    return gradients_accumulation


def compute_accumulated_gradients_pointwise_layers(model, train_data, loss_fn, args):
    """
    Compute the gradients accumulated for the pointwise layers
    """
    gradients_accumulation = []
    first_gradients = True
    mini_batches_number = args.minibatch_number
    convolution_parameters_gradients_indexes = None

    for images, labels in train_data:

        if mini_batches_number <= 0:
            break

        # compute gradients and accumulate it
        with tf.GradientTape() as tape:

            predictions = model(images, training=True)  # tf.keras.applications.mobilenet.preprocess_input(
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        if convolution_parameters_gradients_indexes == None:
            convolution_parameters_gradients_indexes = get_pointwise_layers_gradients_indexes(gradients=gradients)

        convolution_gradients = [gradients[index] for index in convolution_parameters_gradients_indexes]

        if first_gradients:
            [gradients_accumulation.append(gradient) for gradient in convolution_gradients]
            first_gradients = False
        else:
            for gradient_index in range(len(gradients_accumulation)):
                gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                     convolution_gradients[gradient_index])
        mini_batches_number -= 1

    return gradients_accumulation


def compute_accumulated_gradients_dnn(model, train_data, loss_fn, args):
    """
    Accumulate a number of gradients equal to the number of minibatches defined
    @param model: pretrained model
    @param train_data: training dataset
    @return: the accumulated gradients for each layer of the pretrained network
    """
    gradients_accumulation = []
    mini_batches_number = args.minibatch_number
    dense_parameters_gradients_indexes = None
    first_gradients = True

    for images, labels in train_data:

        if mini_batches_number <= 0:
            break

        # compute gradients and accumulate it
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        if dense_parameters_gradients_indexes == None:
            dense_parameters_gradients_indexes = get_fully_connected_layers_gradient_indexes(
                gradients=gradients)

        feature_extraction_gradients = [gradients[index] for index in dense_parameters_gradients_indexes]

        if first_gradients:
            [gradients_accumulation.append(gradient) for gradient in feature_extraction_gradients]
            first_gradients = False
        else:
            for gradient_index in range(len(gradients_accumulation)):
                gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                     feature_extraction_gradients[gradient_index])
        mini_batches_number -= 1

    return gradients_accumulation


def compute_accumulated_gradients_debug(model, train_data, loss_fn, args):
    """
    Accumulate a number of gradients equal to the number of minibatches defined
    @param model: pretrained model
    @param train_data: training dataset
    @return: the accumulated gradients for each layer of the pretrained network
    """
    gradients_accumulation = []
    first_gradients = True
    mini_batches_number = args.minibatch_number
    convolution_parameters_gradients_indexes = None

    # compute gradients and accumulate it
    with tf.GradientTape() as tape:
        predictions = model(
            tf.keras.applications.mobilenet.preprocess_input(tf.convert_to_tensor(tf.random.normal([64, 224, 224, 3]))),
            training=True)

        labels = tf.convert_to_tensor(tf.random.uniform([64], minval=1, maxval=1001, dtype=tf.int32))
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    if convolution_parameters_gradients_indexes == None:
        convolution_parameters_gradients_indexes = get_feature_extraction_layers_gradient_indexes(
            gradients=gradients)

    convolution_gradients = [gradients[index] for index in convolution_parameters_gradients_indexes]

    if first_gradients:
        [gradients_accumulation.append(gradient) for gradient in convolution_gradients]
        first_gradients = False
    else:
        for gradient_index in range(len(gradients_accumulation)):
            gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                 convolution_gradients[gradient_index])
    mini_batches_number -= 1

    return gradients_accumulation


def compute_accumulated_gradients_ds_cnn(model, train_data, loss_fn, args):
    """
    Accumulate a number of gradients equal to the number of minibatches defined
    @param model: pretrained model
    @param train_data: training dataset
    @return: the accumulated gradients for each layer of the pretrained network
    """
    gradients_accumulation = []
    first_gradients = True
    mini_batches_number = args.minibatch_number
    convolution_parameters_gradients_indexes = None

    for images, labels in train_data:

        if mini_batches_number <= 0:
            break

        # compute gradients and accumulate it
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)

        if convolution_parameters_gradients_indexes == None:
            convolution_parameters_gradients_indexes = get_feature_extraction_layers_gradient_indexes(
                gradients=gradients)

        convolution_gradients = [gradients[index] for index in convolution_parameters_gradients_indexes]

        if first_gradients:
            [gradients_accumulation.append(gradient) for gradient in convolution_gradients]
            first_gradients = False
        else:
            for gradient_index in range(len(gradients_accumulation)):
                gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                     convolution_gradients[gradient_index])
        mini_batches_number -= 1

    return gradients_accumulation


def compute_accumulated_gradients_mobilenetv1(model, train_data, loss_fn, args, debug=False):
    """
    Accumulate a number of gradients equal to the number of minibatches defined
    @param model: pretrained model
    @param train_data: training dataset
    @return: the accumulated gradients for each layer of the pretrained network
    """
    gradients_accumulation = []
    first_gradients = True
    mini_batches_number = args.minibatch_number
    convolution_parameters_gradients_indexes = None

    for images, labels in train_data:

        if mini_batches_number <= 0:
            break

        # compute gradients and accumulate it
        with tf.GradientTape() as tape:
            predictions = model(tf.convert_to_tensor(tf.random.normal([64, 224, 224, 3]))) if debug else model(
                tf.keras.applications.mobilenet.preprocess_input(images), training=True)

            labels = tf.convert_to_tensor(labels)
            loss = loss_fn(tf.random.uniform([64], minval=1, maxval=1001, dtype=tf.int32) if debug else labels,
                           predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        if convolution_parameters_gradients_indexes == None:
            convolution_parameters_gradients_indexes = get_feature_extraction_layers_gradient_indexes(
                gradients=gradients)

        convolution_gradients = [gradients[index] for index in convolution_parameters_gradients_indexes]

        if first_gradients:
            [gradients_accumulation.append(gradient) for gradient in convolution_gradients]
            first_gradients = False
        else:
            for gradient_index in range(len(gradients_accumulation)):
                gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                     convolution_gradients[gradient_index])
        mini_batches_number -= 1

    return gradients_accumulation


def get_convolutional_layers_number(model):
    conv_layers_number = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, Reds_2DConvolution_Standard):
            conv_layers_number += 1

    return conv_layers_number


def assign_pretrained_ds_convolution_filters_ds_cnn(model, permuted_convolutional_filters, permuted_convolutional_bias,
                                                    trainable_assigned_depthwise_convolution=True,
                                                    trainable_assigned_pointwise_convolution=True):
    """
    Given a model and a list of permuted convolutional filters and bias assign the corresponding filters and bias to the
    layer
    """

    layer_index = 0
    first_layer = True
    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.weights[0].assign(
                permuted_convolutional_filters[layer_index])
            layer.weights[1].assign(
                permuted_convolutional_bias[layer_index])

            if first_layer:
                first_layer = False
                layer.trainable = False
            else:
                layer.trainable = trainable_assigned_pointwise_convolution

            layer_index += 1

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.weights[0].assign(
                permuted_convolutional_filters[layer_index])
            layer.weights[1].assign(
                permuted_convolutional_bias[layer_index])

            layer.trainable = trainable_assigned_depthwise_convolution
            layer_index += 1


def assign_pretrained_ds_convolution_filters(model, permuted_convolutional_filters,
                                             permuted_convolutional_bias=None,
                                             trainable_assigned_depthwise_convolution=True,
                                             trainable_assigned_pointwise_convolution=True):
    """
    Given a model and a list of permuted convolutional filters and bias assign the corresponding filters and bias to the
    layer
    """

    layer_index = 0
    first_layer = True
    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.weights[0].assign(
                permuted_convolutional_filters[layer_index])
            if permuted_convolutional_bias is not None:
                layer.weights[1].assign(
                    permuted_convolutional_bias[layer_index])

            if first_layer:
                first_layer = False
                layer.trainable = False
            else:
                layer.trainable = trainable_assigned_pointwise_convolution

            layer_index += 1

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.weights[0].assign(
                permuted_convolutional_filters[layer_index])
            if permuted_convolutional_bias is not None:
                layer.weights[1].assign(
                    permuted_convolutional_bias[layer_index])

            layer.trainable = trainable_assigned_depthwise_convolution
            layer_index += 1


def assign_pretrained_dense_layers(model, permuted_dense_weights, permuted_dense_bias, trainable_assigned_dense=True):
    """
    Given a model and a list of permuted dense neurons and bias assign the corresponding neuron and bias to the
    layer
    """

    dense_layer_index = 0
    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Dense):
            layer.weights[0].assign(
                permuted_dense_weights[dense_layer_index])
            layer.weights[1].assign(
                permuted_dense_bias[dense_layer_index])

            layer.trainable = trainable_assigned_dense
            dense_layer_index += 1


def assign_pretrained_convolutional_filters(model, permuted_convolutional_filters, permuted_convolutional_bias,
                                            trainable_assigned_convolution=True):
    """
    Given a model and a list of permuted convolutional filters and bias assign the corresponding filters and bias to the
    layer
    """

    convolution_layer_index = 0
    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(
                layer, Reds_2DConvolution_Standard):
            layer.weights[0].assign(
                permuted_convolutional_filters[convolution_layer_index])
            layer.weights[1].assign(
                permuted_convolutional_bias[convolution_layer_index])

            layer.trainable = trainable_assigned_convolution
            convolution_layer_index += 1


def permute_ds_batch_norm_layer(layer, permutations, trainable_assigned_batch_norm):
    for variable_index in range(len(layer.variables)):
        layer.weights[variable_index].assign(
            tf.gather(layer.weights[variable_index],
                      permutations))
    layer.trainable = trainable_assigned_batch_norm


def permute_batch_norm_ds_cnn_layers(model, permutations_order, trainable_assigned_batch_norm=False,
                                     trainable_pointwise_batch_norm=True, first_indexes_batch_norm=None):
    """
    Permute the batch normalization layers' parameters: gamma, beta, moving_mean and moving_var learned during training
    """
    if first_indexes_batch_norm is None:
        layers_indexes_to_permute = [2, 5]
    else:
        layers_indexes_to_permute = first_indexes_batch_norm

    permute_ds_batch_norm_layer(model.layers[layers_indexes_to_permute[0]], permutations_order[0],
                                trainable_assigned_batch_norm)
    permute_ds_batch_norm_layer(model.layers[layers_indexes_to_permute[1]], permutations_order[0],
                                trainable_assigned_batch_norm)

    permutation_index = 2
    layer_index = 7
    for layer in model.layers[layer_index:]:

        if isinstance(layer, tf.keras.layers.BatchNormalization) and isinstance(model.layers[layer_index - 1],
                                                                                tf.keras.layers.DepthwiseConv2D):
            permute_ds_batch_norm_layer(layer=layer, permutations=permutations_order[permutation_index],
                                        trainable_assigned_batch_norm=trainable_assigned_batch_norm)
            permutation_index += 1

        if isinstance(layer, tf.keras.layers.BatchNormalization) and isinstance(model.layers[layer_index - 1],
                                                                                tf.keras.layers.Conv2D):
            layer.trainable = trainable_pointwise_batch_norm

        layer_index += 1


def permute_batch_norm_mobilenetv1_layers(model, permutations_order, trainable_assigned_batch_norm=False,
                                          trainable_pointwise_batch_norm=True):
    """
    Permute the batch normalization layers' parameters: gamma, beta, moving_mean and moving_var learned during training
    """
    layers_indexes_to_permute = [2, 5]

    permute_ds_batch_norm_layer(model.layers[layers_indexes_to_permute[0]], permutations_order[0],
                                trainable_assigned_batch_norm)
    permute_ds_batch_norm_layer(model.layers[layers_indexes_to_permute[1]], permutations_order[0],
                                trainable_assigned_batch_norm)

    permutation_index = 2
    layer_index = 6
    for layer in model.layers[layer_index:]:

        if isinstance(layer, tf.keras.layers.BatchNormalization) and isinstance(model.layers[layer_index - 1],
                                                                                tf.keras.layers.DepthwiseConv2D):
            permute_ds_batch_norm_layer(layer=layer, permutations=permutations_order[permutation_index],
                                        trainable_assigned_batch_norm=trainable_assigned_batch_norm)
            permutation_index += 1

        if isinstance(layer, tf.keras.layers.BatchNormalization) and isinstance(model.layers[layer_index - 1],
                                                                                tf.keras.layers.Conv2D):
            layer.trainable = trainable_pointwise_batch_norm

        layer_index += 1


def permute_batch_normalization_layers(model, filters_descending_ranking, trainable_assigned_batch_norm=False):
    """
    Permute the batch normalization layers' parameters: gamma, beta, moving_mean and moving_var learned during training
    """

    batch_normalization_layer_index = 0
    convolution_batch_normalization_number = len(get_standard_cnn_feature_extraction_layers(model=model))

    for layer in model.layers:

        if convolution_batch_normalization_number == 0:
            break

        if isinstance(layer, tf.keras.layers.BatchNormalization):

            for variable_index in range(len(layer.variables)):
                layer.weights[variable_index].assign(
                    tf.gather(layer.weights[variable_index],
                              filters_descending_ranking[batch_normalization_layer_index]))

            batch_normalization_layer_index += 1
            convolution_batch_normalization_number -= 1
            layer.trainable = trainable_assigned_batch_norm


def permute_tensor_units_components(tensor, unrolled_dimension, permutation_order):
    """
    Permute the tensor units components according to the permutation order, e.g. the filters' kernels in the convolutional layer
    """

    units = tf.unstack(tensor, axis=unrolled_dimension)
    reordered_tensors = []

    for permutation in permutation_order:
        reordered_tensors.append(units[permutation])

    return tf.stack(reordered_tensors, axis=unrolled_dimension)


def reorder_tensor_units_over_dimension(tensor, unrolled_dimension, permutation_order):
    units = tf.unstack(tensor, axis=unrolled_dimension)
    reordered_tensors = []

    for unit in units:
        reordered_tensors.append(tf.gather(params=unit, indices=permutation_order, axis=unrolled_dimension - 1))

    return tf.stack(reordered_tensors, axis=unrolled_dimension)


def permute_filters_ds_cnn(model, filters_descending_ranking):
    """

        @param model: pretrained model
        @param filters_descending_ranking: list of list containing the indexes of the filters ranked in descent order
        @param convolutional_layer_number: number of convolutional layer contained in the pretrained model
        @return: list of filters permuted accordingly to the ranking order and the order of the last filters permutation used
        to then permute the first layer of the classification layer head.
    """

    model_convolution_layers = get_all_convolution_layers(model=model)
    model_convolution_bias = get_all_convolution_biases(model=model)

    first_layer = True
    first_depthwise_convolution = True

    layer_index = 0
    permutation_index = 0

    while permutation_index < len(filters_descending_ranking):

        filters_permutation = filters_descending_ranking[permutation_index]

        if first_layer is True:

            model_convolution_layers[layer_index] = permute_tensor_units_components(
                tensor=model_convolution_layers[layer_index], unrolled_dimension=3,
                permutation_order=filters_permutation)

            model_convolution_bias[layer_index] = tf.gather(model_convolution_bias[layer_index], filters_permutation)

            # sort next layers filters order based on the first layer filters order to avoid feature extraction loss
            model_convolution_layers[layer_index + 1] = permute_tensor_units_components(
                tensor=model_convolution_layers[layer_index + 1], unrolled_dimension=2,
                permutation_order=filters_permutation)
            model_convolution_bias[layer_index + 1] = tf.gather(model_convolution_bias[layer_index + 1],
                                                                filters_permutation)

            # sort first pointwise kernels to have the same order of the depthwise layer filters
            model_convolution_layers[layer_index + 2] = reorder_tensor_units_over_dimension(
                tensor=model_convolution_layers[layer_index + 2], unrolled_dimension=3,
                permutation_order=filters_permutation)

            model_convolution_bias[layer_index + 2] = tf.gather(model_convolution_bias[layer_index + 2],
                                                                filters_permutation)
            first_layer = False
            permutation_index += 2

        else:

            if first_depthwise_convolution is not True and model_convolution_layers[layer_index].shape[1] != 1:

                model_convolution_layers[layer_index] = permute_tensor_units_components(
                    tensor=model_convolution_layers[layer_index], unrolled_dimension=2,
                    permutation_order=filters_permutation)

                model_convolution_bias[layer_index] = tf.gather(model_convolution_bias[layer_index],
                                                                filters_permutation)

                # swap next pointwise layer kernels based on the previous depthwise layer filters order
                model_convolution_layers[layer_index + 1] = reorder_tensor_units_over_dimension(
                    tensor=model_convolution_layers[layer_index + 1], unrolled_dimension=3,
                    permutation_order=filters_permutation)

                model_convolution_bias[layer_index + 1] = tf.gather(model_convolution_bias[layer_index + 1],
                                                                    filters_permutation)
                permutation_index += 1
            else:
                first_depthwise_convolution = False

        layer_index += 1

    return model_convolution_layers, model_convolution_bias


def permute_filters_mobilenet(model, filters_descending_ranking, bias=False):
    """

        @param model: pretrained model
        @param filters_descending_ranking: list of list containing the indexes of the filters ranked in descent order
        @param convolutional_layer_number: number of convolutional layer contained in the pretrained model
        @return: list of filters permuted accordingly to the ranking order and the order of the last filters permutation used
        to then permute the first layer of the classification layer head.
    """

    model_convolution_layers = get_all_convolution_layers(model=model)
    if bias is True:
        model_convolution_bias = get_all_convolution_biases(model=model)

    first_layer = True
    first_depthwise_convolution = True

    layer_index = 0
    permutation_index = 0

    while permutation_index < len(filters_descending_ranking):

        filters_permutation = filters_descending_ranking[permutation_index]

        if first_layer is True:

            model_convolution_layers[layer_index] = permute_tensor_units_components(
                tensor=model_convolution_layers[layer_index], unrolled_dimension=3,
                permutation_order=filters_permutation)

            if bias:
                model_convolution_bias[layer_index] = tf.gather(model_convolution_bias[layer_index],
                                                                filters_permutation)

            # sort next layers filters order based on the first layer filters order to avoid feature extraction loss
            model_convolution_layers[layer_index + 1] = permute_tensor_units_components(
                tensor=model_convolution_layers[layer_index + 1], unrolled_dimension=2,
                permutation_order=filters_permutation)
            if bias:
                model_convolution_bias[layer_index + 1] = tf.gather(model_convolution_bias[layer_index + 1],
                                                                    filters_permutation)

            # sort first pointwise kernels to have the same order of the depthwise layer filters
            model_convolution_layers[layer_index + 2] = reorder_tensor_units_over_dimension(
                tensor=model_convolution_layers[layer_index + 2], unrolled_dimension=3,
                permutation_order=filters_permutation)
            if bias:
                model_convolution_bias[layer_index + 2] = tf.gather(model_convolution_bias[layer_index + 2],
                                                                    filters_permutation)
            first_layer = False
            permutation_index += 2

        else:

            if first_depthwise_convolution is not True and model_convolution_layers[layer_index].shape[1] != 1:

                model_convolution_layers[layer_index] = permute_tensor_units_components(
                    tensor=model_convolution_layers[layer_index], unrolled_dimension=2,
                    permutation_order=filters_permutation)

                if bias:
                    model_convolution_bias[layer_index] = tf.gather(model_convolution_bias[layer_index],
                                                                    filters_permutation)

                # swap next pointwise layer kernels based on the previous depthwise layer filters order
                model_convolution_layers[layer_index + 1] = reorder_tensor_units_over_dimension(
                    tensor=model_convolution_layers[layer_index + 1], unrolled_dimension=3,
                    permutation_order=filters_permutation)

                if bias:
                    model_convolution_bias[layer_index + 1] = tf.gather(model_convolution_bias[layer_index + 1],
                                                                        filters_permutation)
                permutation_index += 1
            else:
                first_depthwise_convolution = False

        layer_index += 1

    if bias:
        return model_convolution_layers, model_convolution_bias
    else:
        return model_convolution_layers


def get_dense_layers_biases(model):
    dense_layers_biases = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layers_biases.append(layer.weights[1])
    return dense_layers_biases


def permute_neurons_dnn(model, neurons_descending_ranking):
    """

    @param model: pretrained model
    @param neurons_descending_ranking: list of list containing the indexes of the filters ranked in descent order
    @return: list of filters permuted accordingly to the ranking order and the order of the last filters permutation used
    to then permute the first layer of the classification layer head.
    """
    model_dense_layers = get_dense_layers(model=model)
    model_dense_biases = get_dense_layers_biases(model=model)

    for layer_index in range(len(model_dense_layers) - 1):

        neurons_permutation = neurons_descending_ranking[layer_index]

        model_dense_layers[layer_index] = tf.gather(model_dense_layers[layer_index], neurons_permutation, axis=1)
        model_dense_biases[layer_index] = tf.gather(model_dense_biases[layer_index], neurons_permutation)

        if layer_index + 1 <= len(model_dense_layers) - 1:

            permuted_next_layer_units = []

            # For each neuron, permute the order and append it to the list
            for unit_index in range(model_dense_layers[layer_index + 1].shape[1]):
                unit = model_dense_layers[layer_index + 1][:, unit_index]  # Extract the unit.
                permuted_column = tf.gather(unit, neurons_permutation)  # Permute the unit.
                permuted_next_layer_units.append(permuted_column)

            model_dense_layers[layer_index + 1] = tf.stack(permuted_next_layer_units, axis=1)

    return model_dense_layers, model_dense_biases


def permute_filters_cnn(model, filters_descending_ranking):
    """

    @param model: pretrained model
    @param filters_descending_ranking: list of list containing the indexes of the filters ranked in descent order
    @param convolutional_layer_number: number of convolutional layer contained in the pretrained model
    @return: list of filters permuted accordingly to the ranking order and the order of the last filters permutation used
    to then permute the first layer of the classification layer head.
    """
    model_convolution_layers = get_standard_cnn_feature_extraction_layers(model=model)
    model_convolution_bias = get_standard_cnn_convolution_bias(model=model)
    last_permutation_filters_order = None

    for layer_index in range(len(model_convolution_layers)):

        weight_tensor = model_convolution_layers[layer_index]
        filters_permutation = filters_descending_ranking[layer_index]
        model_convolution_bias[layer_index] = tf.gather(model_convolution_bias[layer_index], filters_permutation)
        arranged_filter = []
        for filter_index in filters_permutation:
            filter_i = weight_tensor[:, :, :, filter_index]
            arranged_filter.append(filter_i)

        # stack filters sequentially based on importance score
        model_convolution_layers[layer_index] = tf.stack(arranged_filter, axis=3)

        # convolution layer is not the last, permute next convolution layer kernels to avoid feature extraction loss
        if layer_index + 1 <= len(model_convolution_layers) - 1:
            next_layer_weight_tensor = model_convolution_layers[layer_index + 1]
            filter_kernels = tf.unstack(next_layer_weight_tensor, axis=3)
            reordered_filters = []

            for filter in filter_kernels:

                permuted_kernels = []
                for kernel_index in filters_permutation:
                    kernel = filter[:, :, kernel_index]
                    permuted_kernels.append(kernel)

                reordered_filters.append(tf.stack(permuted_kernels, axis=2))

            model_convolution_layers[layer_index + 1] = tf.stack(reordered_filters, axis=3)

        if layer_index + 1 == len(model_convolution_layers):
            last_permutation_filters_order = filters_permutation

    return model_convolution_layers, model_convolution_bias, last_permutation_filters_order


def get_model_convolutional_layers(model):
    convolutional_layer_number = get_model_convolutional_layers_number(model=model)
    return model.trainable_weights[0:convolutional_layer_number]


def compute_descending_filters_score_indexes_ds_cnn(model, importance_score_filters, units_number=64):
    """
    @param model: pretrained model
    @param importance_score_filters: list of list containing the pre-computed importance score for each convolutional
    layers' filters
    @return: list of list containing sorted tuples with: (filter_index, filter_importance_score) for each layer
    """
    model_convolution_layers = get_feature_extraction_layers(model=model)

    descending_importance_score_indexes = [list(range(units_number)) for
                                           _
                                           in
                                           range(len(model_convolution_layers))]

    descending_importance_score_scores = [list(range(units_number)) for
                                          _
                                          in
                                          range(len(model_convolution_layers))]

    for layer_index in range(len(importance_score_filters)):

        filter_index = 0

        for filter_importance_score in importance_score_filters[layer_index]:
            descending_importance_score_indexes[layer_index][filter_index] = (filter_index, filter_importance_score)
            filter_index += 1

        list_tuples_score = sorted(descending_importance_score_indexes[layer_index],
                                   key=lambda x: x[1], reverse=True)

        descending_importance_score_indexes[layer_index] = list(
            list_tuples_score[x][0] for x in range(len(list_tuples_score)))

        descending_importance_score_scores[layer_index] = list(
            list_tuples_score[x][1] for x in range(len(list_tuples_score)))

    return descending_importance_score_indexes, descending_importance_score_scores

def compute_descending_filters_score_indexes_mobilenet(model, importance_score_filters, units_number=None):
    """
    @param model: pretrained model
    @param importance_score_filters: list of list containing the pre-computed importance score for each convolutional
    layers' filters
    @return: list of list containing sorted tuples with: (filter_index, filter_importance_score) for each layer
    """
    model_convolution_layers = get_feature_extraction_layers(model=model)

    descending_importance_score_indexes = [list(range(len(importance_score_filters[i]))) for
                                           i
                                           in
                                           range(len(model_convolution_layers))]

    descending_importance_score_scores = [list(range(len(importance_score_filters[i]))) for
                                          i
                                          in
                                          range(len(model_convolution_layers))]

    for layer_index in range(len(importance_score_filters)):

        filter_index = 0

        for filter_importance_score in importance_score_filters[layer_index]:
            descending_importance_score_indexes[layer_index][filter_index] = (filter_index, filter_importance_score)
            filter_index += 1

        list_tuples_score = sorted(descending_importance_score_indexes[layer_index],
                                   key=lambda x: x[1], reverse=True)

        descending_importance_score_indexes[layer_index] = list(
            list_tuples_score[x][0] for x in range(len(list_tuples_score)))

        descending_importance_score_scores[layer_index] = list(
            list_tuples_score[x][1] for x in range(len(list_tuples_score)))

    return descending_importance_score_indexes, descending_importance_score_scores


def compute_descending_filters_score_indexes_dnn(model, importance_score_neurons):
    """
    @param model: pretrained model
    @param importance_score_neurons: list of list containing the pre-computed importance score for each dense
    layers' neurons
    @return: list of list containing sorted tuples with: (neuron_index, neuron_importance_score) for each layer
    """
    model_dense_layers = get_dense_layers(model=model)[:-1]

    descending_importance_score_indexes = [list(range(model_dense_layers[layer_index].shape[1])) for
                                           layer_index
                                           in
                                           range(len(model_dense_layers))]

    descending_importance_score_scores = [list(range(model_dense_layers[layer_index].shape[1])) for
                                          layer_index
                                          in
                                          range(len(model_dense_layers))]

    for layer_index in range(len(importance_score_neurons)):

        neuron_index = 0

        for neuron_importance_score in importance_score_neurons[layer_index]:
            descending_importance_score_indexes[layer_index][neuron_index] = (neuron_index, neuron_importance_score)
            neuron_index += 1

        list_tuples_score = sorted(descending_importance_score_indexes[layer_index],
                                   key=lambda x: x[1], reverse=True)

        descending_importance_score_indexes[layer_index] = list(
            list_tuples_score[x][0] for x in range(len(list_tuples_score)))

        descending_importance_score_scores[layer_index] = list(
            list_tuples_score[x][1] for x in range(len(list_tuples_score)))

    return descending_importance_score_indexes, descending_importance_score_scores


def compute_descending_filters_score_indexes_cnn(model, importance_score_filters):
    """
    @param model: pretrained model
    @param importance_score_filters: list of list containing the pre-computed importance score for each convolutional
    layers' filters
    @return: list of list containing sorted tuples with: (filter_index, filter_importance_score) for each layer
    """
    model_convolution_layers = get_standard_cnn_feature_extraction_layers(model=model)
    descending_importance_score_indexes = [list(range(model_convolution_layers[layer_index].shape[3])) for
                                           layer_index
                                           in
                                           range(len(model_convolution_layers))]

    descending_importance_score_scores = [list(range(model_convolution_layers[layer_index].shape[3])) for
                                          layer_index
                                          in
                                          range(len(model_convolution_layers))]

    for layer_index in range(len(importance_score_filters)):

        filter_index = 0

        for filter_importance_score in importance_score_filters[layer_index]:
            descending_importance_score_indexes[layer_index][filter_index] = (filter_index, filter_importance_score)
            filter_index += 1

        list_tuples_score = sorted(descending_importance_score_indexes[layer_index],
                                   key=lambda x: x[1], reverse=True)

        descending_importance_score_indexes[layer_index] = list(
            list_tuples_score[x][0] for x in range(len(list_tuples_score)))

        descending_importance_score_scores[layer_index] = list(
            list_tuples_score[x][1] for x in range(len(list_tuples_score)))

    return descending_importance_score_indexes, descending_importance_score_scores


def get_ds_convolution_layers(model):
    convolution_layers = []

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            convolution_layers.append(layer.weights[0])

        if isinstance(layer, tf.keras.layers.Conv2D):
            convolution_layers.append(layer.weights[0])

    return convolution_layers


def get_ds_convolution_bias(model):
    convolution_bias = []
    first_layer = True
    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D) or isinstance(layer,
                                                                            tf.keras.layers.Conv2D) and first_layer is True:
            convolution_bias.append(layer.weights[1])

            if first_layer is True:
                first_layer = False

    return convolution_bias


def get_all_convolution_layers(model):
    convolution_layers = []

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            convolution_layers.append(layer.weights[0])

        if isinstance(layer, tf.keras.layers.Conv2D):
            convolution_layers.append(layer.weights[0])

    return convolution_layers


def get_all_convolution_biases(model):
    convolution_bias = []

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            convolution_bias.append(layer.weights[1])

        if isinstance(layer, tf.keras.layers.Conv2D):
            convolution_bias.append(layer.weights[1])

    return convolution_bias


def get_standard_cnn_feature_extraction_layers(model):
    convolution_layers = []

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Conv2D):
            convolution_layers.append(layer.weights[0])

    return convolution_layers


def get_feature_extraction_layers(model):
    convolution_layers = []

    first_layer = True

    for layer in model.layers:

        if first_layer and isinstance(layer, tf.keras.layers.Conv2D):
            convolution_layers.append(layer.weights[0])
            first_layer = False

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            convolution_layers.append(layer.weights[0])
    return convolution_layers


def get_standard_cnn_convolution_bias(model):
    convolution_bias = []

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Conv2D):
            convolution_bias.append(layer.weights[1])

    return convolution_bias


def get_convolution_bias(model):
    convolution_bias = []

    first_layer = True
    for layer in model.layers:

        if first_layer and isinstance(layer, tf.keras.layers.Conv2D):
            convolution_bias.append(layer.weights[1])
            first_layer = False

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            convolution_bias.append(layer.weights[1])

    return convolution_bias


def get_dense_layers(model):
    dense_layers = []

    for layer in model.layers:

        if isinstance(layer, tf.keras.layers.Dense):
            dense_layers.append(layer.weights[0])

    return dense_layers


def compute_neurons_importance_score_dnn(model, gradients_accumulation):
    """

    @param model: pretrained CNN model
    @param gradients_accumulation: accumulation of the gradients computed on a subset of the training data
    @return: return a list of list containing the importance score for each filter in the CNN convolutional layer
    """
    model_dense_layers = get_dense_layers(model=model)[:-1]

    # create a list of list for each layer containing a number of space equal to the number of filters
    importance_score_neurons = [list(range(model_dense_layers[layer_index].shape[1])) for layer_index
                                in
                                range(len(model_dense_layers))]

    for layer_index in range(len(model_dense_layers)):

        importance_score_layer = tf.abs(x=tf.math.multiply(x=model_dense_layers[layer_index],
                                                           y=gradients_accumulation[layer_index]))

        neuron_index = 0

        for neuron in tf.transpose(importance_score_layer):
            importance_score_neurons[layer_index][neuron_index] = tf.math.reduce_sum(neuron)
            neuron_index += 1

        for importance_score_neuron_index in range(len(importance_score_neurons[layer_index])):
            importance_score_neurons[layer_index][importance_score_neuron_index] = float(
                importance_score_neurons[
                    layer_index][importance_score_neuron_index])

    return importance_score_neurons


def compute_filters_importance_score_cnn(model, gradients_accumulation):
    """

    @param model: pretrained CNN model
    @param gradients_accumulation: accumulation of the gradients computed on a subset of the training data
    @return: return a list of list containing the importance score for each filter in the CNN convolutional layer
    """
    model_convolution_layers = get_standard_cnn_feature_extraction_layers(model=model)

    # create a list of list for each layer containing a number of space equal to the number of filters
    importance_score_filters = [list(range(model_convolution_layers[layer_index].shape[3])) for layer_index
                                in
                                range(len(model_convolution_layers))]

    for layer_index in range(len(model_convolution_layers)):

        multiplication_and_pow = tf.abs(x=tf.math.multiply(x=model_convolution_layers[layer_index],
                                                           y=gradients_accumulation[layer_index]))

        filter_index = 0

        for filter in tf.transpose(multiplication_and_pow, (3, 0, 1, 2)):
            importance_score_filters[layer_index][filter_index] = tf.math.reduce_sum(filter)
            filter_index += 1

        for importance_score_filter_index in range(len(importance_score_filters[layer_index])):
            importance_score_filters[layer_index][importance_score_filter_index] = float(
                importance_score_filters[
                    layer_index][importance_score_filter_index])

    return importance_score_filters


def compute_pointwise_importance_score_ds_cnn(model, gradients_accumulation_pointwise, int_scale=1e5):
    """
            @param model: pretrained CNN model
            @param gradients_accumulation: accumulation of the gradients computed on a subset of the training data
            @return: return a list of list of list containing the importance score for kernel contained in each pointwise filter in each pointwise layer
            """
    model_pointwise_layers_weights = get_pointwise_convolutions_layers_weights(
        model=model)

    importance_score_pointwise_filters = [[] for _ in range(len(model_pointwise_layers_weights))]

    # create a list of lists of lists to store the score of kernel contained in each pointwise filter in each pointwise layer
    for layer_index in range(len(model_pointwise_layers_weights)):
        for filter_index in range(model_pointwise_layers_weights[layer_index].shape[3]):
            importance_score_pointwise_filters[layer_index].append(
                list(range(model_pointwise_layers_weights[layer_index].shape[2])))

    for layer_index in range(len(model_pointwise_layers_weights)):

        importance_scores_parameters = tf.abs(x=tf.math.multiply(x=model_pointwise_layers_weights[layer_index],
                                                                 y=gradients_accumulation_pointwise[layer_index]))

        if importance_scores_parameters.shape[1] == 1:

            filter_index = 0
            # point-wise convolution layer
            for filter in tf.transpose(importance_scores_parameters, (3, 0, 1, 2)):

                flattened_filter = tf.reshape(filter, [-1])
                for kernel in range(len(flattened_filter)):
                    importance_score_pointwise_filters[layer_index][filter_index][kernel] = round(
                        int_scale * float((flattened_filter[kernel])))

                filter_index += 1

    return importance_score_pointwise_filters

def compute_filters_importance_score_feature_extraction_filters_l2_norm(model):
    """
        @param model: pretrained CNN model
        @return: return a list of list containing the importance score for each filter in the CNN convolutional layer
        """
    model_convolution_layers = get_feature_extraction_layers(model=model)

    importance_score_filters = []
    # add first layer number of filter
    importance_score_filters.append(list(range(model_convolution_layers[0].shape[3])))

    # create a list of list for each depth-wise convolution layer containing the number of spatial filters in the layer
    [importance_score_filters.append(list(range(model_convolution_layers[layer_index].shape[2]))) for layer_index
     in
     range(1, len(model_convolution_layers), 1)]

    first_layer = True

    for layer_index in range(len(model_convolution_layers)):

        importance_scores_parameters = tf.abs(x=model_convolution_layers[layer_index])

        filter_index = 0

        # first layer is a standard convolution layer
        if first_layer is True:
            for filter in tf.transpose(importance_scores_parameters, (3, 0, 1, 2)):
                importance_score_filters[layer_index][filter_index] = float(tf.norm(tensor=filter, ord=2))
                filter_index += 1
            first_layer = False

        else:
            # depth-wise convolution layers
            for filter in tf.transpose(importance_scores_parameters, (2, 0, 1, 3)):
                importance_score_filters[layer_index][filter_index] = float(tf.norm(tensor=filter, ord=2))
                filter_index += 1

    return importance_score_filters
def compute_filters_importance_score_feature_extraction_filters_l1_norm(model):
    """
        @param model: pretrained CNN model
        @return: return a list of list containing the importance score for each filter in the CNN convolutional layer
        """
    model_convolution_layers = get_feature_extraction_layers(model=model)

    importance_score_filters = []
    # add first layer number of filter
    importance_score_filters.append(list(range(model_convolution_layers[0].shape[3])))

    # create a list of list for each depth-wise convolution layer containing the number of spatial filters in the layer
    [importance_score_filters.append(list(range(model_convolution_layers[layer_index].shape[2]))) for layer_index
     in
     range(1, len(model_convolution_layers), 1)]

    first_layer = True

    for layer_index in range(len(model_convolution_layers)):

        importance_scores_parameters = tf.abs(x=model_convolution_layers[layer_index])

        filter_index = 0

        # first layer is a standard convolution layer
        if first_layer is True:
            for filter in tf.transpose(importance_scores_parameters, (3, 0, 1, 2)):
                importance_score_filters[layer_index][filter_index] = float(tf.norm(tensor=filter, ord=1))
                filter_index += 1
            first_layer = False

        else:
            # depth-wise convolution layers
            for filter in tf.transpose(importance_scores_parameters, (2, 0, 1, 3)):
                importance_score_filters[layer_index][filter_index] = float(tf.norm(tensor=filter, ord=1))
                filter_index += 1

    return importance_score_filters


def compute_filters_importance_score_feature_extraction_filters(model, gradients_accumulation):
    """
        @param model: pretrained CNN model
        @param gradients_accumulation: accumulation of the gradients computed on a subset of the training data
        @return: return a list of list containing the importance score for each filter in the CNN convolutional layer
        """
    model_convolution_layers = get_feature_extraction_layers(model=model)

    importance_score_filters = []
    # add first layer number of filter
    importance_score_filters.append(list(range(model_convolution_layers[0].shape[3])))

    # create a list of list for each depth-wise convolution layer containing the number of spatial filters in the layer
    [importance_score_filters.append(list(range(model_convolution_layers[layer_index].shape[2]))) for layer_index
     in
     range(1, len(model_convolution_layers), 1)]

    first_layer = True

    for layer_index in range(len(model_convolution_layers)):

        importance_scores_parameters = tf.abs(x=tf.math.multiply(x=model_convolution_layers[layer_index],
                                                                 y=gradients_accumulation[layer_index]))

        filter_index = 0

        # first layer is a standard convolution layer
        if first_layer is True:
            for filter in tf.transpose(importance_scores_parameters, (3, 0, 1, 2)):
                importance_score_filters[layer_index][filter_index] = float(tf.math.reduce_sum(filter))
                filter_index += 1
            first_layer = False

        else:
            # depth-wise convolution layers
            for filter in tf.transpose(importance_scores_parameters, (2, 0, 1, 3)):
                importance_score_filters[layer_index][filter_index] = float(tf.math.reduce_sum(filter))
                filter_index += 1

    return importance_score_filters
