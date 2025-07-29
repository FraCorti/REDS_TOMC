import numpy as np
import tensorflow as tf

from utils.batch_normalization import Reds_BatchNormalizationBase
from utils.convolution import get_reds_cnn_architecture, Linear_Adaptive, Reds_2DConvolution_Standard
from utils.ds_convolution import get_reds_ds_cnn_architecture, get_ds_cnn_architecture_early_exits, Reds_DepthwiseConv2D
from utils.ds_convolution_vision_data import get_reds_ds_cnn_vision_architectures
from utils.linear import get_reds_dnn_architecture, Reds_Linear
from utils.logs import log_print
from utils.mobilenets import Reds_MobilenetV1, Reds_MobilenetV1_Leaky


def cross_entropy_loss(y_pred, y):
    # Compute cross entropy loss with a sparse operation
    if y_pred.shape.rank - 1 != y.shape.rank:
        y = tf.squeeze(y, axis=[1])
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(sparse_ce)

def accuracy_vision(y_pred, y):
    # Compute accuracy after extracting class predictions
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)

    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


def accuracy(y_pred, y):
    # Compute accuracy after extracting class predictions
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)

    if class_preds.dtype == tf.int64:  # and y.dtype == tf.int32
        class_preds = tf.cast(x=class_preds, dtype=tf.int32)

    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


def accuracy_vision_cifar10(y_pred, y):
    # Compute accuracy after extracting class predictions
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)

    if class_preds.dtype == tf.int64 and y.dtype == tf.int32:
        class_preds = tf.cast(x=class_preds, dtype=tf.int32)

    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))

def train_step_reds(x_batch, y_batch, loss, acc, model, optimizer, subnetworks_number, mobilenet=False):
    alphas = list(0 for _ in range(0, subnetworks_number))

    # scale loss proportional to subnetwork parameters percentage
    for subnetwork_index in range(0, subnetworks_number):
        subnetworks_parameter_percentage_used = model.get_subnetwork_parameters_percentage(
            subnetwork_index=subnetwork_index)

        alpha = pow((1 - (1 - (subnetworks_parameter_percentage_used))), 0.5)
        alphas[subnetwork_index] = alpha

    gradients_accumulation = []
    first_gradients = True
    batch_losses, batch_accuracies = [], []

    # compute and accumulate subnetworks gradients
    with tf.GradientTape(persistent=True) as tape:

        logits = model(inputs=x_batch if not mobilenet else tf.keras.applications.mobilenet.preprocess_input(x_batch),
                       training=True)

        for subnet_output_index in range(len(logits)):
            batch_loss = loss(logits[subnet_output_index], y_batch)
            batch_losses.append(batch_loss)
            batch_accuracies.append(acc(logits[subnet_output_index], y_batch))

            # scale loss and compute subnetwork gradient
            subnetwork_gradients = tape.gradient(batch_loss * float(
                alphas[subnet_output_index] / np.array(alphas).sum()), model.trainable_variables)

            if first_gradients:
                [gradients_accumulation.append(gradient) for gradient in subnetwork_gradients]
                first_gradients = False
            else:
                for gradient_index in range(len(gradients_accumulation)):
                    gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                         subnetwork_gradients[gradient_index])

    for layer_index, (grad, var) in enumerate(zip(gradients_accumulation, model.trainable_variables)):
        optimizer.update_step(grad, var)

    return batch_losses, batch_accuracies

def train_model(model, train_data, val_data, test_data, loss, acc, optimizer, epochs,
                subnetworks_number, subnetworks_macs, args, message_initial_accuracies, message="", full_training=False,
                importance_score=True, architecture_name="", plot=True, batch_norm_finetuning=False,
                debug=False):
    test_acc_subnetworks = [[] for _ in range(subnetworks_number)]

    log_print(message)

    for epoch in range(epochs):

        batch_losses_test, batch_accs_test = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                       range(subnetworks_number)]

        for x_batch, y_batch in train_data:
            batch_loss_train, batch_accuracy_train = train_step_reds(x_batch, y_batch, loss, acc,
                                                                         model, optimizer,
                                                                         subnetworks_number)
        for x_batch, y_batch in test_data:
            batch_loss_test, batch_accuracy_test = val_step_reds(x_batch, y_batch, loss, acc, model)

            for subnetwork_number in range(len(batch_loss_test)):
                    batch_losses_test[subnetwork_number].append(batch_loss_test[subnetwork_number])
                    batch_accs_test[subnetwork_number].append(batch_accuracy_test[subnetwork_number])

        print(f"Epoch {epoch + 1}/{epochs}")
        for subnetwork_number in range(subnetworks_number):
            test_loss, test_acc = tf.reduce_mean(batch_losses_test[subnetwork_number]), tf.reduce_mean(
                batch_accs_test[subnetwork_number])

            test_acc_subnetworks[subnetwork_number].append(test_acc)

            log_print(
                f"epoch: {epochs} subnetworks MACS: {subnetworks_macs[subnetwork_number]} test accuracy: {100 * test_acc:.3f}% test loss: {test_loss:.3f}")

    return [[test_acc_subnetworks[subnetwork_number][-1]] for subnetwork_number in range(subnetworks_number)]
def set_encoder_layers_training(model, trainable=True):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.trainable = trainable
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.trainable = trainable
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = trainable


def classification_head_finetuning(model, optimizer, train_ds, test_ds, initial_pretrained_test_accuracy, args):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(args.finetune_head_epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


        for images, labels in train_ds:
            standard_step_train(model=model, images=images, labels=labels, loss_fn=loss_fn, optimizer=optimizer,
                                    train_loss=train_loss, train_accuracy=train_accuracy)

        for test_images, test_labels in test_ds:
            standard_step_test(model=model, images=test_images, labels=test_labels, loss_fn=loss_fn,
                                   test_loss=test_loss, test_accuracy=test_accuracy)


        print(
            f'Head finetuning epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
        if float(test_accuracy.result()) * 100 >= float(initial_pretrained_test_accuracy):
            print("Early stopping")
            break

    set_encoder_layers_training(model=model, trainable=True)
    return test_accuracy.result() * 100

def classification_head_finetuning_ds_cnn(model, optimizer, train_ds, test_ds, initial_pretrained_test_accuracy, args,
                                          print_info=True):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(args.finetune_head_epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            standard_step_train(model=model, images=images,
                                    labels=labels, loss_fn=loss_fn, optimizer=optimizer,
                                    train_loss=train_loss, train_accuracy=train_accuracy)
        for test_images, test_labels in test_ds:
            standard_step_test(model=model, images=test_images,
                                   labels=test_labels, loss_fn=loss_fn,
                                   test_loss=test_loss, test_accuracy=test_accuracy)


        if print_info:
            print(
                f'Head finetuning epoch {epoch + 1}, '
                f'Loss: {train_loss.result()}, '
                f'Accuracy: {train_accuracy.result() * 100}, '
                f'Test Loss: {test_loss.result()}, '
                f'Test Accuracy: {test_accuracy.result() * 100}'
            )

        if float(test_accuracy.result()) * 100 >= float(initial_pretrained_test_accuracy):
            print("Early stopping")
            break

    set_encoder_layers_training(model=model, trainable=True)
    return test_accuracy.result() * 100


def convert_dnn_model_to_reds(pretrained_model, train_ds, args, hidden_units, model_settings,
                              trainable_parameters=True, training_from_scratch=False):
    reds_model = get_reds_dnn_architecture(classes=args.classes,
                                           subnetworks_number=args.subnets_number, hidden_units=hidden_units,
                                           model_settings=model_settings)

    for images, labels in train_ds.take(1):
        reds_model.set_subnetworks_number(subnetworks_number=1)
        reds_model(images, training=False)
        reds_model.set_subnetworks_number(subnetworks_number=args.subnets_number)

    layer_index = 0
    for layer in reds_model.layers:

        if isinstance(layer, Reds_Linear):

            if not training_from_scratch:
                layer.weights[0].assign(pretrained_model.layers[layer_index].weights[0])
                layer.weights[1].assign(pretrained_model.layers[layer_index].weights[1])

            layer_index += 1
            reds_model.layers[layer_index].trainable = trainable_parameters

        if isinstance(layer, Linear_Adaptive):

            if not training_from_scratch:
                layer.weights[0].assign(pretrained_model.layers[layer_index].weights[0])
                layer.weights[1].assign(pretrained_model.layers[layer_index].weights[1])

            layer_index += 1
            reds_model.layers[layer_index].trainable = trainable_parameters

    return reds_model


def assign_pretrained_batch_norm_parameters(pretrained_layer, reds_layer, training_from_scratch=False, trainable=True,
                                            slimmable=False):
    """
    Given a pretrained layer assign to the corresponding reds layer the weight and bias of it
    @param pretrained_layer:
    @param reds_layer:
    @param trainable:
    @return: None
    """
    if not training_from_scratch and not slimmable:
        for trainable_parameter_index in range(len(pretrained_layer.variables)):
            reds_layer.weights[trainable_parameter_index].assign(
                pretrained_layer.weights[trainable_parameter_index])

    if not training_from_scratch and slimmable:
        for trainable_parameter_index in range(len(pretrained_layer.variables)):
            reds_layer.weights[trainable_parameter_index].assign(
                pretrained_layer.weights[trainable_parameter_index])

    reds_layer.trainable = trainable


def assign_pretrained_trainable_parameters(pretrained_layer, reds_layer, training_from_scratch=False, trainable=True):
    """
    Given a pretrained layer assign to the corresponding reds layer the weight and bias of it
    @param pretrained_layer:
    @param reds_layer:
    @param trainable:
    @return: None
    """
    if not training_from_scratch:
        for trainable_parameter_index in range(len(pretrained_layer.trainable_variables)):
            reds_layer.weights[trainable_parameter_index].assign(
                pretrained_layer.weights[trainable_parameter_index])

    reds_layer.trainable = trainable


def convert_ds_cnn_model_to_vision_reds(pretrained_model, train_ds, args,
                                        use_bias=True,
                                        model_filters=64,
                                        trainable_parameters=True,
                                        model_size="s",
                                        training_from_scratch=False,
                                        trainable_batch_normalization=False):
    """
        Given a pretrained model retrieve its corresponding reds model (with the same architecture) and assign to it the
        weight and bias of the pretrained model
        @return: reds model initialize with the weight and bias of the pretrained model
    """
    pool_size = None
    feature_vector_size = None
    if model_size == "s":
        pool_size = pretrained_model.layers[27].pool_size
        feature_vector_size = pretrained_model.layers[29].weights[0].shape[0]
    elif model_size == "l":
        pool_size = pretrained_model.layers[33].pool_size
        feature_vector_size = pretrained_model.layers[35].weights[0].shape[0]
    reds_model = get_reds_ds_cnn_vision_architectures(classes=10,
                                                      model_size=model_size,
                                                      model_filters=model_filters,
                                                      subnetworks_number=args.subnets_number, use_bias=use_bias,
                                                      in_channels=1 if args.dataset == "mnist" or args.dataset == "fashion_mnist" else 3,
                                                      debug=False, pool_size=pool_size,
                                                      feature_vector_size=feature_vector_size)

    # forward one sample to initialize the model's weights
    for images, _ in train_ds.take(1):
        reds_model.build(input_shape=images.shape)
        reds_model.set_subnetworks_number(subnetworks_number=1)
        reds_model(inputs=images, training=False)
        reds_model.set_subnetworks_number(subnetworks_number=args.subnets_number)

    for layer_index in range(len(reds_model.layers)):

        if isinstance(pretrained_model.layers[layer_index], tf.keras.layers.DepthwiseConv2D):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_parameters)

        if isinstance(pretrained_model.layers[layer_index], tf.keras.layers.Conv2D):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_parameters)

        if isinstance(pretrained_model.layers[layer_index], tf.keras.layers.Dense):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_parameters)

        if isinstance(pretrained_model.layers[layer_index], tf.keras.layers.BatchNormalization):
            assign_pretrained_batch_norm_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                    reds_layer=reds_model.layers[layer_index],
                                                    training_from_scratch=training_from_scratch,
                                                    trainable=trainable_batch_normalization)

    return reds_model



def convert_ds_cnn_model_to_reds(pretrained_model, train_ds, args, model_settings,
                                 model_filters,
                                 use_bias=True,
                                 trainable_parameters=True,
                                 model_size="s",
                                 training_from_scratch=False,
                                 trainable_batch_normalization=False,
                                 slimmable=False):
    """
        Given a pretrained model retrieve its corresponding reds model (with the same architecture) and assign to it the
        weight and bias of the pretrained model
        @return: reds model initialize with the weight and bias of the pretrained model
    """
    reds_model = get_reds_ds_cnn_architecture(classes=args.classes,
                                              model_size=model_size,
                                              model_filters=model_filters,
                                              subnetworks_number=args.subnets_number, use_bias=use_bias,
                                              model_settings=model_settings, debug=False,
                                              slimmable=slimmable)

    # forward one sample to initialize the model's weights
    for images, _ in train_ds.take(1):
        reds_model.build(input_shape=images.shape)
        reds_model.set_subnetworks_number(subnetworks_number=1)
        reds_model(inputs=images, training=False)
        reds_model.set_subnetworks_number(subnetworks_number=args.subnets_number)

    layer_reds_index = 0
    for layer_index in range(len(reds_model.layers)):

        if isinstance(pretrained_model.layers[layer_index], tf.keras.layers.DepthwiseConv2D):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_reds_index] if slimmable else
                                                   reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_parameters)

        elif isinstance(pretrained_model.layers[layer_index], tf.keras.layers.Conv2D):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_reds_index] if slimmable else
                                                   reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_parameters)

        elif isinstance(pretrained_model.layers[layer_index], tf.keras.layers.Dense):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_reds_index] if slimmable else
                                                   reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_parameters)

        elif isinstance(pretrained_model.layers[layer_index], tf.keras.layers.BatchNormalization):

            assign_pretrained_batch_norm_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                        reds_layer=reds_model.layers[layer_index],
                                                        training_from_scratch=training_from_scratch,
                                                        trainable=trainable_batch_normalization)
    return reds_model


def convert_cnn_model_to_reds(pretrained_model, train_ds, args, model_size_info_convolution, model_settings,
                              model_size_info_dense,
                              use_bias=True,
                              trainable_parameters=True,
                              training_from_scratch=False,
                              trainable_batch_normalization=False):
    """
    Given a pretrained model retrieve its corresponding reds model (with the same architecture) and assign to it the
    weight and bias of the pretrained model
    @return: reds model initialize with the weight and bias of the pretrained model
    """
    reds_model = get_reds_cnn_architecture(architecture_name=args.architecture_name, classes=args.classes,
                                           subnetworks_number=args.subnets_number, use_bias=use_bias,
                                           model_size_info_convolution=model_size_info_convolution,
                                           model_settings=model_settings,
                                           model_size_info_dense=model_size_info_dense, debug=False)

    # forward one sample to initialize the model's weights
    for images, _ in train_ds.take(1):
        reds_model.build(input_shape=images.shape)
        reds_model.set_subnetworks_number(subnetworks_number=1)
        reds_model(inputs=images, training=False)
        reds_model.set_subnetworks_number(subnetworks_number=args.subnets_number)

    for layer_index in range(len(reds_model.layers)):

        if isinstance(pretrained_model.layers[layer_index], tf.keras.layers.Conv2D):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_parameters)

        if isinstance(pretrained_model.layers[layer_index], tf.keras.layers.Dense):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_parameters)

        if isinstance(pretrained_model.layers[layer_index], tf.keras.layers.BatchNormalization):
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_batch_normalization)

    return reds_model


def val_step(x_batch, y_batch, acc, model):
    # Evaluate the model on given a batch of validation data
    y_pred = model(x_batch)
    # batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    return batch_acc

def val_step_reds(x_batch, y_batch, loss, acc, model):
    batch_losses, batch_accuracies = [], []

    # Evaluate the model on given a batch of validation data
    y_pred = model(inputs=x_batch, training=False)  # tf.keras.applications.mobilenet.preprocess_input(

    for subnet_output_index in range(len(y_pred)):
        batch_loss = loss(y_pred[subnet_output_index], y_batch)
        batch_losses.append(batch_loss)
        batch_accuracies.append(acc(y_pred[subnet_output_index], y_batch))

    return batch_losses, batch_accuracies


def standard_step_train(images, labels, loss_fn, model, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

def standard_step_test(images, labels, model, loss_fn, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_fn(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
