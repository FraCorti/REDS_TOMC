import logging
import time

import numpy as np
import tensorflow as tf

from utils.convolution import get_reds_cnn_architecture, Linear_Adaptive
from utils.ds_convolution import get_reds_ds_cnn_architecture
from utils.ds_convolution_vision_data import get_reds_ds_cnn_vision_architectures
from utils.linear import get_reds_dnn_architecture, Reds_Linear
from utils.logs import log_print


def cross_entropy_loss(y_pred, y):
    # Compute cross entropy loss with a sparse operation
    if y_pred.shape.rank - 1 != y.shape.rank:
        y = tf.squeeze(y, axis=[1])
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(sparse_ce)


def accuracy_converted(y_pred, y):
    class_preds = tf.argmax(tf
                            .nn.softmax(y_pred), axis=1)
    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


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


def accuracy_TinyML(y_pred, y):
    # Compute accuracy after extracting class predictions
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)

    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))


def train_step_reds_depthwise_convolution(x_batch, y_batch, loss, acc, model, optimizer, subnetworks_number):
    alphas = list(0 for _ in range(0, subnetworks_number))

    for weights_percentage_index in range(0, subnetworks_number):
        alpha = pow((1 - (1 - (pow(0.5, weights_percentage_index)))), 0.5)
        alphas[weights_percentage_index] = alpha

    batch_losses, batch_accuracies = [], []

    # compute and accumulate subnetworks gradients
    for subnetworks_number in range(subnetworks_number):

        model.set_subnet_number(subnet_number=subnetworks_number)
        optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.01,
                                                         momentum=0.9)
        optimizer.build(var_list=model.variables)

        with tf.GradientTape(persistent=True) as tape:

            logits = model(inputs=x_batch, training=True)

            batch_loss = loss(logits, y_batch)
            batch_losses.append(batch_loss)
            batch_accuracies.append(acc(logits, y_batch))

            # scale loss and compute subnetwork gradient
            subnetwork_gradient = tape.gradient(batch_loss * float(
                alphas[subnetworks_number] / np.array(alphas).sum()), model.variables)

            for layer_index, (grad, var) in enumerate(zip(subnetwork_gradient, model.variables)):
                optimizer.update_step(grad, var)

            tape.reset()

    return batch_losses, batch_accuracies


def initial_reds_train_step(x_batch, y_batch, loss, acc, model, optimizer, subnetworks_number):
    alphas = list(0 for _ in range(0, subnetworks_number))

    for weights_percentage_index in range(0, subnetworks_number):
        alpha = pow((1 - (1 - (pow(0.5, weights_percentage_index)))), 0.5)
        alphas[weights_percentage_index] = alpha
        alphas[weights_percentage_index] = alpha

    gradients_accumulation = []
    first_gradients = True
    batch_losses, batch_accuracies = [], []

    # compute and accumulate subnetworks gradients
    with tf.GradientTape(persistent=True) as tape:

        logits = model(inputs=x_batch, training=True)

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


def train_step_reds(x_batch, y_batch, loss, acc, model, optimizer, subnetworks_number):
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

        logits = model(inputs=x_batch, training=True)

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

def val_step_reds_vision(x_batch, y_batch, loss_fn, acc, model, subnetworks_losses):
    logits = model(inputs=x_batch, training=False)

    for subnet_output_index in range(len(logits)):
        batch_loss = loss_fn(y_batch, logits[subnet_output_index])
        acc[subnet_output_index](y_batch, logits[subnet_output_index])
        subnetworks_losses[subnet_output_index](batch_loss)

def train_step_reds_vision(x_batch, y_batch, loss_fn, acc, model, optimizer, subnetworks_losses, subnetworks_number, square_error_scaling=False):
    alphas = list(0 for _ in range(0, subnetworks_number))

    # scale loss proportional to subnetwork parameters percentage
    for subnetwork_index in range(0, subnetworks_number):
        subnetworks_parameter_percentage_used = model.get_subnetwork_parameters_percentage(
            subnetwork_index=subnetwork_index)

        alpha = pow((1 - (1 - (subnetworks_parameter_percentage_used))), 0.5)
        alphas[subnetwork_index] = alpha

    gradients_accumulation = []
    first_gradients = True

    # compute and accumulate subnetworks gradients
    with tf.GradientTape(persistent=True) as tape:

        logits = model(inputs=x_batch, training=True)

        for subnet_output_index in range(len(logits)):
            batch_loss = loss_fn(y_batch, logits[subnet_output_index])

            acc[subnet_output_index](y_batch, logits[subnet_output_index])
            subnetworks_losses[subnet_output_index](batch_loss)

            # scale loss and compute subnetwork gradient
            subnetwork_gradients = tape.gradient(batch_loss * float(
                alphas[subnet_output_index] / np.array(alphas).sum()), model.trainable_variables)  # scaling the loss absolute value with pow

            if first_gradients:
                [gradients_accumulation.append(gradient) for gradient in subnetwork_gradients]
                first_gradients = False
            else:
                for gradient_index in range(len(gradients_accumulation)):
                    gradients_accumulation[gradient_index] = tf.math.add(gradients_accumulation[gradient_index],
                                                                         subnetwork_gradients[gradient_index])

    for layer_index, (grad, var) in enumerate(zip(gradients_accumulation, model.trainable_variables)):
        optimizer.update_step(grad, var)


def train_model_no_val(model, train_data, loss, acc, optimizer, epochs, subnetworks_number, debug=False):
    train_losses, train_accs = [], []

    for epoch in range(epochs):
        batch_losses_train, batch_accs_train = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                         range(subnetworks_number)]

        for x_batch, y_batch in train_data:

            batch_loss_train, batch_accuracy_train = train_step_reds(x_batch, y_batch, loss, acc, model, optimizer,
                                                                     subnetworks_number)

            for subnetwork_number in range(len(batch_losses_train)):
                batch_losses_train[subnetwork_number].append(batch_loss_train[subnetwork_number])
                batch_accs_train[subnetwork_number].append(batch_accuracy_train[subnetwork_number])

            if debug:
                break

        if epoch == epochs - 1:

            log_print(f"Last epoch subnetworks accuracies:")
            for subnetwork_number in range(subnetworks_number):
                train_loss, train_acc = tf.reduce_mean(batch_losses_train[subnetwork_number]), tf.reduce_mean(
                    batch_accs_train[subnetwork_number])

                log_print(
                    f"parameters considered: {pow(0.5, subnetwork_number) * 100}% training accuracy: {100 * train_acc:.3f}%")

    return train_losses, train_accs


def train_model_tinyML(model, train_data, val_data, test_data, loss, acc, optimizer, subnetworks_number, epochs):
    val_acc_subnetworks = [[] for _ in range(subnetworks_number)]
    train_acc_subnetworks = [[] for _ in range(subnetworks_number)]
    test_acc_subnetworks = [[] for _ in range(subnetworks_number)]

    for epoch in range(epochs):

        batch_losses_train, batch_accs_train = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                         range(subnetworks_number)]
        batch_losses_val, batch_accs_val = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                     range(subnetworks_number)]

        batch_losses_test, batch_accs_test = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                       range(subnetworks_number)]
        for x_batch, y_batch in train_data:

            # Compute gradients and update the model's parameters
            batch_loss_train, batch_accuracy_train = initial_reds_train_step(x_batch, y_batch, loss, acc,
                                                                             model, optimizer,
                                                                             subnetworks_number)

            for subnetwork_number in range(len(batch_loss_train)):
                batch_losses_train[subnetwork_number].append(batch_loss_train[subnetwork_number])
                batch_accs_train[subnetwork_number].append(batch_accuracy_train[subnetwork_number])

        for x_batch, y_batch in val_data:

            batch_loss_val, batch_accuracy_val = val_step_reds(x_batch, y_batch, loss, acc, model)

            for subnetwork_number in range(len(batch_loss_val)):
                batch_losses_val[subnetwork_number].append(batch_loss_val[subnetwork_number])
                batch_accs_val[subnetwork_number].append(batch_accuracy_val[subnetwork_number])

        for x_batch, y_batch in test_data:

            batch_loss_test, batch_accuracy_test = val_step_reds(x_batch, y_batch, loss, acc, model)

            for subnetwork_number in range(len(batch_loss_test)):
                batch_losses_test[subnetwork_number].append(batch_loss_test[subnetwork_number])
                batch_accs_test[subnetwork_number].append(batch_accuracy_test[subnetwork_number])

        for subnetwork_number in range(subnetworks_number):

            train_loss, train_acc = tf.reduce_mean(batch_losses_train[subnetwork_number]), tf.reduce_mean(
                batch_accs_train[subnetwork_number])

            val_loss, val_acc = tf.reduce_mean(batch_losses_val[subnetwork_number]), tf.reduce_mean(
                batch_accs_val[subnetwork_number])

            test_loss, test_acc = tf.reduce_mean(batch_losses_test[subnetwork_number]), tf.reduce_mean(
                batch_accs_test[subnetwork_number])

            train_acc_subnetworks[subnetwork_number].append(train_acc)
            val_acc_subnetworks[subnetwork_number].append(val_acc)
            test_acc_subnetworks[subnetwork_number].append(test_acc)

            log_print(
                f"epoch: {epoch} subnetwork number: {subnetwork_number} training accuracy: {100 * train_acc:.4f}% validation accuracy: {100 * val_acc:.4f}% test accuracy: {100 * test_acc:.4f}% training loss: {train_loss:.4f} validation loss: {val_loss:.4f} test loss: {test_loss:.4f}")

            if epoch == epochs - 1:
                test_acc_subnetworks[subnetwork_number] = test_acc_subnetworks[subnetwork_number][-1]

    return test_acc_subnetworks


def train_model(model, train_data, val_data, test_data, loss, acc, optimizer, epochs,
                subnetworks_number, subnetworks_macs, args, message_initial_accuracies, message="", full_training=False,
                importance_score=True, architecture_name="", plot=True, batch_norm_finetuning=False,
                debug=False):
    val_acc_subnetworks = [[] for _ in range(subnetworks_number)]
    train_acc_subnetworks = [[] for _ in range(subnetworks_number)]
    test_acc_subnetworks = [[] for _ in range(subnetworks_number)]

    log_print(message)

    for epoch in range(epochs):

        batch_losses_train, batch_accs_train = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                         range(subnetworks_number)]
        batch_losses_val, batch_accs_val = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                     range(subnetworks_number)]

        batch_losses_test, batch_accs_test = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                       range(subnetworks_number)]
        try:
            for x_batch, y_batch in train_data:
                # y_batch = tf.cast(y_batch, dtype=tf.int64)
                # Compute gradients and update the model's parameters
                batch_loss_train, batch_accuracy_train = train_step_reds(x_batch, y_batch, loss, acc,
                                                                         model, optimizer,
                                                                         subnetworks_number)

                for subnetwork_number in range(len(batch_loss_train)):
                    batch_losses_train[subnetwork_number].append(batch_loss_train[subnetwork_number])
                    batch_accs_train[subnetwork_number].append(batch_accuracy_train[subnetwork_number])
        except Exception as e:
            print(e)

        try:
            for x_batch, y_batch in val_data:
                # y_batch = tf.cast(y_batch, dtype=tf.int64)
                batch_loss_val, batch_accuracy_val = val_step_reds(x_batch, y_batch, loss, acc, model)

                for subnetwork_number in range(len(batch_loss_val)):
                    batch_losses_val[subnetwork_number].append(batch_loss_val[subnetwork_number])
                    batch_accs_val[subnetwork_number].append(batch_accuracy_val[subnetwork_number])
        except Exception as e:
            print(e)

        try:
            for x_batch, y_batch in test_data:
                # y_batch = tf.cast(y_batch, dtype=tf.int64)
                batch_loss_test, batch_accuracy_test = val_step_reds(x_batch, y_batch, loss, acc, model)

                for subnetwork_number in range(len(batch_loss_test)):
                    batch_losses_test[subnetwork_number].append(batch_loss_test[subnetwork_number])
                    batch_accs_test[subnetwork_number].append(batch_accuracy_test[subnetwork_number])
        except Exception as e:
            print(e)

        for subnetwork_number in range(subnetworks_number):
            train_loss, train_acc = tf.reduce_mean(batch_losses_train[subnetwork_number]), tf.reduce_mean(
                batch_accs_train[subnetwork_number])

            val_loss, val_acc = tf.reduce_mean(batch_losses_val[subnetwork_number]), tf.reduce_mean(
                batch_accs_val[subnetwork_number])

            test_loss, test_acc = tf.reduce_mean(batch_losses_test[subnetwork_number]), tf.reduce_mean(
                batch_accs_test[subnetwork_number])

            train_acc_subnetworks[subnetwork_number].append(train_acc)
            val_acc_subnetworks[subnetwork_number].append(val_acc)
            test_acc_subnetworks[subnetwork_number].append(test_acc)

            log_print(
                f"subnetworks MACS: {subnetworks_macs[subnetwork_number]} training accuracy: {100 * train_acc:.3f}% validation accuracy: {100 * val_acc:.3f}% test accuracy: {100 * test_acc:.3f}% training loss: {train_loss:.3f} validation loss: {val_loss:.3f} test loss: {test_loss:.3f}")

    return [[test_acc_subnetworks[subnetwork_number][-1]] for subnetwork_number in range(subnetworks_number)]


def train_model_vision(model, train_data, val_data, test_data, acc, optimizer, epochs,
                       subnetworks_number, subnetworks_macs, args, message_initial_accuracies, message="",
                       square_error_scaling=False,
                       full_training=False,
                       importance_score=True, architecture_name="", plot=True, batch_norm_finetuning=False,
                       debug=False):
    train_acc_subnetworks = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_{}'.format(subnetwork_number)) for
        subnetwork_number in range(subnetworks_number)]

    test_acc_subnetworks = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_{}'.format(subnetwork_number)) for
        subnetwork_number in range(subnetworks_number)]

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss_subnetworks = [tf.keras.metrics.Mean(name='train_loss_{}'.format(subnetwork_number)) for
                              subnetwork_number in range(subnetworks_number)]
    test_loss_subnetworks = [tf.keras.metrics.Mean(name='test_loss_{}'.format(subnetwork_number)) for subnetwork_number
                             in range(subnetworks_number)]

    log_print(message)

    for epoch in range(epochs):

        [train_acc.reset_states() for train_acc in train_acc_subnetworks]
        [test_acc.reset_states() for test_acc in test_acc_subnetworks]
        [train_loss.reset_states() for train_loss in train_loss_subnetworks]
        [test_loss.reset_states() for test_loss in test_loss_subnetworks]

        try:
            for x_batch, y_batch in train_data:
                train_step_reds_vision(x_batch=x_batch, y_batch=y_batch, loss_fn=loss_fn, acc=train_acc_subnetworks,
                                       model=model, optimizer=optimizer,
                                       subnetworks_number=subnetworks_number, subnetworks_losses=train_loss_subnetworks)
        except Exception as e:
            print(e)

        try:
            for x_batch, y_batch in test_data:
                val_step_reds_vision(x_batch=x_batch, y_batch=y_batch, loss_fn=loss_fn, acc=test_acc_subnetworks,
                                     model=model,
                                     subnetworks_losses=test_loss_subnetworks)
        except Exception as e:
            print(e)

        for subnetwork_number in range(subnetworks_number):
            log_print(
                f"subnetworks MACS: {subnetworks_macs[subnetwork_number]} training accuracy: {train_acc_subnetworks[subnetwork_number].result() * 100:.3f}% test accuracy: {100 * test_acc_subnetworks[subnetwork_number].result():.3f}%  training loss: {train_loss_subnetworks[subnetwork_number].result()} test loss: {test_loss_subnetworks[subnetwork_number].result()} ")

    return [test_acc_subnetworks[subnetwork_number].result() for subnetwork_number in range(subnetworks_number)]


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

        try:
            for images, labels in train_ds:
                standard_step_train(model=model, images=images, labels=labels, loss_fn=loss_fn, optimizer=optimizer,
                                    train_loss=train_loss, train_accuracy=train_accuracy)

            for test_images, test_labels in test_ds:
                standard_step_test(model=model, images=test_images, labels=test_labels, loss_fn=loss_fn,
                                   test_loss=test_loss, test_accuracy=test_accuracy)

            for test_images, test_labels in test_ds:
                standard_step_test(model=model, images=test_images, labels=test_labels, loss_fn=loss_fn,
                                   test_loss=test_loss, test_accuracy=test_accuracy)
        except Exception as e:
            print(e)

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


def classification_head_finetuning_ds_cnn(model, optimizer, train_ds, test_ds, initial_pretrained_test_accuracy, args, print_info=True):
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

        try:
            start_time = time.time()
            for images, labels in train_ds:
                standard_step_train(model=model, images=images,
                                    labels=labels, loss_fn=loss_fn, optimizer=optimizer,
                                    train_loss=train_loss, train_accuracy=train_accuracy)
            print("--- head finetuning epoch takes: %s seconds ---" % (time.time() - start_time))
            for test_images, test_labels in test_ds:
                standard_step_test(model=model, images=test_images,
                                   labels=test_labels, loss_fn=loss_fn,
                                   test_loss=test_loss, test_accuracy=test_accuracy)
        except Exception as e:
            print(e)

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
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
                                                   reds_layer=reds_model.layers[layer_index],
                                                   training_from_scratch=training_from_scratch,
                                                   trainable=trainable_batch_normalization)

    return reds_model


def convert_ds_cnn_model_to_reds(pretrained_model, train_ds, args, model_settings,
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
    reds_model = get_reds_ds_cnn_architecture(classes=args.classes,
                                              model_size=model_size,
                                              model_filters=model_filters,
                                              subnetworks_number=args.subnets_number, use_bias=use_bias,
                                              model_settings=model_settings, debug=False)

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
            assign_pretrained_trainable_parameters(pretrained_layer=pretrained_model.layers[layer_index],
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


def reds_initial_accuracies(train_data, val_data, loss, acc, model, subnetworks_number, subnetworks_macs,
                            test_data=None, message=""):
    val_acc_subnetworks = [[] for _ in range(subnetworks_number)]
    train_acc_subnetworks = [[] for _ in range(subnetworks_number)]
    test_acc_subnetworks = [[] for _ in range(subnetworks_number)]

    batch_losses_train, batch_accs_train = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                     range(subnetworks_number)]
    batch_losses_val, batch_accs_val = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                 range(subnetworks_number)]

    batch_losses_test, batch_accs_test = [[] for _ in range(subnetworks_number)], [[] for _ in
                                                                                   range(subnetworks_number)]
    for x_batch, y_batch in train_data:

        batch_loss_train, batch_accuracy_train = val_step_reds(x_batch, y_batch, loss, acc,
                                                               model)

        for subnetwork_number in range(len(batch_losses_train)):
            batch_losses_train[subnetwork_number].append(batch_loss_train[subnetwork_number])
            batch_accs_train[subnetwork_number].append(batch_accuracy_train[subnetwork_number])

    for x_batch, y_batch in val_data:

        batch_loss_val, batch_accuracy_val = val_step_reds(x_batch, y_batch, loss, acc, model)

        for subnetwork_number in range(len(batch_loss_val)):
            batch_losses_val[subnetwork_number].append(batch_loss_val[subnetwork_number])
            batch_accs_val[subnetwork_number].append(batch_accuracy_val[subnetwork_number])

    for x_batch, y_batch in test_data:

        batch_loss_test, batch_accuracy_test = val_step_reds(x_batch, y_batch, loss, acc, model)

        for subnetwork_number in range(len(batch_loss_test)):
            batch_losses_test[subnetwork_number].append(batch_loss_test[subnetwork_number])
            batch_accs_test[subnetwork_number].append(batch_accuracy_test[subnetwork_number])

    print(message)
    for subnetwork_number in range(subnetworks_number):
        train_loss, train_acc = tf.reduce_mean(batch_losses_train[subnetwork_number]), tf.reduce_mean(
            batch_accs_train[subnetwork_number])

        val_loss, val_acc = tf.reduce_mean(batch_losses_val[subnetwork_number]), tf.reduce_mean(
            batch_accs_val[subnetwork_number])

        test_loss, test_acc = tf.reduce_mean(batch_losses_test[subnetwork_number]), tf.reduce_mean(
            batch_accs_test[subnetwork_number])

        val_acc_subnetworks[subnetwork_number].append(val_acc)
        train_acc_subnetworks[subnetwork_number].append(train_acc)
        test_acc_subnetworks[subnetwork_number].append(test_acc)

        log_print(
            f"Initial accuracies: subnetworks MACS: {subnetworks_macs[subnetwork_number]} training accuracy: {100 * train_acc:.3f}% validation accuracy: {100 * val_acc:.3f}% test accuracy: {100 * test_acc:.3f} training loss: {train_loss:.3f} validation loss: {val_loss:.3f} test loss: {test_loss:.3f}")

    return train_acc_subnetworks, val_acc_subnetworks, test_acc_subnetworks


def val_step(x_batch, y_batch, acc, model):
    # Evaluate the model on given a batch of validation data
    y_pred = model(x_batch)
    # batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    return batch_acc


def single_batch_forward_model_print_intermediates(model, args):
    model.set_print_hidden_feature(print=True)
    _, val_data = get_dataset_tf(dataset_name=args.dataset_name, batch_size=1)

    for x_batch, y_batch in val_data:
        _ = model(x_batch)
        break

    model.set_print_hidden_feature(print=False)


def single_batch_forward_model_print_intermediates_custom(model, args, val_data):
    model.set_debug(debug=True)

    for x_batch, y_batch in val_data:
        _ = model(x_batch)
        break

    model.set_debug(debug=False)


def measure_forward_time_subnetworks(model, dataset_name):
    model.set_measuring_forward_time(measure_time=True)

    train_loader_tf, _, _, _ = get_custom_dataset(
        dataset_name=dataset_name,
        batch_size=1,
        all_dataset=True,
        store_dataset=False, test_size=0.0)

    for x_batch, y_batch in train_loader_tf:
        _ = model(x_batch)
        break

    model.set_measuring_forward_time(measure_time=False)


def compute_validation_accuracy(model, val_data, acc, dataset_type="validation", message="REDS convert model"):
    batch_losses, batch_accs = [], []

    for x_batch, y_batch in val_data:
        batch_acc = val_step(x_batch, y_batch, acc, model)
        batch_accs.append(batch_acc)

    val_loss, val_acc = tf.reduce_mean(batch_losses), tf.reduce_mean(batch_accs)
    log_print(f"{message} {dataset_type} accuracy: {100 * val_acc:.3f}%")



def val_step_reds(x_batch, y_batch, loss, acc, model):
    batch_losses, batch_accuracies = [], []

    # Evaluate the model on given a batch of validation data
    y_pred = model(inputs=x_batch, training=False)  

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
