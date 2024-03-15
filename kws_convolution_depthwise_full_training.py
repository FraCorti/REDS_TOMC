import os

import numpy as np
import tensorflow as tf

import argparse

from utils.cuda import gpu_selection
from utils.deterministic import setup_deterministic_computation
from utils.keyword_spotting import prepare_model_settings, get_model_size_info_ds_cnn
from utils.keyword_spotting_data import prepare_words_list, get_audio_data
from utils.keyword_spotting_models import create_ds_cnn_model
from utils.logs import log_print, setup_logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--subnets_number', default=1, type=int, help='number of subnetworks to train')
    parser.add_argument(
        '--data_url',
        type=str,
        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='{}/datasets/speech_dataset/'.format(os.getcwd()),
        help="""\
                Where to download the speech training data to.
                """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
                How loud the background noise should be, between 0 and 1.
                """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
                How many of the training samples have background noise mixed in.
                """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
                How much of the training data should be silence.
                """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
                How much of the training data should be unknown words.
                """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
                Range to randomly shift the training audio by in time.
                """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs')
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs')
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=40,
        help='How long each spectrogram timeslice is')
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=20,
        help='How long each spectrogram timeslice is')
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=10,
        help='How many bins to use for the MFCC fingerprint')
    parser.add_argument(
        '--classes',
        type=int,
        default=12,
        help='How many classes the model needs to predict')
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once')
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)')
    parser.add_argument(
        '--architecture_name',
        type=str,
        default='cnn',
        help='What model architecture to use')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='speech_dataset')

    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001,0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--training_steps',
        type=str,
        default='20000,20000',
        help='How many training loops to run', )

    parser.add_argument('--experimental_runs', default=1, type=int)
    parser.add_argument('--epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cuda_device', default=-1, type=int)
    parser.add_argument('--model_sizes', default='s', type=str,
                        help='model sizes')

    parser.add_argument(
        '--configurations_indexes',
        type=str,
        default='0,1,2,3',
        help='Subnetworks configurations to train from scratch', )

    args = parser.parse_args()

    setup_deterministic_computation(seed=args.seed)
    gpu_selection(gpu_number=args.cuda_device)

    setup_logging(args=args,
                  experiment_name="Training subnetworks configuration from scratch")

    filters_used = {
        's': [[63, 63, 63, 63, 63], [30, 62, 63, 41, 41], [20, 29, 47, 40, 62], [8, 21, 47, 11, 62]],
        'l': [[275, 275, 275, 275, 275, 275], [89, 242, 264, 275, 68, 275], [68, 212, 196, 275, 33, 275],
              [43, 138, 113, 275, 16, 273]]
    }

    configurations_indexes_str = args.configurations_indexes.split(',')
    configurations_indexes = list([])

    for it in configurations_indexes_str:
        configurations_indexes.append(int(it))

    subnetwork_macs_percentage = ["100%", "75%", "50%", "25%"]

    for model_size in args.model_sizes.split(','):

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        training_steps_list = list(map(int, args.training_steps.split(',')))
        learning_rates_list = list(map(float, args.learning_rate.split(',')))
        lr_boundary_list = training_steps_list[:-1]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundary_list,
                                                                           values=learning_rates_list)

        model_settings = prepare_model_settings(len(prepare_words_list(args.wanted_words.split(','))),
                                                args.sample_rate, args.clip_duration_ms, args.window_size_ms,
                                                args.window_stride_ms, args.dct_coefficient_count)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for subnetwork_configuration in configurations_indexes:

            final_test_set_accuracy = []

            for experimental_run in range(args.experimental_runs):

                model_size_info = get_model_size_info_ds_cnn(model_size=model_size)
                layer_index = 0
                for filters_index in range(1, len(model_size_info), 5):
                    model_size_info[filters_index] = filters_used[model_size][subnetwork_configuration][layer_index] + 1
                    layer_index += 1

                model = create_ds_cnn_model(model_settings=model_settings, model_size_info=model_size_info)

                print(f"subnetwork configuration: {filters_used[model_size][subnetwork_configuration]}")

                # Save the TFLite model to a file

                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                tflite_model = converter.convert()

                # Save the TFLite model to a file
                print("Storing model: {}".format(
                    '{}/models/knapsack_solver/kws_ds_cnn_size{}_{}macs.tflite'.format(os.getcwd(),
                                                                                       str(model_size).upper(),
                                                                                       subnetwork_macs_percentage[
                                                                                           subnetwork_configuration])))

                with open('{}/models/knapsack_solver/kws_ds_cnn_size{}_{}macs.tflite'.format(os.getcwd(),
                                                                                             str(model_size).upper(),
                                                                                             subnetwork_macs_percentage[
                                                                                                 subnetwork_configuration]),
                          'wb') as f:
                    f.write(tflite_model)

                train_data, val_data, test_data = get_audio_data(args=args, model_settings=model_settings)

                optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=lr_schedule)
                optimizer.build(var_list=model.trainable_variables)


                @tf.function
                def step_test(images, labels):
                    # training=False is only needed if there are layers with different
                    # behavior during training versus inference (e.g. Dropout).
                    predictions = model(images, training=False)
                    t_loss = loss_fn(labels, predictions)

                    test_loss(t_loss)
                    test_accuracy(labels, predictions)


                @tf.function
                def step_train(images, labels):
                    with tf.GradientTape() as tape:
                        # training=True is only needed if there are layers with different
                        # behavior during training versus inference (e.g. Dropout).
                        predictions = model(images, training=True)
                        loss = loss_fn(labels, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    train_loss(loss)
                    train_accuracy(labels, predictions)


                # Prepare the metrics.
                train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
                val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

                for epoch in range(args.epochs):

                    train_loss.reset_states()
                    train_accuracy.reset_states()
                    test_loss.reset_states()
                    test_accuracy.reset_states()

                    for images, labels in train_data:
                        step_train(images, labels)

                    for test_images, test_labels in test_data:
                        step_test(test_images, test_labels)

                final_test_set_accuracy.append(test_accuracy.result() * 100)

            log_print(
                f"model size: {model_size} subnetwork configuration: {filters_used[model_size][subnetwork_configuration]} test accuracy mean: {np.array(final_test_set_accuracy).mean():.4f}% std: {np.array(final_test_set_accuracy).std():.4f}")
