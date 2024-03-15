import argparse
import os
import time

import numpy as np

from utils.batch_normalization import Reds_BatchNormalizationBase
from utils.cuda import gpu_selection
from utils.datasets import get_dataset, get_cifar10_train_test
from utils.deterministic import setup_deterministic_computation
import tensorflow as tf

from utils.forward import classification_head_finetuning_ds_cnn, \
    convert_ds_cnn_model_to_reds, accuracy, cross_entropy_loss, train_model, convert_ds_cnn_model_to_vision_reds, \
    accuracy_vision, accuracy_vision_cifar10, train_model_vision
from utils.importance_score import compute_filters_importance_score_feature_extraction_filters, \
    permute_filters_mobilenet, permute_batch_norm_ds_cnn_layers, assign_pretrained_ds_convolution_filters, \
    compute_descending_filters_score_indexes_mobilenet, compute_accumulated_gradients_pointwise_layers, \
    compute_pointwise_importance_score_ds_cnn, compute_accumulated_gradients_ds_cnn, \
    compute_accumulated_gradients_ds_cnn_layers, compute_descending_filters_score_indexes_ds_cnn, permute_filters_ds_cnn
from utils.keyword_spotting import load_pre_trained_kws_model, compute_accuracy_test
from utils.keyword_spotting_data import get_audio_data
from utils.knapsack import knapsack_find_splits_ds_cnn, \
    initialize_nested_knapsack_solver_ds_cnn

from utils.logs import setup_logging, log_print
from utils.vision_models import load_pre_trained_vision_model

if __name__ == '__main__':
    print(tf.__version__)
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--gurobi_home',
                        type=str,
                        default="",
                        help="""\
            Gurobi Linux absolute path.
            """)

    parser.add_argument('--gurobi_license_file',
                        type=str,
                        default=",
                        help="""\
                Gurobi license absolute path.
                """)
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
        default='ds_cnn',
        help='What model architecture to use')

    parser.add_argument('--subnets_number', default=4, type=int, help='number of subnetworks to train')  # 9
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001,0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--training_steps',
        type=str,
        default='20000,20000',
        help='How many training loops to run and learning rate schedule', )

    parser.add_argument('--cuda_device', default=-1, type=int)
    parser.add_argument('--solver_max_iterations', default=3, type=int)
    parser.add_argument('--solver_time_limit', default=100000, type=int)
    parser.add_argument('--epochs', type=int, default=250, help='training epochs')
    parser.add_argument('--model_sizes', default='s', type=str,
                        help='model sizes')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--experimental_runs', default=3, type=int)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print intermediate activations and weights cuttings dimensions')
    parser.add_argument('--last_pointwise_filters', default=60, type=int)
    parser.add_argument('--print', default=False, action='store_true',
                        help='print all the subnetworks accuracies')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='plot the subnetworks finetuning and importance score')
    parser.add_argument('--minibatch_number', default=100, type=int)
    parser.add_argument('--finetune_head_epochs', default=70, type=int, help='number of epochs to train the model')
    parser.add_argument('--finetune_batch_norm_epochs', default=10, type=int,
                        help='number of epochs to train the model')
    parser.add_argument('--save_path', default='{}/result/{}/KWS_Knapsack_alpha_{}_{}epochs_{}batch_{}subnetworks_{}',
                        type=str)
    parser.add_argument('--dataset', default='cifar10',  
                        type=str)
    parser.add_argument('--bottom_up', default=True, action='store_false',
                        help='default run bottom up knapsack, if passed run top down knapsack')

    parser.add_argument(
        '--constraints_percentages',
        type=str,
        default='0.25, 0.5,0.75',  
        help='Constraints percentages', )

    args, _ = parser.parse_known_args()
    setup_logging(args=args,
                  experiment_name="{}_KWS_ARM_{}_{}_DS_CNN_Benchmark".format(args.dataset,
                                                                             "Bottom_up" if args.bottom_up else "Top_down",
                                                                             args.architecture_name))

    setup_deterministic_computation(seed=args.seed)
    gpu_selection(gpu_number=args.cuda_device)

    os.environ[
        'GUROBI_HOME'] = args.gurobi_home
    os.environ['GRB_LICENSE_FILE'] = args.gurobi_license_file

    print("Gurobi settings:")
    print(os.getenv('GUROBI_HOME'))
    print(os.getenv('GRB_LICENSE_FILE'))

    layers_units = {
        's': 64,
        'l': 276
    }

    input_shape = {
        'mnist': (28, 28, 1),
        'fashion_mnist': (28, 28, 1),
        'cifar10': (32, 32, 3)
    }

    constraints_percentages = list(map(float, args.constraints_percentages.split(',')))
    log_print("{} Knapsack".format("Top Down" if not args.bottom_up else "Bottom Up"))

    for model_size in args.model_sizes.split(','):

        log_print(
            "Dataset {} Loading model {} size {} minibatch number {} learning rates: {} last pointwise filters: {}".format(
                args.dataset, args.architecture_name, model_size,
                args.minibatch_number, list(map(float, args.learning_rate.split(','))), args.last_pointwise_filters))

        average_final_subnetworks_accuracy, average_final_subnetworks_loss = [[] for _ in
                                                                              range(args.subnets_number)], [[] for _
                                                                                                            in
                                                                                                            range(
                                                                                                                args.subnets_number)]
        pretrained_model_accuracy = []
        permuted_classification_head_finetuned_accuracy = []
        subnetworks_macs_print = [[] for _ in range(args.subnets_number)]

        for experimental_run in range(args.experimental_runs):

            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[16406, 30468],
                                                                               values=[0.001, 0.00025, 0.00001])

            log_print("Run experimental run number: {}".format(experimental_run))

            pretrained_model = load_pre_trained_vision_model(
                dataset_name=args.dataset,
                model_size=model_size)

            train_data, test_data = get_cifar10_train_test(batch_size=128)

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            start = time.time()
            gradients_accumulation = compute_accumulated_gradients_ds_cnn(model=pretrained_model, train_data=train_data,
                                                                          loss_fn=loss_fn,
                                                                          args=args)

            importance_score_feature_extraction_filters = compute_filters_importance_score_feature_extraction_filters(
                model=pretrained_model,
                gradients_accumulation=gradients_accumulation)

            descending_importance_score_indexes_depthwise_filters, descending_importance_score_scores_depthwise_filters = compute_descending_filters_score_indexes_ds_cnn(
                model=pretrained_model,
                importance_score_filters=importance_score_feature_extraction_filters,
                units_number=layers_units[model_size])

            permuted_convolution_filters, permuted_convolution_bias = permute_filters_ds_cnn(
                model=pretrained_model,
                filters_descending_ranking=descending_importance_score_indexes_depthwise_filters)

            permute_batch_norm_ds_cnn_layers(model=pretrained_model,
                                             permutations_order=descending_importance_score_indexes_depthwise_filters,
                                             trainable_assigned_batch_norm=False,
                                             first_indexes_batch_norm=[1, 4],
                                             trainable_pointwise_batch_norm=True)

            assign_pretrained_ds_convolution_filters(model=pretrained_model,
                                                     permuted_convolutional_filters=permuted_convolution_filters,
                                                     permuted_convolutional_bias=permuted_convolution_bias,
                                                     trainable_assigned_depthwise_convolution=False,
                                                     trainable_assigned_pointwise_convolution=True)

            optimizer_permute_model = tf.keras.optimizers.experimental.Adam() 
            optimizer_permute_model.build(var_list=pretrained_model.trainable_variables)

            pointwise_filters_batch_norm_finetuned_accuracy = classification_head_finetuning_ds_cnn(
                model=pretrained_model,
                optimizer=optimizer_permute_model,
                train_ds=train_data,
                test_ds=test_data, args=args, print_info=False,
                initial_pretrained_test_accuracy=80.0)

            pointwise_layers_gradients = compute_accumulated_gradients_pointwise_layers(model=pretrained_model,
                                                                                        train_data=train_data,
                                                                                        loss_fn=loss_fn,
                                                                                        args=args)

            importance_score_pointwise_filters_kernels = compute_pointwise_importance_score_ds_cnn(
                model=pretrained_model,
                gradients_accumulation_pointwise=pointwise_layers_gradients)

            model_units_importance_scores = []
            model_units_importance_scores.append(descending_importance_score_scores_depthwise_filters.pop(0))

            for layer_index in range(len(importance_score_pointwise_filters_kernels)):
                model_units_importance_scores.append(
                    descending_importance_score_scores_depthwise_filters[layer_index])

            reds_pretrained_model = convert_ds_cnn_model_to_vision_reds(pretrained_model=pretrained_model,
                                                                        train_ds=train_data,
                                                                        args=args,
                                                                        model_size=model_size,
                                                                        model_filters=layers_units[model_size],
                                                                        trainable_parameters=True,
                                                                        trainable_batch_normalization=False,
                                                                        training_from_scratch=False)

            layers_filters_macs, layers_filters_byte = reds_pretrained_model.compute_lookup_table(
                input_shape=input_shape[args.dataset])

            importance_list, macs_list, memory_list, macs_targets, memory_targets = initialize_nested_knapsack_solver_ds_cnn(
                layers_filters_macs=layers_filters_macs,
                descending_importance_score_scores=model_units_importance_scores,
                layers_filters_byte=layers_filters_byte,
                subnetworks_number=args.subnets_number,
                constraints_percentages=constraints_percentages)

            end = time.time()
            print("Time taken to permute the model based on importance score: ", end - start)
            subnetworks_filters_first_convolution, subnetworks_filters_depthwise, subnetworks_filters_pointwise, subnetworks_macs = knapsack_find_splits_ds_cnn(
                args=args,
                layers_filter_macs=layers_filters_macs,
                memory_list=memory_list,
                memory_targets=memory_targets,
                importance_list=importance_list,
                model_size=model_size,
                macs_list=macs_list,
                macs_targets=macs_targets,
                importance_score_pointwise_filters_kernels=importance_score_pointwise_filters_kernels,
                last_pointwise_filters=args.last_pointwise_filters,
                bottom_up=args.bottom_up,
                units_layer_size=layers_units[model_size])

            reds_pretrained_model.set_subnetwork_indexes(
                subnetworks_filters_first_convolution=subnetworks_filters_first_convolution,
                subnetworks_filters_depthwise=subnetworks_filters_depthwise,
                subnetworks_filters_pointwise=subnetworks_filters_pointwise)


            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[16406, 30468],
                                                                               values=[0.001, 0.00025, 0.00001])
            optimizer = tf.optimizers.experimental.Adam(learning_rate=lr_schedule)
            optimizer.build(var_list=reds_pretrained_model.trainable_variables)

            _ = train_model_vision(model=reds_pretrained_model, train_data=train_data, test_data=test_data,
                            val_data=test_data, debug=args.debug,
                            plot=args.plot,
                            acc=accuracy_vision_cifar10,
                            message_initial_accuracies="Initial accuracy REDS model WITH filters permutation WITH finetuned classification head",
                            architecture_name=args.architecture_name + f"_{model_size}",
                            importance_score=True,
                            message=f"REDS finetuning {reds_pretrained_model.get_model_name()} KWS pretrained model WITH filters permutation",
                            optimizer=optimizer, epochs=args.epochs,
                            subnetworks_number=args.subnets_number, args=args, subnetworks_macs=subnetworks_macs)

            reds_pretrained_model.finetune_batch_normalization()
            optimizer_batch = tf.keras.optimizers.experimental.Adam(learning_rate=lr_schedule)
            optimizer_batch.build(var_list=reds_pretrained_model.trainable_variables)

            final_subnetworks_accuracy = train_model_vision(model=reds_pretrained_model,
                                                     plot=args.plot,
                                                     train_data=train_data, test_data=test_data,
                                                     val_data=test_data, debug=args.debug,
                                                    acc=accuracy_vision_cifar10,
                                                     architecture_name=args.architecture_name + f"_{model_size}",
                                                     importance_score=True,
                                                     batch_norm_finetuning=True,
                                                     message=f"REDS WITH filters permutation finetuning {reds_pretrained_model.get_model_name()} Batch Normalization layers",
                                                     optimizer=optimizer_batch,
                                                     message_initial_accuracies="Initial accuracies finetuned REDS model",
                                                     epochs=args.finetune_batch_norm_epochs,
                                                     subnetworks_number=args.subnets_number,
                                                     args=args,
                                                     subnetworks_macs=subnetworks_macs)

            for subnetwork_index in range(args.subnets_number):
                log_print(
                    f"Experimental run: {experimental_run} Subnetwork {subnetworks_macs[subnetwork_index]} MACs test accuracy: {100 * np.array(final_subnetworks_accuracy[subnetwork_index]).mean()}%")
                subnetworks_macs_print.append(subnetworks_macs[subnetwork_index])

            [average_final_subnetworks_accuracy[subnetwork_number].append(
                100 * final_subnetworks_accuracy[subnetwork_number][0])
                for subnetwork_number in range(args.subnets_number)]

        for subnetwork_number in range(args.subnets_number):
            log_print(
                f"dataset: {args.dataset} subnetworks MACS: {np.array(subnetworks_macs_print[subnetwork_number]).mean()} test accuracy mean: {np.array(average_final_subnetworks_accuracy[subnetwork_number]).mean():.4f}% test accuracy std: {np.array(average_final_subnetworks_accuracy[subnetwork_number]).std():.4f}")

        log_print(
            f"dataset: {args.dataset} pretrained model average test accuracy: {np.array(pretrained_model_accuracy).mean():.4f}% std: {np.array(pretrained_model_accuracy).std():.4f}")

