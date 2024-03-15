import argparse
import os
from collections import namedtuple

import numpy as np
import tensorflow as tf

from utils.cuda import gpu_selection
from utils.deterministic import setup_deterministic_computation
from utils.forward import classification_head_finetuning_vww, train_model_visual_wake_words, \
    convert_mobilenetv1_to_reds_leaky
from utils.importance_score import compute_accumulated_gradients_mobilenetv1, \
    compute_filters_importance_score_feature_extraction_filters, \
    permute_filters_mobilenet, assign_pretrained_ds_convolution_filters, \
    compute_descending_filters_score_indexes_mobilenet, compute_accumulated_gradients_pointwise_layers, \
    compute_pointwise_importance_score_ds_cnn, permute_batch_norm_mobilenetv1_layers
from utils.knapsack import \
    knapsack_find_splits_mobilenetv1, \
    initialize_nested_knapsack_solver_visual_wake_words
from utils.logs import setup_logging, log_print
from utils.mobilenets import load_mobilenet_v1_visual_wake_words_leaky_relu
from utils.visual_wake_words import load_visual_wake_words_dataset


def main(args):
    input_shapes = list(map(int, args.input_shapes.split(',')))
    for input_image_shape in input_shapes:

        log_print("{} Knapsack".format("Top Down" if not args.bottom_up else "Bottom Up"))

        log_print("Loading MobileNetV1 minibatch number {} last pointwise filters: {} input image shape: {}".format(
            args.minibatch_number, args.last_pointwise_filters, input_image_shape))

        average_final_subnetworks_accuracy, average_final_subnetworks_loss = [[] for _ in
                                                                              range(args.subnets_number)], [[] for _
                                                                                                            in
                                                                                                            range(
                                                                                                                args.subnets_number)]
        for experimental_run in range(args.experimental_runs):

            pretrained_model = load_mobilenet_v1_visual_wake_words_leaky_relu(args=args,
                                                                              input_image_size=input_image_shape)

            ImageShape = namedtuple('ImageShape', 'height width channels')
            input_shape = ImageShape(height=input_image_shape, width=input_image_shape,
                                     channels=args.input_shape_channels)
            train_data = load_visual_wake_words_dataset(input_shape, split="train", batch_size=args.batch_size)
            val_data = load_visual_wake_words_dataset(input_shape, split="val", batch_size=args.batch_size)

            if not args.debug:

                loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                gradients_accumulation = compute_accumulated_gradients_mobilenetv1(model=pretrained_model,
                                                                                   train_data=train_data,
                                                                                   loss_fn=loss_fn, debug=args.debug,
                                                                                   args=args)
                importance_score_feature_extraction_filters = compute_filters_importance_score_feature_extraction_filters(
                    model=pretrained_model,
                    gradients_accumulation=gradients_accumulation)

                descending_importance_score_indexes_depthwise_filters, descending_importance_score_scores_depthwise_filters = compute_descending_filters_score_indexes_mobilenet(
                    model=pretrained_model,
                    importance_score_filters=importance_score_feature_extraction_filters)

                permuted_convolution_filters = permute_filters_mobilenet(
                    model=pretrained_model,
                    filters_descending_ranking=descending_importance_score_indexes_depthwise_filters)

                permute_batch_norm_mobilenetv1_layers(model=pretrained_model,
                                                      permutations_order=descending_importance_score_indexes_depthwise_filters,
                                                      trainable_assigned_batch_norm=False,
                                                      trainable_pointwise_batch_norm=True)

                assign_pretrained_ds_convolution_filters(model=pretrained_model,
                                                         permuted_convolutional_filters=permuted_convolution_filters,
                                                         trainable_assigned_depthwise_convolution=False,
                                                         trainable_assigned_pointwise_convolution=True)

                lr_schedule_permutation_head_finetuning = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-3,
                    decay_steps=100000,
                    decay_rate=0.98,
                    staircase=True)

                optimizer_permute_model = tf.keras.optimizers.experimental.Adam(
                    learning_rate=lr_schedule_permutation_head_finetuning)
                optimizer_permute_model.build(var_list=pretrained_model.trainable_variables)

                _ = classification_head_finetuning_vww(
                    model=pretrained_model,
                    epochs=args.finetune_head_epochs,
                    optimizer=optimizer_permute_model,
                    train_ds=train_data,
                    test_ds=val_data, args=args)

                pointwise_layers_gradients = compute_accumulated_gradients_pointwise_layers(model=pretrained_model,
                                                                                            train_data=train_data,
                                                                                            loss_fn=loss_fn,
                                                                                            debug=args.debug,
                                                                                            args=args)

                importance_score_pointwise_filters_kernels = compute_pointwise_importance_score_ds_cnn(
                    model=pretrained_model,
                    gradients_accumulation_pointwise=pointwise_layers_gradients)

                model_units_importance_scores = []
                model_units_importance_scores.append(descending_importance_score_scores_depthwise_filters.pop(0))

                for layer_index in range(len(descending_importance_score_scores_depthwise_filters)):
                    model_units_importance_scores.append(
                        descending_importance_score_scores_depthwise_filters[layer_index])

            reds_pretrained_model = convert_mobilenetv1_to_reds_leaky(pretrained_model=pretrained_model,
                                                                      trainable_parameters=True,
                                                                      trainable_batch_normalization=False,
                                                                      args=args,
                                                                      input_shape=(
                                                                          1, input_image_shape, input_image_shape, 3))
            if not args.debug:
                layers_filters_macs, layers_filters_byte, layer_filters_activation_maps_byte = reds_pretrained_model.compute_inference_estimations(
                    input_shape=(1, input_image_shape, input_image_shape, 3))
                importance_list, macs_list, memory_list, layer_filters_activation_maps_byte, macs_targets, memory_targets, peak_memory_usage_targets = initialize_nested_knapsack_solver_visual_wake_words(
                    layers_filters_macs=layers_filters_macs,
                    layer_filters_activation_maps_byte=layer_filters_activation_maps_byte,
                    descending_importance_score_scores=model_units_importance_scores,
                    layers_filters_byte=layers_filters_byte,
                    peak_memory_usage_target_byte=args.peak_memory_usage_target_byte)

                log_print(
                    f"Knapsack formulation, subnetworks number {args.subnets_number} peak memory usage constraint {args.peak_memory_constraint}")

                subnetworks_filters_first_convolution, subnetworks_filters_depthwise, subnetworks_filters_pointwise, subnetworks_macs = knapsack_find_splits_mobilenetv1(
                    args=args,
                    layers_filter_macs=layers_filters_macs,
                    peak_memory_usage_targets=peak_memory_usage_targets,
                    peak_memory_usage_list=layer_filters_activation_maps_byte,
                    memory_list=memory_list,
                    memory_targets=memory_targets,
                    importance_score_feature_extraction_filters=importance_list,
                    macs_list=macs_list,
                    macs_targets=macs_targets,
                    importance_score_pointwise_filters_kernels=importance_score_pointwise_filters_kernels,
                    last_pointwise_filters=args.last_pointwise_filters,
                    peak_memory_constraint=args.peak_memory_constraint,
                    bottom_up=True)

                print("---- Subnetworks macs: {} ----".format(subnetworks_macs))

            print("subnetworks filters first convolution: {}".format(subnetworks_filters_first_convolution))
            print("subnetworks filters depthwise: {}".format(subnetworks_filters_depthwise))
            print("subnetworks filters pointwise: {}".format(subnetworks_filters_pointwise))

        if args.debug:
            subnetworks_filters_first_convolution = [[13]]
            subnetworks_filters_depthwise = [[[13, 20, 72, 93, 134, 246, 294, 457, 394, 392, 361, 511, 598]]]
            subnetworks_filters_pointwise = [[[20, 72, 93, 134, 246, 294, 457, 394, 392, 361, 511, 598, 1021]]]

        reds_pretrained_model.set_subnetwork_indexes(
            subnetworks_filters_first_convolution=subnetworks_filters_first_convolution,
            subnetworks_filters_depthwise=subnetworks_filters_depthwise,
            subnetworks_filters_pointwise=subnetworks_filters_pointwise)

        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[16406, 30468],
                                                                           values=[0.001, 0.00025, 0.00001])

        optimizer_no_batch = tf.keras.optimizers.experimental.SGD(learning_rate=lr_schedule)
        optimizer_no_batch.build(var_list=reds_pretrained_model.trainable_variables)

        _ = train_model_visual_wake_words(model=reds_pretrained_model, train_data=train_data, test_data=val_data,
                                          message=f"REDS finetuning {reds_pretrained_model.get_model_name()} pretrained model with filters permutation",
                                          optimizer=optimizer_no_batch, epochs=args.epochs,
                                          subnetworks_number=args.subnets_number,
                                          subnetworks_macs=subnetworks_macs if not args.debug or not args.skip_knapsack else [
                                              100, 50],
                                          correction_factor=6,
                                          args=args)

        reds_pretrained_model.finetune_batch_normalization()
        optimizer_batch = tf.keras.optimizers.experimental.SGD(learning_rate=lr_schedule)
        optimizer_batch.build(var_list=reds_pretrained_model.trainable_variables)

        final_subnetworks_accuracy = train_model_visual_wake_words(model=reds_pretrained_model,
                                                                   train_data=train_data, test_data=val_data,
                                                                   message=f"REDS filters permutation finetuning {reds_pretrained_model.get_model_name()} Batch Normalization layers",
                                                                   optimizer=optimizer_batch,
                                                                   epochs=args.finetune_batch_norm_epochs,
                                                                   subnetworks_number=args.subnets_number,
                                                                   subnetworks_macs=subnetworks_macs if not args.debug else [
                                                                       100, 50],
                                                                   correction_factor=6,
                                                                   args=args)

        for subnetwork_index in range(args.subnets_number):
            log_print(
                f"Subnetwork {subnetworks_macs[subnetwork_index]} MACs test accuracy: {final_subnetworks_accuracy[subnetwork_index]}%")

        [average_final_subnetworks_accuracy[subnetwork_number].append(
            final_subnetworks_accuracy[subnetwork_number])
            for subnetwork_number in range(args.subnets_number)]

    for subnetwork_number in range(args.subnets_number):
        log_print(
            f"subnetworks MACS: {subnetworks_macs[subnetwork_number]} test accuracy mean: {np.array(average_final_subnetworks_accuracy[subnetwork_number]).mean():.4f}% test accuracy std: {np.array(average_final_subnetworks_accuracy[subnetwork_number]).std():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gurobi_home',
                        type=str,
                        default='',
                        help="""\
                Gurobi Linux absolute path.
                """)

    parser.add_argument('--gurobi_license_file',
                        type=str,
                        default='',
                        help="""\
                    Gurobi license absolute path.
                    """)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='')

    parser.add_argument('--subnets_number', default=2, type=int, help='number of su bnetworks to train')
    parser.add_argument('--cuda_device', default=1, type=int)
    parser.add_argument('--solver_max_iterations', default=3, type=int)
    parser.add_argument('--solver_time_limit', default=10000, type=int)
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--model_sizes', default='l', type=str,
                        help='model sizes')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--experimental_runs', default=1, type=int)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print intermediate activations and weights cuttings dimensions')
    parser.add_argument('--skip_knapsack', default=False, action='store_true',
                        help='')
    parser.add_argument('--last_pointwise_filters', default=1, type=int)
    parser.add_argument('--input_shape_height', default=96, type=int)
    parser.add_argument('--input_shape_width', default=96, type=int)
    parser.add_argument('--input_shape_channels', default=3, type=int)
    parser.add_argument('--peak_memory_constraint', default=True, action='store_false',
                        help='constraint the peak memory usage of the subnetworks')
    parser.add_argument('--print', default=False, action='store_true',
                        help='print all the subnetworks accuracies')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='plot the subnetworks finetuning and importance score')
    parser.add_argument('--minibatch_number', default=100, type=int)
    parser.add_argument('--finetune_head_epochs', default=5, type=int, help='number of epochs to train')
    parser.add_argument('--peak_memory_usage_target_byte', default=200000, type=int)
    parser.add_argument('--finetune_batch_norm_epochs', default=10, type=int,
                        help='number of epochs to train the model')
    parser.add_argument('--save_path', default='{}/result/{}/KWS_Knapsack_alpha_{}_{}epochs_{}batch_{}subnetworks_{}',
                        type=str)
    parser.add_argument('--bottom_up', default=True, action='store_false',
                        help='default run bottom up knapsack, if passed run top down knapsack')
    parser.add_argument(
        '--input_shapes',
        type=str,
        default='96, 224',
        help='Constraints percentages', )

    args, _ = parser.parse_known_args()
    setup_logging(args=args,
                  experiment_name="MobileNetV1_{}_Leaky_Relu".format(
                      "Bottom_up" if args.bottom_up else "Top_down"))
    setup_deterministic_computation(seed=args.seed)
    gpu_selection(gpu_number=args.cuda_device)

    os.environ[
        'GUROBI_HOME'] = args.gurobi_home
    os.environ['GRB_LICENSE_FILE'] = args.gurobi_license_file

    print("Gurobi settings:")
    print(os.getenv('GUROBI_HOME'))
    print(os.getenv('GRB_LICENSE_FILE'))

    main(args=args)
