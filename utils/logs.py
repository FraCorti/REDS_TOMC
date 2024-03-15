import os
import sys
from datetime import datetime, date
import logging
import tensorflow as tf


def setup_basic_logging(experiment_name,
                        save_path='{}/result/{}/{}_day:{}_time:{}.log'):
    now = datetime.now()
    today = date.today()
    day = today.strftime("%d_%m_%Y")
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename=save_path.format(
            os.getcwd(), 'logs', experiment_name,
            day, time),
        level=logging.INFO)


def setup_logging_benchmark(experiment_name, save_path='{}/result/{}/{}_day:{}_time:{}.log'):
    """
        Setup logging file to save experiments results
        @param args:
        @param model_architecture:
        @param subnetworks_number:
        @param model_depth:
        @param save_path:
        @return:
        """
    now = datetime.now()
    today = date.today()
    day = today.strftime("%d_%m_%Y")
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename=save_path.format(
            os.getcwd(), 'edge_impulse_logs', experiment_name,
            day, time),
        level=logging.INFO)


def setup_logging(args, experiment_name,
                  save_path='{}/result/{}/{}_day:{}_time:{}.log'):
    """
    Setup logging file to save experiments results
    @param args:
    @param model_architecture:
    @param subnetworks_number:
    @param model_depth:
    @param save_path:
    @return:
    """
    now = datetime.now()
    today = date.today()
    day = today.strftime("%d_%m_%Y")
    time = now.strftime("%H_%M_%S")
    logging.basicConfig(
        filename=save_path.format(
            os.getcwd(), 'logs', experiment_name,
            day, time),
        level=logging.INFO)

    log_print("------------------- Setup Configuration -------------------")
    log_print(
        f"Subnetworks configuration: {args.subnets_number} batch: {args.batch_size} epochs: {args.epochs}")


def log_print(out, printing=True):
    logging.info(out)

    if printing:
        print(out)


def print_intermediate_activations(inputs, layer_number=None, print_hidden_feature=False, message=""):
    log_print(f"{layer_number} layer output shape: ") if layer_number else print(message)

    subnet_number = 0
    for input in inputs:
        print("{}% input shape: {}".format(int(pow(0.5, subnet_number) * 100), input.shape))

        if print_hidden_feature:
            tf.print(input, output_stream=sys.stdout, summarize=-1, sep=',')

        subnet_number += 1
