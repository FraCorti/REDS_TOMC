import tensorflow as tf


def setup_deterministic_computation(seed):
    tf.keras.utils.set_random_seed(seed)
