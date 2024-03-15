import tensorflow as tf


def setup_deterministic_computation(seed):
    tf.keras.utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()
    # tf.random.set_seed(seed)
    # np.random.seed(seed)
