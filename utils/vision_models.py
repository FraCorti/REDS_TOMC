import tensorflow as tf


def load_pre_trained_vision_model(model_size, dataset_name):
    if dataset_name == "mnist" or dataset_name == "fashion_mnist" or dataset_name == "cifar10":
        model = tf.keras.models.load_model(
            f"models/pretrained_vision_models/{dataset_name}_ds_cnn_size{model_size.upper()}.keras")
        return model
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
