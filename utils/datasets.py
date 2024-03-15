import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def augment_mnist(image, label):
    image = tf.image.random_flip_left_right(image)
    # Random rotation by +/- 0.2 rad with p=0.3
    if tf.random.uniform([]) < 0.3:
        image = tfa.image.rotate(image, angles=tf.random.uniform([], minval=-0.2, maxval=0.2))

    # Random shift by 2 pixels in height and width
    orig = image.shape
    image = tf.pad(image, mode="SYMMETRIC",
                   paddings=tf.constant([[2, 2], [2, 2], [0, 0]]))
    image = tf.image.random_crop(image, size=orig)
    label = label.astype(tf.int64)
    return image, label

def get_dataset(dataset_name, batch_size=128, normalization=True):
    if dataset_name == "mnist":
        return get_mnist_train_test(batch_size)
    elif dataset_name == "fashion_mnist":
        batch_size = 32
        return get_fashion_mnist_train_test(batch_size)
    elif dataset_name == "cifar10":
        return get_cifar10_train_test(batch_size)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def preprocess_augment_cifar10(image, label):
    image = image.astype('float32') / 255.0
    image = (image - np.array((0.4914, 0.4822, 0.4465))) / np.array((0.2470, 0.2435, 0.2616))
    image = tf.image.random_flip_left_right(image)
    orig = image.shape
    image = tf.pad(image, mode="SYMMETRIC",
                   paddings=tf.constant([[2, 2], [2, 2], [0, 0]]))
    image = tf.image.random_crop(image, size=orig)
    label = label.astype("int64")
    # label = (label < 5).astype(np.uint8)
    return image, label

def normalize_cifar10(image, label):
    image = image.astype('float32') / 255.0
    image = (image - np.array((0.4914, 0.4822, 0.4465))) / np.array((0.2470, 0.2435, 0.2616))
    label = label.astype("int64")
    return image, label

def get_mnist_train_test(batch_size):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    y_test = y_test.astype("int32")
    y_train = y_train.astype("int32")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    train_ds = train_ds.map(augment_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds


def get_fashion_mnist_train_test(batch_size):
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    y_test = y_test.astype("int64")
    y_train = y_train.astype("int64")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    train_ds = train_ds.map(augment_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds


def get_cifar10_train_test(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train))
    train_ds = train_ds.map(preprocess_augment_cifar10, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(normalize_cifar10).batch(batch_size)
    return train_ds, test_ds
