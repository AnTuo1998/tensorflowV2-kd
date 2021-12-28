import tensorflow as tf
import tensorflow_datasets as tfds

def train_augment(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 36, 36)
    image = tf.image.resize_with_crop_or_pad(image, 32, 32)
    image = tf.image.random_flip_left_right(image)
    
    # transforms.Normalize((0.5071, 0.4867, 0.4408),
    #                      (0.2675, 0.2565, 0.2761)),
    return  image, label
    

def cifar(dataset, batch_size):
        train_ds, test_ds = tfds.load(dataset,
                                      split=['train', 'test'],
                                      as_supervised=True)
        train_ds.map(train_augment).shuffle().batch(batch_size)
        test_ds.batch(batch_size)
        return train_ds, test_ds