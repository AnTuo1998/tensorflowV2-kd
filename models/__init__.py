import tensorflow as tf
from distiller import Distiller

model_dict = {
    "resnet50": tf.keras.applications.resnet.ResNet50,
    "mobilenetv2": tf.keras.applications.mobilenet_v2.MobileNetV2,
}
