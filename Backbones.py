import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# CNN TYPE 01#

def CNN_1(input_shape=(224, 224, 3), name="backbone"):
    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x   = layers.Conv2D(32,  3, activation="relu", padding="same")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(64,  3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    x   = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D(2)(x)
    return Model(inp, x, name=name)

# CNN TYPE 02#

def CNN_2(input_shape=(224, 224, 3), name="backbone"):
    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    return Model(inp, x, name=name)


def CNN_3(input_shape=(224, 224, 3), name="backbone"):
    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    
    x = layers.Conv2D(32,  3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64,  3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    shortcut = x
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # same channels, no projection needed

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    shortcut = x
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.MaxPooling2D(2)(x)
    return Model(inp, x, name=name)

def CNN_4(input_shape=(224, 224, 3), name="backbone"):
    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    return Model(inp, x, name=name)


def CNN_5(input_shape=(224, 224, 3), name="backbone"):
    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x = layers.Conv2D(32, (2,2), padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(64, (2,2), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Conv2D(128, (2,2), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2,2)(x)
    # Note: no GlobalAveragePooling here — your attention layers expect a spatial output
    return Model(inp, x, name=name)


# MobileNetV2

def MobileNetV2_1(input_shape=(224, 224, 3), name="backbone"):
    base = tf.keras.applications.MobileNetV2(
        input_shape = input_shape,
        include_top = False,
        weights     = "imagenet"   # ← pretrained on ImageNet
    )

    base.trainable = False
    
    for layer in base.layers[-10:]:
        layer.trainable = True
    
    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x   = layers.Lambda(lambda x: mobilenet_preprocess(x * 255.0))(inp)
    x   = base(x, training=False)              # (batch, 7, 7, 1280)
    x   = layers.Conv2D(128, 1, padding="same",
                         name=f"{name}_proj")(x) # project to 128
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return Model(inp, x, name=name) 

 # EfficientNetB0   

def EfficientNetB0_1(input_shape=(224, 224, 3), name="backbone"):
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    for layer in base.layers[-50:]:
        layer.trainable = True

    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x   = layers.Lambda(lambda x: efficientnet_preprocess(x * 255.0))(inp)
    x   = base(x, training=False)
    x   = layers.Conv2D(128, 1, padding="same", name=f"{name}_proj")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    return Model(inp, x, name=name)

#DenseNet

def DenseNet_1(input_shape=(224, 224, 3), name="backbone"):
    base = tf.keras.applications.DenseNet121(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x   = layers.Lambda(lambda x: densenet_preprocess(x * 255.0))(inp)
    x   = base(x, training=False)
    x   = layers.Conv2D(128, 1, padding="same", name=f"{name}_proj")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return Model(inp, x, name=name)


def DenseNet169_1(input_shape=(224, 224, 3), name="backbone"):
    base = tf.keras.applications.DenseNet169(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x   = layers.Lambda(lambda x: densenet_preprocess(x * 255.0))(inp)
    x   = base(x, training=False)
    x   = layers.Conv2D(128, 1, padding="same", name=f"{name}_proj")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return Model(inp, x, name=name)


# ResNet50

def ResNet50_1(input_shape=(224, 224, 3), name="backbone"):
    base = tf.keras.applications.ResNet50(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x   = layers.Lambda(lambda x: resnet_preprocess(x * 255.0))(inp)
    x   = base(x, training=False)
    x   = layers.Conv2D(128, 1, padding="same", name=f"{name}_proj")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    return Model(inp, x, name=name)


BACKBONE_REGISTRY = {
    "cnn1": CNN_1,
    "cnn2": CNN_2,
    "cnn3": CNN_3,
    "cnn4": CNN_4,  
    "cnn5": CNN_5,
    "mobilenetv2":    MobileNetV2_1,
    "efficientnetb0": EfficientNetB0_1,
    "densenet121":    DenseNet_1,
    "densenet169": DenseNet169_1,
    "resnet50":       ResNet50_1,
}

def get_backbone(backbone, input_shape, name):
    return BACKBONE_REGISTRY[backbone](input_shape, name=name)