import tensorflow as tf
from src.config import IMG_HEIGHT, IMG_WIDTH, LEARNING_RATE

def build_model(num_classes):
    """
    Builds a Transfer Learning model using MobileNetV2.
    MobileNetV2 is fast, accurate, and suitable for rapid training.
    """
    # 1. Base Model (Pre-trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers initially
    base_model.trainable = False

    # 2. Add custom classification head
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Preprocess inputs for MobileNetV2 (scales pixels to [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Regularization
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs, outputs)

    # 3. Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
