import tensorflow as tf
from src.config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, VALIDATION_SPLIT, SEED

def get_data_loaders():
    """
    Loads data from directory, applies train/val split, and sets up data augmentation.
    """
    print(f"Loading data from: {DATA_DIR}")
    
    # Check if data directory is empty
    import os
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise ValueError(f"Data directory {DATA_DIR} is empty. Please place your dataset folders inside.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical' # For multi-class classification
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = train_ds.class_names

    # Data Augmentation Layer to prevent overfitting
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(0.2, seed=SEED),
        tf.keras.layers.RandomZoom(0.2, seed=SEED),
    ], name="data_augmentation")

    # Optimize datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    # We apply augmentation ONLY to the training set by mapping it.
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    # Pre-fetching
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    # Quick test
    train, val, classes = get_data_loaders()
    print("Classes found:", classes)
