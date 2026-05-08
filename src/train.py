import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from src.config import EPOCHS, MODEL_SAVE_DIR, OUTPUT_DIR, DATA_DIR
from src.data_loader import get_data_loaders
from src.model import build_model

def plot_history(history):
    """Plots and saves the training history for publication."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

def main():
    print("Initializing training pipeline...")
    
    # 1. Load Data
    train_ds, val_ds, class_names = get_data_loaders()
    num_classes = len(class_names)
    
    # Save class names for inference later
    with open(os.path.join(MODEL_SAVE_DIR, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

    # 2. Build Model
    model = build_model(num_classes)
    model.summary()

    # 3. Callbacks for robust training
    callbacks = [
        # Save the best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # Stop training early if it stops improving to save time and prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 4. Compute Class Weights for Imbalanced Data
    import numpy as np
    class_counts = {}
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        # Count images in each folder
        class_counts[i] = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    total_samples = sum(class_counts.values())
    class_weight = {}
    for i in range(num_classes):
        # Weight rare classes heavily, common classes lightly
        class_weight[i] = total_samples / (num_classes * max(1, class_counts[i]))
    
    print(f"Applying Class Weights to fix imbalance: {class_weight}")

    # 5. Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # 6. Plot results
    print("Training complete. Generating plots...")
    plot_history(history)
    print(f"Model saved to {MODEL_SAVE_DIR}/best_model.keras")

if __name__ == "__main__":
    main()
