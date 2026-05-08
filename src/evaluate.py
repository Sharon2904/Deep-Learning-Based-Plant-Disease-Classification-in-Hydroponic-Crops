import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from src.config import MODEL_SAVE_DIR, OUTPUT_DIR, DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def main():
    # 1. Load Model and Classes
    model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.keras')
    class_names_path = os.path.join(MODEL_SAVE_DIR, 'class_names.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")

    model = tf.keras.models.load_model(model_path)
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # 2. Load Validation Data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2, # Must match config.py VALIDATION_SPLIT
        subset="validation",
        seed=42, # Must match config.py SEED
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True # Must be True so it splits exactly like train.py did!
    )

    # 3. Predict
    print("Running predictions on validation set...")
    y_true = []
    y_pred = []
    
    # Iterate exactly once to ensure y_true and y_pred stay perfectly aligned
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 4. Generate Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0)
    print(report)
    
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 5. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Evaluation artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
