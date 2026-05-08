# Hydroponic Lettuce Disease Classification

training and evaluating a deep learning model to classify diseases in hydroponic lettuce.

## Architecture

*   **Model**: MobileNetV2 (Transfer Learning)
*   **Framework**: TensorFlow/Keras
*   **Interface**: Gradio Web App

## Setup Instructions

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset Preparation:**
    Place your dataset in a `data/raw/` directory. The structure should be:
    ```
    data/
      raw/
        healthy/
          img1.jpg
          ...
        downy_mildew/
          ...
        powdery_mildew/
          ...
    ```

3.  **Training the Model:**
    ```bash
    python -m src.train
    ```
    This will save the best model to `models/best_model.keras` and training plots to the `outputs/` directory.

4.  **Evaluating the Model:**
    ```bash
    python -m src.evaluate
    ```
    This generates a classification report and confusion matrix in the `outputs/` directory.

5.  **Running the Web Interface:**
    ```bash
    python -m src.app
    ```
    This starts a local web server to test the model on new images with confidence scores.
